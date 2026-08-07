import numpy as np
import torch
import torch.nn as nn
import os
import time
import warnings
import psutil
import threading
import subprocess
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from numpy.random import choice
from omegaconf import OmegaConf
from accelerate import Accelerator

from data_provider.data_factory import data_provider
from exp.exp_basic_llm import Exp_Basic_LLM
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_LLM_Forecasting(Exp_Basic_LLM):
    def __init__(self, args, accelerator, config):
        super(Exp_LLM_Forecasting, self).__init__(args)
        self.task_name   = args.task_name
        self.accelerator = accelerator
        self.config      = config

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self):
        model_name = self.args.model
        if model_name not in self.model_dict:
            raise ValueError(f"Model '{model_name}' not supported. "
                             f"Available: {list(self.model_dict.keys())}")
        if model_name == 'TEMPO':
            model = self.model_dict[model_name].TEMPO(self.args).float()
        else:
            model = self.model_dict[model_name].Model(self.args).float()
        return model

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _set_dataset_args(self, dataset_name):
        """Overwrite args in-place from OmegaConf config for a single dataset."""
        cfg = self.config['datasets'][dataset_name]
        self.args.data      = cfg.data
        self.args.root_path = cfg.root_path
        self.args.data_path = cfg.data_path
        self.args.data_name = cfg.data_name
        self.args.features  = cfg.features
        self.args.freq      = cfg.freq if cfg.freq != 0 else 'h'
        self.args.target    = cfg.target

    def _build_multi_dataset_loaders(self):
        """
        Load all datasets listed in args.datasets, apply equal-sampling logic,
        and return concatenated train/val DataLoaders plus the test loader for
        args.target_data.
        """
        args = self.args

        train_datas    = []
        val_datas      = []
        min_sample_num = float('inf')

        # ---- first pass: collect val sets and find min_sample_num ----
        for dataset_name in args.datasets.split(','):
            self._set_dataset_args(dataset_name)
            train_data, _ = data_provider(args, 'train')
            if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
                min_sample_num = min(min_sample_num, len(train_data))
            vali_data, _ = data_provider(args, 'val')
            val_datas.append(vali_data)

        # ---- second pass: apply subsampling ----
        for dataset_name in args.datasets.split(','):
            self._set_dataset_args(dataset_name)
            train_data, _ = data_provider(args, 'train')

            if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1:
                train_data = Subset(train_data, choice(len(train_data), int(min_sample_num)))
            if args.electri_multiplier > 1 and args.equal == 1 and dataset_name == 'electricity':
                train_data = Subset(
                    train_data,
                    choice(len(train_data), int(min_sample_num * args.electri_multiplier))
                )
            if args.traffic_multiplier > 1 and args.equal == 1 and dataset_name == 'traffic':
                train_data = Subset(
                    train_data,
                    choice(len(train_data), int(min_sample_num * args.traffic_multiplier))
                )
            train_datas.append(train_data)

        # ---- concatenate across datasets ----
        if len(train_datas) > 1:
            train_data = torch.utils.data.ConcatDataset(train_datas)
            vali_data  = torch.utils.data.ConcatDataset(val_datas)
        else:
            train_data = train_datas[0]
            vali_data  = val_datas[0]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
        )
        vali_loader = torch.utils.data.DataLoader(
            vali_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )

        # ---- test loader for target_data ----
        self._set_dataset_args(args.target_data)
        test_data, test_loader = data_provider(args, 'test')

        return train_data, train_loader, vali_data, vali_loader, test_data, test_loader

    # ------------------------------------------------------------------
    # Optimizer / criterion
    # ------------------------------------------------------------------

    def _select_optimizer(self):
        trained_parameters = [p for p in self.model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # ------------------------------------------------------------------
    # Memory / profiling helpers
    # ------------------------------------------------------------------

    def get_memory_usage(self):
        process  = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)   # MB

    def get_gpu_memory_usage(self):
        try:
            gpu_usage = os.popen(
                'nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader'
            ).read()
            return int(gpu_usage.strip())
        except Exception:
            return 0

    def monitor_memory_usage(self, memory_usage_list, gpu_memory_usage_list, stop_event):
        while not stop_event.is_set():
            memory_usage_list.append(self.get_memory_usage())
            gpu_memory_usage_list.append(self.get_gpu_memory_usage())
            time.sleep(1)

    def get_gpu_info():
        try:
            gpu_info = subprocess.run(
                ['nvidia-smi'], stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
        except FileNotFoundError:
            gpu_info = ("nvidia-smi tool not found. "
                        "It may not be installed or it's not in your PATH.")
        return gpu_info

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, setting):
        args        = self.args
        accelerator = self.accelerator

        args.content = load_content(args)

        train_data, train_loader, vali_data, vali_loader, test_data, test_loader = \
            self._build_multi_dataset_loaders()

        # FIX: only main process creates checkpoint directory
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()

        time_now    = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        model_optim = self._select_optimizer()
        criterion   = self._select_criterion()
        mae_metric  = nn.L1Loss()

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=20, eta_min=1e-8
            )
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=args.pct_start,
                epochs=args.train_epochs,
                max_lr=args.learning_rate
            )

        # RAM_Usage measurement
        process    = psutil.Process()
        ram_before = process.memory_info().rss / 1024 ** 2
        print(f"RAM before {ram_before}")

        time_list             = []
        RAM_Usage_list        = []
        memory_usage_list     = []
        gpu_memory_usage_list = []
        stop_event            = threading.Event()
        monitor_thread        = threading.Thread(
            target=self.monitor_memory_usage,
            args=(memory_usage_list, gpu_memory_usage_list, stop_event)
        )
        monitor_thread.start()

        # training profiler accumulators
        data_time_sum      = 0.0
        fwd_time_sum       = 0.0
        bwd_time_sum       = 0.0
        step_count         = 0
        last_step_end_time = None

        # reset CUDA peak stats
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        # Prepare with accelerator
        (train_loader, vali_loader, test_loader,
         self.model, model_optim, scheduler) = accelerator.prepare(
            train_loader, vali_loader, test_loader,
            self.model, model_optim, scheduler
        )

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    seq_trend, seq_seasonal, seq_resid) in enumerate(train_loader):

                step_start_time = time.time()

                if last_step_end_time is None:
                    data_time = 0.0
                else:
                    data_time = step_start_time - last_step_end_time

                iter_count += 1
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(accelerator.device)
                batch_y      = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp], dim=1
                ).float().to(accelerator.device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_start = time.time()

                if self.args.model in ['TEMPO']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_end = time.time()

                f_dim   = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd_start = time.time()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now   = time.time()

                accelerator.backward(loss)
                model_optim.step()
                scheduler.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_opt_end = time.time()

                fwd_time_sum  += (t_fwd_end  - t_fwd_start)
                bwd_time_sum  += (t_opt_end  - t_bwd_start)
                data_time_sum += data_time
                step_count    += 1
                last_step_end_time = t_opt_end

                if args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # epoch-level wall-clock breakdown
            if step_count > 0:
                avg_data   = data_time_sum / step_count
                avg_fwd    = fwd_time_sum  / step_count
                avg_bwd    = bwd_time_sum  / step_count
                total_step = avg_data + avg_fwd + avg_bwd
                if total_step <= 0:
                    total_step = 1e-9
                print("---- Wall-clock (this epoch, per-step avg) ----")
                print(f"Data loader:        {avg_data:.6f} s  ({100.0*avg_data/total_step:5.1f}%)")
                print(f"Forward pass:       {avg_fwd:.6f} s  ({100.0*avg_fwd/total_step:5.1f}%)")
                print(f"Backward+Optimizer: {avg_bwd:.6f} s  ({100.0*avg_bwd/total_step:5.1f}%)")
                print("------------------------------------------------")

            train_loss = np.average(train_loss)

            # FIX 1: Synchronize vali/test loss across ALL ranks before
            # EarlyStopping so both ranks see the same value and make
            # the identical checkpoint decision — preventing NCCL deadlock.
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data,  test_loader,  criterion)

            accelerator.print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            # FIX 2: Barrier before AND after EarlyStopping so both ranks
            # enter save_checkpoint together — prevents rank divergence.
            accelerator.wait_for_everyone()
            early_stopping(vali_loss, self.model, path)
            accelerator.wait_for_everyone()

            if early_stopping.early_stop:
                accelerator.print("Early stopping")

                # training time measurement
                t1 = (time.time() - epoch_time)
                time_list.append(t1)
                average_time = sum(time_list) / len(time_list)

                # RAM usage
                ram_after = process.memory_info().rss / 1024 ** 2
                RAM_usage = ram_after - ram_before
                RAM_Usage_list.append(RAM_usage)
                RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)

                # Stop memory monitor thread
                stop_event.set()
                monitor_thread.join()

                average_memory_usage     = sum(memory_usage_list)     / len(memory_usage_list)
                average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)

                break

            if args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
            else:
                accelerator.print(
                    'Updating learning rate to {}'.format(scheduler.get_last_lr()[0])
                )

        # training time measurement (normal end)
        t1 = (time.time() - epoch_time)
        time_list.append(t1)
        average_time = sum(time_list) / len(time_list)

        # RAM usage
        ram_after = process.memory_info().rss / 1024 ** 2
        RAM_usage = ram_after - ram_before
        RAM_Usage_list.append(RAM_usage)
        RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)

        # FIX 3: Fixed indentation bug — stop monitor thread when training
        # completes normally (not just on early stop).
        if not stop_event.is_set():
            stop_event.set()
            monitor_thread.join()

        average_memory_usage     = sum(memory_usage_list)     / len(memory_usage_list)
        average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)

        print(f"epoch is {epoch}")

        # FIX 4: Only main process prints system stats to avoid duplicate output
        if accelerator.is_main_process:
            print('_______________________________________GPU Information_____________________________________')
            print(Exp_LLM_Forecasting.get_gpu_info())

            print('_______________________________________Efficiency and Running Time_____________________________________')
            print(f"| {'Metric':<40} | {'Value':>20} |")
            print("--------------------------------------------------------------------------------------------------------")
            print(f"| {'Average training time per epoch':<40} | {average_time:>20.4f} seconds |")
            print(f"| {'RAM before':<40} | {ram_before:>20.2f} MB |")
            print(f"| {'RAM after':<40} | {ram_after:>20.2f} MB |")
            print(f"| {'RAM usage (After -Before) per epoch':<40} | {RAM_usage:>20.2f} MB |")
            print(f"| {'Average memory usage':<40} | {average_memory_usage:>20.2f} MB |")
            print(f"| {'Average GPU memory usage':<40} | {average_gpu_memory_usage:>20.2f} MB |")
            print()

            print('_______________________________________CPU Information_____________________________________')
            print(f"| {'Metric':<40} | {'Value':>20} |")
            print("--------------------------------------------------------------------------------------------------------")
            physical_cores = psutil.cpu_count(logical=False)
            total_cores    = psutil.cpu_count(logical=True)
            cpu_usage      = psutil.cpu_percent(interval=1)
            print(f"| {'Physical cores':<40} | {physical_cores:>20} |")
            print(f"| {'Total cores':<40} | {total_cores:>20} |")
            print(f"| {'Total CPU Usage':<40} | {cpu_usage:>20.2f} % |")
            print("--------------------------------------------------------------------------------------------------------")

            try:
                print('_______________________________________GPU Memory (Peak/Reserved)_____________________________________')
                if torch.cuda.is_available():
                    device_idx    = torch.cuda.current_device()
                    props         = torch.cuda.get_device_properties(device_idx)
                    total_vram    = getattr(props, 'total_memory', 0) / (1024 ** 2)
                    peak_alloc    = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    try:
                        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                    except Exception:
                        peak_reserved = float('nan')
                    try:
                        curr_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    except Exception:
                        curr_reserved = float('nan')
                    print(f"| {'GPU total VRAM':<40} | {total_vram:>20.2f} MB |")
                    print(f"| {'Peak allocated (since reset)':<40} | {peak_alloc:>20.2f} MB |")
                    print(f"| {'Peak reserved (since reset)':<40} | {peak_reserved:>20.2f} MB |")
                    print(f"| {'Current reserved':<40} | {curr_reserved:>20.2f} MB |")
                else:
                    print("CUDA not available.")
                print("--------------------------------------------------------------------------------------------------------")
            except Exception:
                pass

            try:
                if step_count > 0:
                    overall_avg_data = data_time_sum / step_count
                    overall_avg_fwd  = fwd_time_sum  / step_count
                    overall_avg_bwd  = bwd_time_sum  / step_count
                    total_step       = overall_avg_data + overall_avg_fwd + overall_avg_bwd
                    if total_step <= 0:
                        total_step = 1e-9
                    print('_______________________________________Overall Wall-Clock Breakdown (Per-step Avg)_____________________')
                    print(f"| {'Data loader':<40} | {overall_avg_data:>10.6f} s | {100.0*overall_avg_data/total_step:>7.2f}% |")
                    print(f"| {'Forward pass':<40} | {overall_avg_fwd:>10.6f} s | {100.0*overall_avg_fwd/total_step:>7.2f}% |")
                    print(f"| {'Backward+Optimizer':<40} | {overall_avg_bwd:>10.6f} s | {100.0*overall_avg_bwd/total_step:>7.2f}% |")
                    print("--------------------------------------------------------------------------------------------------------")
            except Exception:
                pass

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            del_files(path)
            accelerator.print('success delete checkpoints')

        return self.model

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    seq_trend, seq_seasonal, seq_resid) in enumerate(vali_loader):

                batch_x      = batch_x.float().to(self.accelerator.device)
                batch_y      = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
                batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.accelerator.device)

                if self.args.model in ['TEMPO']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.accelerator.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        # FIX 5: Reduce loss across ALL ranks so every rank sees the
        # same value — this is the critical fix that prevents the
        # EarlyStopping divergence and subsequent NCCL deadlock.
        total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
        total_loss_tensor = self.accelerator.reduce(total_loss_tensor, reduction='mean')
        total_loss = total_loss_tensor.item()

        self.model.train()
        return total_loss

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds  = []
        trues  = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        # inference profiler accumulators
        inf_samples   = 0
        inf_batches   = 0
        inf_time_sum  = 0.0
        inf_latencies = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    seq_trend, seq_seasonal, seq_resid) in enumerate(test_loader):

                batch_x      = batch_x.float().to(self.accelerator.device)
                batch_y      = batch_y.float().to(self.accelerator.device)
                batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
                batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.accelerator.device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inf_start = time.time()

                if self.args.model in ['TEMPO']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inf_end     = time.time()
                batch_latency = t_inf_end - t_inf_start
                inf_time_sum += batch_latency
                inf_latencies.append(batch_latency)
                inf_batches  += 1
                try:
                    inf_samples += int(batch_x.size(0))
                except Exception:
                    pass

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.accelerator.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())

        preds  = np.array(preds)
        trues  = np.array(trues)
        inputx = np.array(inputx)
        preds  = preds.reshape(-1,  preds.shape[-2],  preds.shape[-1])
        trues  = trues.reshape(-1,  trues.shape[-2],  trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr, r2, adj_r2 = metric(
            preds, trues, n_features=preds.shape[-1]
        )

        # FIX 6: Only main process prints metrics and writes files —
        # prevents duplicate output and simultaneous file writes from
        # both DDP ranks corrupting result.txt and .npy files.
        if self.accelerator.is_main_process:
            print(f"MAE:{mae:.6f}, MSE:{mse:.6f}, RMSE:{rmse:.6f}, MAPE:{mape:.6f}, "
                  f"MSPE:{mspe:.6f}, RSE:{rse:.6f}, R2:{r2:.6f}, Adj_R2 {adj_r2:.6f}")

            f = open("result.txt", 'a')
            f.write(f"MAE:{mae:.6f}, MSE:{mse:.6f}, RMSE:{rmse:.6f}, MAPE:{mape:.6f}, "
                    f"MSPE:{mspe:.6f}, RSE:{rse:.6f}, R2:{r2:.6f}, Adj_R2:{adj_r2:.6f}\n")
            f.close()

            np.save(folder_path + 'pred.npy',  preds)
            np.save(folder_path + 'true.npy',  trues)
            np.save(folder_path + 'x.npy',     inputx)

            # Inference-time reporting
            try:
                avg_latency = inf_time_sum / inf_batches if inf_batches > 0 else float('nan')
                throughput  = inf_samples  / inf_time_sum if inf_time_sum > 0 else float('nan')

                if len(inf_latencies) > 0:
                    lat_sorted = sorted(inf_latencies)

                    def _pct(arr, p):
                        k = int(round((p / 100.0) * (len(arr) - 1)))
                        return arr[k]

                    p50 = _pct(lat_sorted, 50)
                    p95 = _pct(lat_sorted, 95)
                else:
                    p50 = p95 = float('nan')

                print('_______________________________________Inference-Time Results__________________________________________')
                print(f"| {'Avg latency per batch (forward only)':<55} | {avg_latency:>12.6f} s |")
                print(f"| {'p50 latency per batch':<55} | {p50:>12.6f} s |")
                print(f"| {'p95 latency per batch':<55} | {p95:>12.6f} s |")
                print(f"| {'Throughput (samples/sec)':<55} | {throughput:>12.2f} |")
                print("--------------------------------------------------------------------------------------------------------")
            except Exception:
                pass

        # FIX 7: Barrier at end of test so both ranks finish together
        # before the next experiment starts.
        self.accelerator.wait_for_everyone()

        return

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path            = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    seq_trend, seq_seasonal, seq_resid) in enumerate(pred_loader):

                batch_x      = batch_x.float().to(self.accelerator.device)
                batch_y      = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
                batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

                # decoder input
                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                ).float().to(batch_y.device)
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.accelerator.device)

                if self.args.model in ['TEMPO']:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save — only main process writes
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        if self.accelerator.is_main_process:
            np.save(folder_path + 'real_prediction.npy', preds)

        self.accelerator.wait_for_everyone()
        return