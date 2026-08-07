import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from omegaconf import OmegaConf

from exp.exp_forecasting_llm import Exp_LLM_Forecasting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_init_config(config_path):
    return OmegaConf.load(config_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Time-LLM')

# Random seed
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, '
                         'imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='TEMPO',
                    help='model name, options: [Autoformer, DLinear, TimeLLM, TEMPO, '
                         'ST_TimeLLM_1, ST_TimeLLM_2, ST_TimeLLM_3]')

# Dataset
parser.add_argument('--datasets', type=str, default='ETTh1',
                    help='comma-separated list of training datasets')
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')

parser.add_argument('--target_data', type=str, default='ETTh1',
                    help='dataset used for validation / test')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task: M (multivariate→multivariate), '
                         'S (univariate→univariate), MS (multivariate→univariate)')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset loader type')
parser.add_argument('--freq', type=str, default='h',
                    help='time-feature encoding frequency: '
                         's/t/h/d/b/w/m or detailed like 15min / 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='location of model checkpoints')

# Forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')
parser.add_argument('--scale', type=str, default=True, help='whether to use normalize; True 1 False 0')
parser.add_argument('--timeenc', type=int, default=1, help='Type of time encoding: 0 for manual encoding, 1 for learned encoding')

# Model definition
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=768, help='dimension of FCN')
parser.add_argument('--factor', type=int, default=3, help='attention factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding: timeF / fixed / learned')
parser.add_argument('--activation', type=str, default='gelu', help='activation function')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--llm_model', type=str, default='GPT2',
                    help='LLM backbone: LLAMA / GPT2 / BERT')
parser.add_argument('--llm_dim', type=int, default=768,
                    help='LLM hidden dim (LLaMA-7B:4096, GPT2-small:768, BERT-base:768)')

# Optimization
parser.add_argument('--num_workers', type=int, default=10, help='DataLoader num_workers')
parser.add_argument('--itr', type=int, default=1, help='number of experiment repetitions')
parser.add_argument('--train_epochs', type=int, default=1, help='training epochs')
parser.add_argument('--align_epochs', type=int, default=1, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--eval_batch_size', type=int, default=8, help='evaluation batch size')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='experiment description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='learning rate adjustment strategy')
parser.add_argument('--pct_start', type=float, default=0.2, help='OneCycleLR pct_start')
parser.add_argument('--llm_layers', type=int, default=1, help='number of LLM layers to use')
parser.add_argument('--percent', type=int, default=100, help='percentage of training data')

# Alignment / decomposition flags
parser.add_argument('--output_attn_map', action='store_true',
                    help='output attention map of patches and prototype tokens')
parser.add_argument('--align_text', action='store_true', help='align text or not')
parser.add_argument('--align_trend', action='store_true', help='align trend or not')
parser.add_argument('--align_seasonal', action='store_true', help='align seasonal or not')
parser.add_argument('--align_residual', action='store_true', help='align residual or not')
parser.add_argument('--noise_anchors', action='store_true', help='use noise anchors')
parser.add_argument('--synonymous_anchors', action='store_true', help='use synonymous anchors')
parser.add_argument('--combination', type=str, default='late',
                    help='combine components before model: early / late')
parser.add_argument('--decomp_level', type=int, default=1,
                    help='decomposition level: 1=TimeLLM, 2=trend+seasonal, 3=trend+seasonal+residual')
parser.add_argument('--decomp_method', type=str, default='STL',
                    help='decomposition method for level 3: STL / TEMPO')

# Config / multipliers
parser.add_argument('--config_path', type=str, default='./configs/multiple_datasets.yml')
parser.add_argument('--electri_multiplier', type=int, default=1,
                    help='oversampling multiplier for electricity dataset')
parser.add_argument('--traffic_multiplier', type=int, default=1,
                    help='oversampling multiplier for traffic dataset')
parser.add_argument('--equal', type=int, default=1,
                    help='1: equal sampling across datasets, 0: no equal sampling')
#TEMPO
# ── TEMPO-specific args ──────────────────────────────────────────────
parser.add_argument('--prompt',     type=int,   default=1,    help='use prompt tokens in TEMPO')
parser.add_argument('--is_gpt',     type=int,   default=1,    help='use GPT2 backbone in TEMPO')
parser.add_argument('--cos',        type=int,   default=0,    help='use cosine annealing LR')
parser.add_argument('--tmax',       type=int,   default=20,   help='T_max for cosine annealing')
parser.add_argument('--decay_fac',  type=float, default=0.75, help='LR decay factor')
parser.add_argument('--stl_weight', type=float, default=0.01, help='STL auxiliary loss weight')
parser.add_argument('--pool',       type=int,   default=0,    help='use prompt pool in TEMPO')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')
parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple GPUs')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids for multi-GPU')

args = parser.parse_args()

# -------------------------------
# FORCE SINGLE-DATASET MODE
# -------------------------------
args.datasets = args.data
args.target_data = args.data

print("FINAL DATA CONFIG:")
print("data        =", args.data)
print("datasets    =", args.datasets)
print("target_data =", args.target_data)

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices   = args.devices.replace(' ', '')
    device_ids     = args.devices.split(',')
    args.device_ids = [int(i) for i in device_ids]
    args.gpu       = args.device_ids[0]

# ---------------------------------------------------------------------------
# Accelerator setup
# ---------------------------------------------------------------------------

os.environ['CURL_CA_BUNDLE']           = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF']  = 'max_split_size_mb:64'

ddp_kwargs        = DistributedDataParallelKwargs(find_unused_parameters=True)
#deepspeed_plugin  = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config = get_init_config(args.config_path)
accelerator.print(args)

# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

Exp = Exp_LLM_Forecasting

mses = []
maes = []

for ii in range(args.itr):
    setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.target_data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.d_ff,
        ii,
    )

    exp = Exp(args, accelerator, config)

    if args.is_training:
        accelerator.print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        accelerator.print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

    else:
        accelerator.print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

    torch.cuda.empty_cache()

accelerator.print('Average MSE: {:.7f} | Average MAE: {:.7f}'.format(
    np.average(mses), np.average(maes)
))