import os
import torch
from models import TEMPO, gpt2, LLaMA, Bert


class Exp_Basic_LLM(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'gpt2':    gpt2,
            'TEMPO':   TEMPO,
            'LLaMA': LLaMA,
            'Bert':Bert,
        }
        self.device = self._acquire_device()
        self.model = self._build_model()

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

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Using GPU: {device}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError