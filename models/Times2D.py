__all__ = ['Times2D']

from typing import Optional, Callable
import math
import numpy as np
from scipy.fft import rfft

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from layers.RevIN import RevIN
from layers.Conv_Blocks import Inception_Block_V1
from layers.encoders import TSTiEncoder, TSTEncoder, TSTEncoderLayer


def compute_derivative_heatmaps(x):
    """
    x: [B, T, N]
    returns: [B, N, T, 2]
    """
    first_derivative = x[:, 1:] - x[:, :-1]
    first_derivative = torch.cat(
        [torch.zeros_like(first_derivative[:, :1, :]), first_derivative], dim=1
    )

    second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
    second_derivative = torch.cat(
        [torch.zeros_like(second_derivative[:, :1, :]), second_derivative], dim=1
    )

    heatmap = torch.stack([first_derivative, second_derivative], dim=-1)
    heatmap = heatmap.permute(0, 2, 1, 3)  # [B, N, T, 2]
    return heatmap


class Times2DBackbone(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Times2DBackbone, self).__init__()

        # Load parameters from configs
        self.data = configs.data
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.add = configs.add
        self.affine = configs.affine
        self.head_dropout = configs.head_dropout
        self.subtract_last = configs.subtract_last
        self.n_layers = configs.e_layers
        self.wo_conv = configs.wo_conv
        self.serial_conv = configs.serial_conv
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.attn_dropout = configs.attn_dropout
        self.kwargs = kwargs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.revin_layer = RevIN(
            self.enc_in, affine=self.affine, subtract_last=self.subtract_last
        )

        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(self.seq_len, self.pred_len)

        self.dropout = configs.dropout
        self.act = 'gelu'
        self.norm = 'BatchNorm'
        self.key_padding_mask = 'auto'
        self.padding_var = None
        self.attn_mask = None
        self.res_attention = True
        self.pre_norm = False
        self.store_attn = False
        self.pe = 'zeros'
        self.learn_pe = True
        self.verbose = False

        # Period list
        self.period_list = [720, 360, 140, 70, 48]  # weather
        #self.period_list = [720, 360, 110, 96, 48]  # M4 yearly EETT (h and m)

        self.top_k = len(self.period_list)
        self.period_len = [math.ceil(self.seq_len / i) for i in self.period_list]

        # Kernel and stride lists
        self.kernel_list = [(n, self.patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = self.kernel_list

        # Dim and token lists
        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]

        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.enc_in, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.enc_in, num_kernels=configs.num_kernels
            ),
        )

        self.conv2D = nn.ModuleList([
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])

        self.head = Head(
            self.seq_len,
            self.top_k,
            self.pred_len,
            head_dropout=self.head_dropout,
            Concat=not self.add,
        )

        self.backbone = nn.ModuleList([
            nn.Sequential(
                TSTiEncoder(
                    self.enc_in,
                    patch_num=token,
                    patch_len=self.dim_list[i],
                    max_seq_len=self.seq_len,
                    n_layers=self.n_layers,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_k=None,
                    d_v=None,
                    d_ff=self.d_ff,
                    norm=self.norm,
                    attn_dropout=self.attn_dropout,
                    head_dropout=self.head_dropout,
                    dropout=self.dropout,
                    act=self.act,
                    key_padding_mask=self.key_padding_mask,
                    padding_var=self.padding_var,
                    attn_mask=self.attn_mask,
                    res_attention=self.res_attention,
                    pre_norm=self.pre_norm,
                    store_attn=self.store_attn,
                    pe=self.pe,
                    learn_pe=self.learn_pe,
                    verbose=self.verbose,
                    **self.kwargs
                ),
                nn.Flatten(start_dim=-2),
                nn.Linear(self.tokens_list[i] * self.d_model, self.seq_len)
                if self.tokens_list[i] * self.d_model != self.seq_len
                else nn.Identity(),
            )
            for i, token in enumerate(self.tokens_list)
        ])

        # Batch-independent learnable weights for 2 derivative channels
        self.weights = nn.Parameter(torch.randn(2))

        # Maps [B, T, N] -> [B, pred_len, N] by treating T as channels
        self.heatmap_to_pred = nn.Conv1d(
            in_channels=self.seq_len,
            out_channels=self.pred_len,
            kernel_size=1
        )

        self.to(self.device)

    def forward(self, x):
        """
        x: [B, N, T]
        """
        x = x.to(self.device)
        B, N, T = x.size()

        # [B, N, T] -> [B, T, N]
        x = x.permute(0, 2, 1)

        # Normalize
        x = self.revin_layer(x, 'norm')

        # Heatmap branch
        # compute_derivative_heatmaps expects [B, T, N] and returns [B, N, T, 2]
        heatmap = compute_derivative_heatmaps(x)                   # [B, N, T, 2]
        heatmap_features = self.conv(heatmap).permute(0, 2, 1, 3) # [B, T, N, 2]

        if self.data == 'm4':
            weights = torch.randn(B, T, N, 2, device=self.device)
        else:
            weights = self.weights.view(1, 1, 1, 2).expand(B, T, N, 2)

        Final_heatmap = torch.sum(heatmap_features * weights, dim=-1)  # [B, T, N]

        # Convert heatmap branch from seq_len -> pred_len
        # Conv1d takes [B, C, L], here C=seq_len and L=N
        Final_heatmap = self.heatmap_to_pred(Final_heatmap)            # [B, pred_len, N]
        Final_heatmap = Final_heatmap.permute(0, 2, 1)                 # [B, N, pred_len]

        # Main branch
        x = x.permute(0, 2, 1)  # [B, N, T]
        res = []

        for i, (period, (_, kernel_width)) in enumerate(zip(self.period_list, self.kernel_list)):
            if T % period != 0:
                pad1 = nn.ConstantPad1d((0, period - T % period), 0)
                padded_X = pad1(x)
                padded_X = padded_X.reshape(
                    padded_X.shape[0],
                    padded_X.shape[1],
                    padded_X.shape[2] // period,
                    period
                )
            else:
                padded_X = x.reshape(B, N, T // period, period)

            if period % kernel_width != 0:
                pad2 = nn.ConstantPad1d((0, kernel_width - period % kernel_width), 0)
                out = pad2(padded_X)
            else:
                out = padded_X  # [B, N, patch, periods]

            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
            out = out.unsqueeze(-3)  # [B*N, 1, patch, periods]
            out = self.conv2D[i](out)
            out = self.flatten(out)
            out = rearrange(out, '(b n) d p -> b n d p', b=B)
            glo = self.backbone[i](out)  # [B, N, T]

            res.append(glo)

        z = self.head(res)               # [B, N, pred_len]
        combined = z + Final_heatmap     # [B, N, pred_len]
        combined = combined.permute(0, 2, 1)   # [B, pred_len, N]
        combined = self.revin_layer(combined, 'denorm')
        combined = combined.permute(0, 2, 1)   # [B, N, pred_len]

        return combined


class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0, Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(
            context_window * (num_period if Concat else 1),
            target_window
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)

        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = Times2DBackbone(configs, **kwargs)

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        x = batch_x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x