import math
import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.task_name = configs.task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        ).to(self.device)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]

        # normalization and permute: [B, S, C] -> [B, C, S]
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: [B, C, S] -> [B*C, seg_num_x, period_len] -> [B*C, period_len, seg_num_x]
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting: [B*C, period_len, seg_num_x] -> [B*C, period_len, seg_num_y]
        y = self.linear(x)

        # upsampling: [B*C, period_len, seg_num_y] -> [B*C, seg_num_y, period_len]
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.seg_num_y * self.period_len)

        # keep exact pred_len
        y = y[:, :, :self.pred_len]

        # permute and denorm: [B, C, pred_len] -> [B, pred_len, C]
        y = y.permute(0, 2, 1) + seq_mean

        if self.task_name == 'Multivariate_forecasting':
            y = y[:, :, -1:]

        return y