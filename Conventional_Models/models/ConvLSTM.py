import torch
import torch.nn as nn

class ConvLSTM_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout=0.5):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, seq_len, enc_in]

        x = x.transpose(1, 2)          # [B, enc_in, seq_len]
        x = self.conv1d(x)             # [B, d_model, seq_len]
        x = x.transpose(1, 2)          # [B, seq_len, d_model]

        out, (h_n, _) = self.lstm(x)   # h_n: [num_layers, B, d_model]
        out = self.layer_norm(out)

        return out, h_n[-1]            # last-layer hidden state


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.conv_lstm_block = ConvLSTM_Block(
            input_dim=configs.enc_in,
            hidden_dim=configs.d_model,
            kernel_size=configs.convlstm_kernel_size,
            num_layers=configs.e_layers,
            dropout=configs.dropout
        )

        # 🔑 Explicit prediction generation
        self.projection = nn.Linear(configs.d_model, configs.pred_len)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        _, last_hidden = self.conv_lstm_block(x)   # [B, d_model]
        out = self.projection(last_hidden)         # [B, pred_len]
        out = out.unsqueeze(-1)                    # [B, pred_len, 1]

        if self.task_name == 'Multivariate_forecasting':
            return out                             # [B, pred_len, 1]

        return out
