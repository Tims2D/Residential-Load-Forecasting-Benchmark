import torch
import torch.nn as nn

class ResLSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)   # out: [B, seq_len, d_model]
        out = self.layer_norm(out)
        out = out + self.residual(x)
        return out, h_n[-1]            # last layer hidden state


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.res_lstm = ResLSTM_Block(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout
        )

        # 🔑 Generate pred_len explicitly
        self.projection = nn.Linear(configs.d_model, configs.pred_len)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        _, last_hidden = self.res_lstm(x)     # [B, d_model]
        out = self.projection(last_hidden)    # [B, pred_len]
        out = out.unsqueeze(-1)               # [B, pred_len, 1]

        if self.task_name == 'Multivariate_forecasting':
            return out                        # [B, pred_len, 1]

        return out
