import torch
import torch.nn as nn

class GRU_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(GRU_Block, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [B, T, C]
        out, _ = self.gru(x)       # [B, T, d_model]
        out = self.layer_norm(out)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.gru_block = GRU_Block(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout
        )

        self.projection = nn.Linear(
            configs.d_model,
            configs.c_out
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """
        x: [B, seq_len, enc_in]
        returns:
            [B, pred_len, enc_in] or [B, pred_len, 1]
        """
        x = x.to(self.device)

        preds = []
        cur_x = x
        hidden = None

        for _ in range(self.pred_len):
            gru_out, hidden = self.gru_block.gru(cur_x, hidden)
            gru_out = self.gru_block.layer_norm(gru_out)

            next_step = self.projection(gru_out[:, -1:, :])  # [B, 1, enc_in]
            preds.append(next_step)

            # roll window
            cur_x = torch.cat([cur_x[:, 1:, :], next_step], dim=1)

        out = torch.cat(preds, dim=1)  # [B, pred_len, enc_in]

        if self.task_name == 'Multivariate_forecasting':
            out = out[:, :, -1:].contiguous()  # [B, pred_len, 1]

        return out

