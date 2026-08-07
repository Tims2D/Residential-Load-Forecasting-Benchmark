import torch
import torch.nn as nn


class BiLSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(BiLSTM_Block, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # x: [B, T, C]
        out, (h_n, c_n) = self.lstm(x)   # out: [B, T, 2H]
        out = self.layer_norm(out)
        return out, (h_n, c_n)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers

        # ---------------- Encoder ----------------
        self.encoder = BiLSTM_Block(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=configs.dropout
        )

        # ---------------- Decoder ----------------
        # Encoder is bidirectional, so merged hidden state size = 2 * hidden_size
        self.decoder = nn.LSTM(
            input_size=self.c_out,
            hidden_size=self.hidden_size * 2,
            num_layers=self.num_layers,
            batch_first=True
        )

        # ---------------- Projection ----------------
        self.projection = nn.Linear(self.hidden_size * 2, self.c_out)

    def _merge_bidir_hidden(self, h):
        """
        Convert bidirectional hidden state from:
            [num_layers * 2, B, H]
        to:
            [num_layers, B, 2H]
        """
        num_layers_times_dirs, batch_size, hidden_size = h.shape
        num_layers = num_layers_times_dirs // 2

        h = h.view(num_layers, 2, batch_size, hidden_size)
        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=2)  # [num_layers, B, 2H]
        return h

    def forward(self, x):
        """
        x: [B, seq_len, enc_in]
        returns:
            [B, pred_len, c_out] normally
            [B, pred_len, 1] for Multivariate_forecasting
        """

        # -------------------------------------------------
        # Ensure model parameters are on same device as input
        # -------------------------------------------------
        if next(self.parameters()).device != x.device:
            self.to(x.device)

        # ================= Encoder =================
        enc_out, (h_n, c_n) = self.encoder(x)
        # enc_out: [B, T, 2H]
        # h_n, c_n: [num_layers * 2, B, H]

        # Merge forward/backward encoder states
        h_0 = self._merge_bidir_hidden(h_n)   # [num_layers, B, 2H]
        c_0 = self._merge_bidir_hidden(c_n)   # [num_layers, B, 2H]

        # Start token: last observed target variable(s)
        decoder_input = x[:, -1:, -self.c_out:]   # [B, 1, c_out]

        outputs = []

        # ================= Autoregressive decoding =================
        for _ in range(self.pred_len):
            dec_out, (h_0, c_0) = self.decoder(decoder_input, (h_0, c_0))
            # dec_out: [B, 1, 2H]

            step_out = self.projection(dec_out[:, -1, :])   # [B, c_out]
            outputs.append(step_out)

            # Feed prediction back to decoder
            decoder_input = step_out.unsqueeze(1)           # [B, 1, c_out]

        outputs = torch.stack(outputs, dim=1)               # [B, pred_len, c_out]

        # ================= Task formatting =================
        if self.task_name == 'Multivariate_forecasting':
            outputs = outputs[:, :, -1:].contiguous()       # [B, pred_len, 1]

        return outputs