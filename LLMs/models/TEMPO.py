"""
TEMPO.py — adapted to work with:
  • data_loader that returns target-only univariate windows [B, seq_len, 1]
  • STL stamps that are all zeros (disabled in data_loader)
  • exp_forecasting_llm.py calling convention:
      outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                      seq_trend, seq_seasonal, seq_resid)
  • task_name = 'Multivariate_forecasting'
  • Multi-GPU (no global Accelerator instantiation)

FIX SUMMARY vs original TEMPO.py:
  FIX 1 — No global Accelerator(): original had none, confirmed clean.
  FIX 2 — squeeze() → squeeze(-1): original .squeeze() collapses B or M
           when batch_size=1 or M=1, causing shape errors.
  FIX 3 — in_layers now project patch_len → llm_dim (768) instead of
           d_model (32). GPT2 inputs_embeds must be 768-dim. Using
           configs.d_model=32 caused a silent shape mismatch.
  FIX 4 — out_layers updated to use llm_dim consistently.
  FIX 5 — Removed unused loss_local computation on zero STL stamps.
           (trend/season/noise from data_loader are all zeros — computing
           MSE against local decomposition is valid but adds overhead.)
  FIX 6 — Added task_name guard so 'Multivariate_forecasting' works.
  FIX 7 — gpt2_trend.wte access after LoRA wrapping uses base_model path.
  FIX 8 — GPT2 loaded with local_files_only fallback (same as gpt2.py).
  FIX 9 — num_nodes derived from enc_in (not hardcoded to 1) for
           generality, though data_loader always passes M=1.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from peft import get_peft_model, LoraConfig
from utils.rev_in import RevIn

import transformers
transformers.logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


def _print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[TEMPO] trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.2f}%)")


# ---------------------------------------------------------------------------
# TEMPO
# ---------------------------------------------------------------------------

class TEMPO(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.pred_len  = configs.pred_len
        self.seq_len   = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride     = configs.stride

        # FIX 9: derive num_nodes from enc_in; data_loader always gives M=1
        self.num_nodes = getattr(configs, 'enc_in', 1)

        # ── Patch settings ────────────────────────────────────────────
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1   # +1 for the padded patch

        # ── Local decomposition ───────────────────────────────────────
        kernel_size    = 25
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.map_trend  = nn.Linear(configs.seq_len, configs.seq_len)
        self.map_season = nn.Sequential(
            nn.Linear(configs.seq_len, 4 * configs.seq_len),
            nn.ReLU(),
            nn.Linear(4 * configs.seq_len, configs.seq_len)
        )
        self.map_resid  = nn.Linear(configs.seq_len, configs.seq_len)

        # ── GPT2 backbone ─────────────────────────────────────────────
        # FIX 8: local_files_only fallback identical to gpt2.py
        try:
            self.gpt2_trend = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                output_attentions=True,
                output_hidden_states=True,
            )
        except EnvironmentError:
            print("[TEMPO] Local GPT2 not found. Downloading...")
            self.gpt2_trend = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                output_attentions=True,
                output_hidden_states=True,
            )

        # Use only the first llm_layers transformer blocks
        self.gpt2_trend.h = self.gpt2_trend.h[:configs.llm_layers]

        # FIX 3: GPT2 inputs_embeds must be llm_dim (768), NOT d_model (32)
        # The original code used configs.d_model which is 32 in your config —
        # that causes a silent shape mismatch inside GPT2.
        self.llm_dim = configs.llm_dim   # 768

        # Tokenizer & fixed prompt tokens
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2', trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2', trust_remote_code=True, local_files_only=False)

        self.gpt2_trend_token    = self.tokenizer(
            text="Predict the future time step given the trend",    return_tensors="pt")
        self.gpt2_season_token   = self.tokenizer(
            text="Predict the future time step given the season",   return_tensors="pt")
        self.gpt2_residual_token = self.tokenizer(
            text="Predict the future time step given the residual", return_tensors="pt")
        self.token_len = len(self.gpt2_trend_token['input_ids'][0])

        # Freeze GPT2 (ln + wpe will be unfrozen below)
        for param in self.gpt2_trend.parameters():
            param.requires_grad = False
        for name, param in self.gpt2_trend.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True

        # LoRA
        lora_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.1, bias="lora_only"
        )
        self.gpt2_trend = get_peft_model(self.gpt2_trend, lora_config)
        _print_trainable(self.gpt2_trend)

        # ── Input projection: patch_len → llm_dim (FIX 3) ────────────
        self.in_layer_trend  = nn.Linear(configs.patch_len, self.llm_dim)
        self.in_layer_season = nn.Linear(configs.patch_len, self.llm_dim)
        self.in_layer_noise  = nn.Linear(configs.patch_len, self.llm_dim)

        # Prompt projection layers (project token embeds to llm_dim)
        self.prompt_layer_trend  = nn.Linear(self.llm_dim, self.llm_dim)
        self.prompt_layer_season = nn.Linear(self.llm_dim, self.llm_dim)
        self.prompt_layer_noise  = nn.Linear(self.llm_dim, self.llm_dim)

        # ── Output projection: llm_dim * (patch_num + token_len) → pred_len ─
        # FIX 4: use llm_dim (768) consistently
        out_in_dim = self.llm_dim * (self.patch_num + self.token_len)
        self.out_layer_trend  = nn.Linear(out_in_dim, configs.pred_len)
        self.out_layer_season = nn.Linear(out_in_dim, configs.pred_len)
        self.out_layer_noise  = nn.Linear(out_in_dim, configs.pred_len)

        # ── RevIN normalization ───────────────────────────────────────
        self.rev_in_trend  = RevIn(num_features=self.num_nodes)
        self.rev_in_season = RevIn(num_features=self.num_nodes)
        self.rev_in_noise  = RevIn(num_features=self.num_nodes)

        # pool is not used (set False)
        self.pool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_norm(self, x):
        """Normalize x to zero mean, unit variance."""
        means = x.mean(1, keepdim=True).detach()
        x     = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev
        return x, means, stdev

    def get_patch(self, x):
        """Patch the input: [B, L, M] → [(B*M), patch_num, patch_size]."""
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        return x

    def _get_prompt(self, token_ids, prompt_layer, n_repeat, device):
        """
        Embed fixed text tokens and project to llm_dim.
        FIX 7: access wte through base_model after LoRA wrapping.
        """
        try:
            # After get_peft_model, access via base_model
            wte = self.gpt2_trend.base_model.model.wte
        except AttributeError:
            wte = self.gpt2_trend.wte
        prompt_x = wte(token_ids.to(device))          # [1, token_len, llm_dim]
        prompt_x = prompt_x.repeat(n_repeat, 1, 1)    # [B*M, token_len, llm_dim]
        prompt_x = prompt_layer(prompt_x)
        return prompt_x

    def _get_emb(self, x, token_ids, prompt_layer):
        """Prepend prompt tokens to patch embeddings and return."""
        n_repeat = x.shape[0]
        prompt_x = self._get_prompt(token_ids, prompt_layer, n_repeat, x.device)
        return torch.cat((prompt_x, x), dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark,
                trend, season, noise):
        """
        Args:
            x            : [B, seq_len, M]  — input series (M=1 in our setup)
            batch_x_mark : [B, seq_len, time_features]  — not used by TEMPO
            dec_inp      : [B, label_len+pred_len, M]   — not used by TEMPO
            batch_y_mark : [B, label_len+pred_len, tf]  — not used by TEMPO
            trend        : [B, seq_len, M]  — zeros from data_loader
            season       : [B, seq_len, M]  — zeros from data_loader
            noise        : [B, seq_len, M]  — zeros from data_loader

        Returns:
            outputs      : [B, pred_len, M]
        """
        B, L, M = x.shape
        device   = x.device

        # ── RevIN normalise input ─────────────────────────────────────
        x = self.rev_in_trend(x, 'norm')

        # ── Local decomposition ───────────────────────────────────────
        # FIX 2: squeeze(-1) not squeeze() to avoid collapsing B or M
        trend_local  = self.moving_avg(x)
        trend_local  = self.map_trend(trend_local.squeeze(-1)).unsqueeze(-1)

        season_local = x - trend_local
        # squeeze/unsqueeze safely even when B=1
        season_local = self.map_season(
            season_local.squeeze(-1).unsqueeze(1)
        ).squeeze(1).unsqueeze(-1)

        noise_local  = x - trend_local - season_local

        # FIX 5: Skip loss_local — STL stamps are zeros so the auxiliary
        # loss is meaningless and wastes computation. The local decomposition
        # still runs (trend_local, season_local, noise_local) since they are
        # used as the actual patch inputs below.

        # ── Patch + embed ─────────────────────────────────────────────
        trend_patch  = self.get_patch(trend_local)    # [(B*M), patch_num, patch_size]
        season_patch = self.get_patch(season_local)
        noise_patch  = self.get_patch(noise_local)

        trend_emb  = self.in_layer_trend(trend_patch)   # [(B*M), patch_num, llm_dim]
        season_emb = self.in_layer_season(season_patch)
        noise_emb  = self.in_layer_noise(noise_patch)

        # ── Prepend prompt tokens ─────────────────────────────────────
        trend_emb  = self._get_emb(trend_emb,  self.gpt2_trend_token['input_ids'],
                                   self.prompt_layer_trend)
        season_emb = self._get_emb(season_emb, self.gpt2_season_token['input_ids'],
                                   self.prompt_layer_season)
        noise_emb  = self._get_emb(noise_emb,  self.gpt2_residual_token['input_ids'],
                                   self.prompt_layer_noise)

        # ── Single GPT2 forward on concatenated components ────────────
        x_all = torch.cat((trend_emb, season_emb, noise_emb), dim=1)
        x_out = self.gpt2_trend(
            inputs_embeds=x_all.to(torch.float32)
        ).last_hidden_state   # [(B*M), 3*(token_len+patch_num), llm_dim]

        # ── Slice output back into components ─────────────────────────
        seg = self.token_len + self.patch_num
        trend_out  = x_out[:, :seg,         :]   # [(B*M), seg, llm_dim]
        season_out = x_out[:, seg:2*seg,     :]
        noise_out  = x_out[:, 2*seg:3*seg,  :]

        # ── Output projection ─────────────────────────────────────────
        trend_out  = self.out_layer_trend(
            trend_out.reshape(B * M, -1))          # [(B*M), pred_len]
        season_out = self.out_layer_season(
            season_out.reshape(B * M, -1))
        noise_out  = self.out_layer_noise(
            noise_out.reshape(B * M, -1))

        trend_out  = rearrange(trend_out,  '(b m) l -> b l m', b=B)  # [B, pred_len, M]
        season_out = rearrange(season_out, '(b m) l -> b l m', b=B)
        noise_out  = rearrange(noise_out,  '(b m) l -> b l m', b=B)

        outputs = trend_out + season_out + noise_out   # [B, pred_len, M]

        # ── RevIN denormalise ─────────────────────────────────────────
        outputs = self.rev_in_trend(outputs, 'denorm')

        # FIX 6: task_name guard — slice to target column for
        # Multivariate_forecasting (consistent with gpt2.py)
        if self.task_name == 'Multivariate_forecasting':
            outputs = outputs[:, :, -1:]   # [B, pred_len, 1]

        return outputs