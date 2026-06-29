"""
PeriodAttentionBlock (PAB): a 2D-variation block with self-attention across periods.

Improvements over the original TimesBlock (times_block.py):
  - Replaces mean-pool over hidden channel with per-period tokenization + Transformer self-attention
  - Per-period tokens are learned jointly; the model decides which period matters
  - Returns attention weights for interpretability
  - Stacks residual + LayerNorm + FFN (standard Transformer block)
  - Outputs are concatenated across periods and projected to the forecast horizon

Reference architecture pattern: PerioGT (Nature Comput Sci 2025) — period-graph
attention — but simpler (no graph construction, just per-period self-attention).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ampd import AMPD


class PeriodAttentionBlock(nn.Module):
    """
    Per-period 2D-variation block with self-attention across periods.

    For one period p:
      Input  x2d: (B, C, num_periods, p)  [B=batch, C=channels]
      Step 1: Conv2D -> hidden channels
      Step 2: mean-pool over intra-period axis -> per-period tokens (B, num_periods, hidden)
      Step 3: self-attention across periods (B, num_periods, hidden)
      Step 4: residual + LayerNorm
      Step 5: FFN + residual + LayerNorm
      Output: tokens (B, num_periods, hidden), attn_weights (B, num_heads, num_periods, num_periods)
    """

    def __init__(self, n_channels: int, hidden: int = 32, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv2d = nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.hidden = hidden
        self.num_heads = num_heads

    def forward(self, x2d):
        """
        Args:
            x2d: (B, C, num_periods, p) input 2D representation
        Returns:
            tokens: (B, num_periods, hidden)
            attn_w: (B, num_heads, num_periods, num_periods)
        """
        B, C, NP, P = x2d.shape
        h = self.conv2d(x2d)  # (B, hidden, NP, P)
        # Per-period token: mean over intra-period axis (collapse P -> 1)
        tokens = h.mean(dim=-1).transpose(1, 2)  # (B, NP, hidden)
        # Self-attention across periods
        attn_out, attn_w = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens, attn_w


class PeriodAttentionTimesBlock(nn.Module):
    """
    Full 2D-variation model with period attention. Wraps one PeriodAttentionBlock per
    discovered period, aggregates by mean across periods, and projects to forecast horizon.

    Args:
        seq_len: input window length T
        horizon: forecast horizon H
        n_channels: number of variates C (1 for univariate, >1 for multi-asset)
        top_k: number of top periods (used when calling fit_periods with no data)
        hidden: hidden dim in 2D conv and self-attention
        num_heads: number of attention heads (must divide hidden)
        dropout: dropout rate in FFN
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        n_channels: int = 1,
        top_k: int = 2,
        hidden: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.top_k = top_k
        self.hidden = hidden
        self.num_heads = num_heads
        self.periods: List[int] = []
        self.blocks: nn.ModuleList = nn.ModuleList()
        # Output projection: concat per-period aggregated hidden -> horizon * C
        self.proj: Optional[nn.Linear] = None  # created after init_periods

    def init_periods(self, periods: List[int]):
        self.periods = periods
        self.blocks = nn.ModuleList([
            PeriodAttentionBlock(self.n_channels, self.hidden, self.num_heads) for _ in periods
        ])
        # proj: (num_periods * hidden) -> (horizon * n_channels)
        self.proj = nn.Linear(len(periods) * self.hidden, self.horizon * self.n_channels)

    def fit(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 6):
        """Discover periods from training data using AMPD."""
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        amp = AMPD(top_k=self.top_k, max_period=max_period or self.seq_len, min_period=min_period)
        periods = amp.fit_discover(x_train[:, 0])
        self.init_periods([max(int(round(p)), min_period) for p in periods])

    def fit_periods(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 6):
        self.fit(x_train, max_period, min_period)

    def _reshape_2d(self, x: torch.Tensor, period: int) -> torch.Tensor:
        B, T, C = x.shape
        num_periods = (T + period - 1) // period
        pad = num_periods * period - T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        return x.reshape(B, num_periods, period, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C)
        Returns:
            y: (B, H, C) forecast
            attn_weights_per_period: list of (B, num_heads, NP, NP) tensors
        """
        B, T, C = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert C == self.n_channels, f"Expected n_channels={self.n_channels}, got {C}"

        per_period_aggs = []
        attn_weights_all = []
        for p, block in zip(self.periods, self.blocks):
            x2d = self._reshape_2d(x, p)
            tokens, attn_w = block(x2d)  # tokens: (B, NP, hidden)
            # Aggregate: mean across periods (could also use learned query token)
            agg = tokens.mean(dim=1)  # (B, hidden)
            per_period_aggs.append(agg)
            attn_weights_all.append(attn_w)

        if per_period_aggs:
            combined = torch.cat(per_period_aggs, dim=-1)  # (B, num_periods * hidden)
        else:
            # Fallback: linear over flattened input
            combined = x.reshape(B, -1)

        assert self.proj is not None, "init_periods must be called before forward"
        y = self.proj(combined)
        y = y.reshape(B, self.horizon, C)
        return y, attn_weights_all


class PeriodAttentionTimesNetLite(nn.Module):
    """
    Lightweight wrapper with input/output LayerNorm, similar to TimesNetLite
    but using PeriodAttentionTimesBlock as the core.
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        n_channels: int = 1,
        top_k: int = 2,
        hidden: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.block = PeriodAttentionTimesBlock(seq_len, horizon, n_channels, top_k, hidden, num_heads, dropout)
        # Use per-channel BatchNorm1d (over time dim) instead of LayerNorm over [T, C]
        # LayerNorm over [T, C] normalizes across the whole window, killing periodic signal
        # (a sine wave has zero mean and approximately unit variance over a full period).
        self.norm_in = nn.BatchNorm1d(n_channels)
        self.norm_out = nn.Identity()  # output already well-scaled by training

    def fit_periods(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 6):
        self.block.fit(x_train, max_period=max_period, min_period=min_period)
        # Move ALL newly-created modules to device (init_periods creates new submodules)
        device = next(self.parameters()).device
        self.block.blocks = self.block.blocks.to(device)
        if self.block.proj is not None:
            self.block.proj = self.block.proj.to(device)
        # Also move the input norm (it was created in __init__ but uses learned params)
        self.norm_in = self.norm_in.to(device)
        self.norm_out = self.norm_out.to(device)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C) -> BatchNorm1d expects (B, C, T)
        x_t = x.transpose(1, 2)
        x_t = self.norm_in(x_t)
        x = x_t.transpose(1, 2)
        y, _attn = self.block(x)
        y = self.norm_out(y)
        return y
