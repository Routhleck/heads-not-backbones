"""
HybridPAB: Hybrid Period-Attention + TimesNet multi-scale block.

Motivation: PAB (period_attention.py) replaced TimesNet's SUM-aggregation across
periods with MEAN-aggregation + cross-period attention. That destroyed the
multi-scale signal — the very thing TimesNet does best. Walk-forward benchmarks
(Phase 3, Phase 4) confirmed: PAB wins h=1 only, loses to TimesNet at h>=3 and
multivariate.

Architecture:
  Per discovered period p:
    1. Reshape (B, T, C) -> (B, C, NP, p)  [2D-variation, same as TimesBlock]
    2. Conv2D -> (B, hidden, NP, p)
    3. Within-period self-attention over the intra-period axis (p axis)
       [captures intra-period structure PAB was supposed to capture]
    4. Mean-pool over intra-period axis -> (B, NP, hidden) per-period token

  Cross-period fusion:
    5. Concatenate per-period tokens (zero-padded for varying NP_p)
    6. Cross-period self-attention + residual + LN + FFN + LN
       [learned fusion across periods, attention-based aggregation]
    7. Mean over token axis -> (B, hidden) cross-period summary

  Multi-scale aggregation (TimesNet-style):
    8. Sum the per-period mean-pooled vectors across periods -> (B, hidden)
       [this is what TimesNet does that PAB dropped]

  Output:
    9. Concat [multi_scale, cross_summary] -> Linear -> (B, horizon, output_channels)
       [output_channels=1 for multivariate-to-univariate forecasting]

Multi-channel AMPD: discover periods per channel, union (dedup), use as period set.
This addresses Phase 4's "AMPD only on target channel" issue.

Single-target output: predict only the target channel (output_channels=1) to
avoid wasting capacity on irrelevant outputs in multivariate settings.

References:
  - TimesNet (Wu et al. ICLR 2024) for 2D-variation and multi-scale SUM aggregation
  - PAB (period_attention.py in this repo) for period-attention inspiration
  - PerioGT (Nature Comput Sci 2025) for period-aware attention design
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ampd import AMPD


class HybridPeriodBlock(nn.Module):
    """Per-period block: 2D conv -> within-period self-attention -> mean-pool.

    For one period p:
      Input  x2d: (B, C, num_periods, p)
      Step 1: Conv2D -> hidden channels
      Step 2: within-period self-attention over the intra-period axis (length p)
      Step 3: residual + LN
      Step 4: FFN + residual + LN
      Step 5: mean-pool over intra-period axis -> per-period token (B, num_periods, hidden)
    """

    def __init__(self, n_channels: int, hidden: int = 32, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.conv2d = nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1)
        self.attn_within = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout)
        self.norm_within1 = nn.LayerNorm(hidden)
        self.ffn_within = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm_within2 = nn.LayerNorm(hidden)

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x2d: (B, C, num_periods, p)
        Returns:
            tokens: (B, num_periods, hidden) per-period tokens
        """
        B, C, NP, P = x2d.shape
        h = self.conv2d(x2d)  # (B, hidden, NP, p)
        # Treat each period's p timesteps as a sequence; reshape for attention
        # (B, NP, p, hidden) -> (B*NP, p, hidden)
        h = h.permute(0, 2, 3, 1).reshape(B * NP, P, self.hidden)
        attn_out, _ = self.attn_within(h, h, h)
        h = self.norm_within1(h + attn_out)
        h = self.norm_within2(h + self.ffn_within(h))
        # Mean-pool over intra-period axis -> per-period token
        tokens = h.mean(dim=1)  # (B*NP, hidden)
        tokens = tokens.reshape(B, NP, self.hidden)
        return tokens


class HybridPeriodTimesBlock(nn.Module):
    """Full HybridPAB block: per-period blocks + cross-period attention + multi-scale SUM aggregation.

    Args:
        seq_len: input window length T
        horizon: forecast horizon H
        n_channels: number of input variates C
        hidden: hidden dim in 2D conv and attention
        num_heads: number of attention heads (must divide hidden)
        dropout: dropout rate
        output_channels: number of forecast channels (1 for univariate or multi-to-uni; C for multi-to-multi)
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        n_channels: int = 1,
        hidden: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
        output_channels: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.hidden = hidden
        self.num_heads = num_heads
        self.output_channels = output_channels
        self.periods: List[int] = []
        self.blocks: nn.ModuleList = nn.ModuleList()

        # Cross-period attention (operates on concatenated per-period tokens)
        self.attn_cross = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout)
        self.norm_cross1 = nn.LayerNorm(hidden)
        self.ffn_cross = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm_cross2 = nn.LayerNorm(hidden)

        # Projection: concat [multi_scale (TimesNet SUM), cross_summary (PAB)] -> horizon * output_channels
        self.proj: Optional[nn.Linear] = None

    def init_periods(self, periods: List[int]):
        self.periods = periods
        self.blocks = nn.ModuleList([
            HybridPeriodBlock(self.n_channels, self.hidden, self.num_heads) for _ in periods
        ])
        self.proj = nn.Linear(2 * self.hidden, self.horizon * self.output_channels)

    def fit(self, x_train: np.ndarray, max_period: Optional[int] = None,
            min_period: int = 4, top_k: int = 2, max_periods: int = 6):
        """Multi-channel AMPD: discover periods per channel, union dedup, cap at max_periods.

        On real data with top_k=2 per channel, 5 channels typically yield ~6-10
        candidate periods; dedup usually gives 4-8 distinct. We cap at max_periods
        to avoid overfitting on small training sets (e.g. 70-340 pairs in walk-forward).
        """
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        n_ch = x_train.shape[1]
        all_periods = []
        for c in range(n_ch):
            amp = AMPD(top_k=top_k, max_period=max_period or self.seq_len, min_period=min_period)
            ps = amp.fit_discover(x_train[:, c])
            all_periods.extend([max(int(round(p)), min_period) for p in ps])
        # Dedup and sort ascending
        all_periods = sorted(set(all_periods))
        if not all_periods:
            all_periods = [min_period]
        # Cap at max_periods; keep the largest (longer periods tend to dominate)
        if len(all_periods) > max_periods:
            # Take a mix of small/medium/large periods
            step = max(1, len(all_periods) // max_periods)
            all_periods = all_periods[::step][:max_periods]
        self.init_periods(all_periods)

    def _reshape_2d(self, x: torch.Tensor, period: int) -> torch.Tensor:
        B, T, C = x.shape
        NP = (T + period - 1) // period
        pad = NP * period - T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        return x.reshape(B, NP, period, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            y: (B, horizon, output_channels)
        """
        B, T, C = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert C == self.n_channels, f"Expected n_channels={self.n_channels}, got {C}"

        # Per-period: 2D conv + within-period attention + per-period tokens
        per_period_tokens = []
        for p, block in zip(self.periods, self.blocks):
            x2d = self._reshape_2d(x, p)
            tokens = block(x2d)  # (B, NP_p, hidden)
            per_period_tokens.append(tokens)

        # Multi-scale aggregation (TimesNet-style SUM)
        # Each per-period token is (B, NP_p, hidden); mean over NP_p -> (B, hidden)
        per_period_aggs = [t.mean(dim=1) for t in per_period_tokens]
        multi_scale = torch.stack(per_period_aggs, dim=0).sum(dim=0)  # (B, hidden)

        # Cross-period attention summary (PAB-style learned fusion)
        # Concatenate per-period tokens; zero-pad to max NP_p
        max_NP = max(t.shape[1] for t in per_period_tokens)
        padded = []
        for t in per_period_tokens:
            NP = t.shape[1]
            if NP < max_NP:
                pad = torch.zeros(B, max_NP - NP, self.hidden, device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        all_tokens = torch.cat(padded, dim=1)  # (B, sum_NP, hidden)
        attn_out, _ = self.attn_cross(all_tokens, all_tokens, all_tokens)
        h = self.norm_cross1(all_tokens + attn_out)
        h = self.norm_cross2(h + self.ffn_cross(h))
        cross_summary = h.mean(dim=1)  # (B, hidden)

        # Combine: TimesNet multi-scale (SUM) + PAB cross-attention summary
        combined = torch.cat([multi_scale, cross_summary], dim=-1)  # (B, 2*hidden)
        assert self.proj is not None, "init_periods must be called before forward"
        y = self.proj(combined)
        y = y.reshape(B, self.horizon, self.output_channels)
        return y


class HybridPAB(nn.Module):
    """Lightweight wrapper with per-channel BatchNorm1d input norm.

    Args:
        seq_len: input window length T
        horizon: forecast horizon H
        n_channels: number of input variates C
        hidden: hidden dim
        num_heads: attention heads
        dropout: dropout rate
        output_channels: number of forecast channels (1 for univariate target)
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        n_channels: int = 1,
        hidden: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
        output_channels: Optional[int] = None,
        top_k: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        if output_channels is None:
            output_channels = n_channels
        self.output_channels = output_channels
        self.block = HybridPeriodTimesBlock(
            seq_len, horizon, n_channels, hidden, num_heads, dropout, output_channels
        )
        # Per-channel BatchNorm1d over time dim (preserves periodicity, unlike LayerNorm over [T,C])
        self.norm_in = nn.BatchNorm1d(n_channels)

    def fit_periods(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 4,
                    top_k: int = 2):
        self.block.fit(x_train, max_period=max_period, min_period=min_period, top_k=top_k)
        # Move newly-created modules to device
        device = next(self.parameters()).device
        self.block.blocks = self.block.blocks.to(device)
        if self.block.proj is not None:
            self.block.proj = self.block.proj.to(device)
        self.block.attn_cross = self.block.attn_cross.to(device)
        self.block.norm_cross1 = self.block.norm_cross1.to(device)
        self.block.norm_cross2 = self.block.norm_cross2.to(device)
        self.block.ffn_cross = self.block.ffn_cross.to(device)
        self.norm_in = self.norm_in.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> BatchNorm1d expects (B, C, T)
        x_t = x.transpose(1, 2)
        x_t = self.norm_in(x_t)
        x = x_t.transpose(1, 2)
        return self.block(x)
