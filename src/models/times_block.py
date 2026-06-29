"""
TimesBlock: Simplified 2D-variation block from TimesNet (ICLR 2024).

Core idea: reshape 1D time series into 2D (period x intra-period) representations
where each period is discovered by FFT. 2D convolutions then capture both
intra-period evolution and inter-period variation.

This implementation:
  - Reuses AMPD for robust period discovery (top-K peaks in FFT)
  - Uses 2D Inception-style conv (multi-kernel) per period
  - Supports multi-variate input (B, T, C) -> (B, H, C) forecast

Simplifications vs original TimesNet:
  - No embedding tokenizer; raw values used as channels
  - Single TimesBlock stack (not N stacked blocks) - for small datasets
  - AMPD used for period discovery (instead of raw FFT top-1)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ampd import AMPD


class Inception2D(nn.Module):
    """2D Inception-style block: parallel conv with kernel sizes 1, 3, 5."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = out_channels // 4
        c = max(c, 1)
        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c, kernel_size=5, padding=2)
        self.conv_extra = nn.Conv2d(in_channels, out_channels - 3 * c, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv1(x)
        b = self.conv3(x)
        c = self.conv5(x)
        d = self.conv_extra(x)
        return torch.cat([a, b, c, d], dim=1)


class TimesBlock(nn.Module):
    """
    Single TimesBlock: 1D -> 2D (per period) -> 2D Inception -> 1D -> residual.

    Args:
        seq_len: input window length T
        horizon: forecast horizon H
        n_channels: number of variates C
        top_k: number of top periods to use
        hidden: hidden channel size in 2D conv
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        n_channels: int = 1,
        top_k: int = 2,
        hidden: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.top_k = top_k
        self.periods: List[int] = []
        self.inceptions: nn.ModuleList = nn.ModuleList()
        self.hidden = hidden
        self.proj = nn.Linear(seq_len * n_channels, horizon * n_channels)

    def init_periods(self, periods: List[int]):
        self.periods = periods
        self.inceptions = nn.ModuleList(
            [Inception2D(self.n_channels, self.hidden) for _ in periods]
        )

    def fit(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 6):
        """Discover periods from training data using AMPD."""
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        amp = AMPD(top_k=self.top_k, max_period=max_period or self.seq_len, min_period=min_period)
        periods = amp.fit_discover(x_train[:, 0])
        self.init_periods([max(int(round(p)), min_period) for p in periods])

    def _reshape_to_2d(self, x: torch.Tensor, period: int) -> torch.Tensor:
        B, T, C = x.shape
        num_periods = (T + period - 1) // period
        pad_len = num_periods * period - T
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), mode="replicate")
        x = x.reshape(B, num_periods, period, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert C == self.n_channels, f"Expected n_channels={self.n_channels}, got {C}"

        outs = []
        for p, inception in zip(self.periods, self.inceptions):
            x2d = self._reshape_to_2d(x, p)
            y2d = inception(x2d)
            y2d = y2d.reshape(B, self.hidden, -1).permute(0, 2, 1)
            y_flat = y2d.mean(dim=-1)
            cur_len = y_flat.shape[1]
            if cur_len > T:
                y_flat = y_flat[:, :T]
            elif cur_len < T:
                y_flat = F.pad(y_flat, (0, T - cur_len), mode="replicate")
            outs.append(y_flat)

        if outs:
            agg = torch.stack(outs, dim=0).sum(dim=0)
        else:
            agg = x.squeeze(-1)
        agg = agg.unsqueeze(-1).expand(-1, -1, C)
        h = agg + x
        h_flat = h.reshape(B, -1)
        y = self.proj(h_flat)
        y = y.reshape(B, self.horizon, C)
        return y


class TimesNetLite(nn.Module):
    """
    Lightweight TimesNet-like model with AMPD-driven period discovery.
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        n_channels: int = 1,
        top_k: int = 2,
        hidden: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.block = TimesBlock(seq_len, horizon, n_channels, top_k, hidden)
        self.norm_in = nn.LayerNorm([seq_len, n_channels])
        self.norm_out = nn.LayerNorm([horizon, n_channels])

    def fit_periods(self, x_train: np.ndarray, max_period: Optional[int] = None, min_period: int = 6):
        self.block.fit(x_train, max_period=max_period, min_period=min_period)
        # Move newly-created inception modules to device
        self.block.inceptions = self.block.inceptions.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(x)
        y = self.block(x)
        y = self.norm_out(y)
        return y
