"""
NBEATSBackbone: N-BEATS (Neural Basis Expansion Analysis for Time Series) backbone.

Reference:
    Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).
    "N-BEATS: Neural basis expansion analysis for interpretable time series
    forecasting." International Conference on Learning Representations (ICLR).
    https://arxiv.org/abs/1905.10437

Implementation choices for this project:
    - Generic basis (not interpretable trend/seasonality decomposition): a
      stack of fully-connected blocks, each producing backcast residuals
      theta_b and forecast coefficients theta_f, with theta_b projected onto
      a polynomial basis {t, t^2, ..., t^theta_dim} over t in [0, 1].
    - Returns (B, hidden) features (last block's theta_f projected), suitable
      for downstream point / Gaussian / GMM heads used in this project's
      12-variant Phase 14 experiment.

Conventions matched across the project:
    All experiments in `experiments/` that need an NBEATS backbone import this
    module rather than redefining the class. As of v2.18, four experiment
    files (22, 27, 34, 36e) had near-identical inline copies; this module
    is the single source of truth.

Verification:
    `verification/verify_nbeats.py` constructs a `NBEATSBackbone` from this
    module and exercises shape, gradient flow, overfit sanity, and
    reproducibility. The four legacy inline copies in `experiments/` are
    kept untouched (they have slightly different docstrings / comments) so
    existing result JSONs in `results/` remain bit-reproducible. New code
    should import from this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NBEATSBackbone(nn.Module):
    """N-BEATS generic-basis backbone.

    A stack of `n_blocks` fully-connected blocks. Each block computes a
    backcast from the current residual via a polynomial basis, then subtracts
    that backcast from the residual. The final block's theta_f coefficients
    are projected to `hidden` features for downstream heads.

    Args:
        seq_len: input window length T. The polynomial basis t ∈ [0, 1] has
            `seq_len` evaluation points.
        hidden: width of the FC stack and the output feature dimension.
        n_blocks: number of doubly-residual NBEATS blocks.
        theta_dim: dimension of the polynomial basis and theta_b / theta_f
            coefficients per block.

    Input:
        x: tensor of shape (B, T, 1) — single-channel time series window.

    Output:
        features: tensor of shape (B, hidden) — last block's theta_f after a
            linear projection.
    """

    def __init__(
        self,
        seq_len: int,
        hidden: int = 64,
        n_blocks: int = 2,
        theta_dim: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.theta_dim = theta_dim

        # Per-block FC stack (4-layer MLP) + theta_b / theta_f heads.
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "fc": nn.Sequential(
                            nn.Linear(seq_len, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, hidden),
                            nn.ReLU(),
                        ),
                        "theta_b": nn.Linear(hidden, theta_dim),
                        "theta_f": nn.Linear(hidden, theta_dim),
                    }
                )
            )

        # Generic polynomial basis t^k for k = 1..theta_dim, evaluated at
        # `seq_len` evenly-spaced points in [0, 1].
        t_b = torch.linspace(0.0, 1.0, seq_len)
        self.register_buffer(
            "basis_b",
            torch.stack([t_b ** (i + 1) for i in range(theta_dim)], dim=-1),
        )  # (T, theta_dim)

        # Project last block's theta_f (theta_dim,) -> hidden-dim features.
        self.feat_proj = nn.Linear(theta_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) -> (B, T)
        x = x.squeeze(-1)
        residual = x
        last_theta_f: torch.Tensor | None = None

        for block in self.blocks:
            h = block["fc"](residual)              # (B, hidden)
            theta_b = block["theta_b"](h)            # (B, theta_dim)
            theta_f = block["theta_f"](h)            # (B, theta_dim)

            # backcast = sum_k theta_b[:, k] * basis_b[:, k]  along T axis
            #   = (theta_b[:, None, :] * basis_b[None, :, :]).sum(-1)
            #   shape (B, T)
            backcast = (theta_b.unsqueeze(1) * self.basis_b.unsqueeze(0)).sum(dim=-1)
            residual = residual - backcast
            last_theta_f = theta_f

        # Last block's theta_f projected to hidden features.
        return self.feat_proj(last_theta_f)


# ============================================================
# Heads — canonical point / Gaussian / GMM heads for use with
# NBEATSBackbone. Mirror the inline copies in experiments/22, 27, 34, 36e.
# ============================================================

class NBEATSPointHead(nn.Module):
    """Linear: hidden -> (B, horizon). Same projection as the inline engine."""

    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # (B, horizon)


class NBEATSGaussianHead(nn.Module):
    """Linear: hidden -> (B, horizon, 1, 2) -> (mu, sigma)."""

    def __init__(self, hidden: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(hidden, horizon * 2)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, 2)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma) + 1e-3
        return mu, sigma


class NBEATSGMMHead(nn.Module):
    """Linear: hidden -> (B, horizon, 1, n_mixtures, 3) -> (mu, sigma, pi)."""

    def __init__(self, hidden: int, horizon: int, n_mixtures: int = 4) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_mixtures = n_mixtures
        self.proj = nn.Linear(hidden, horizon * n_mixtures * 3)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, self.n_mixtures, 3)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        logit_pi = params[..., 2]
        sigma = F.softplus(log_sigma) + 1e-3
        pi = F.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


# ============================================================
# End-to-end forecaster (legacy / standalone; not used in Phase 14/15)
# Mirrors experiments/10_sota_baselines.py NBEATS + NBEATSBlock structure.
# ============================================================

class NBEATSForecaster(nn.Module):
    """Standalone N-BEATS with summed block forecasts and a fixed horizon.

    Each block produces a (B, horizon) forecast via a polynomial basis
    evaluated on the forecast grid; forecasts are summed across blocks.
    Useful when you want a self-contained N-BEATS model without a separate
    head (e.g., older experiments in exp 10 / 11 / 13 used this shape).

    Args:
        seq_len: input window length T
        horizon: forecast horizon H
        hidden:  block FC hidden dim
        n_blocks: number of stacked blocks
        theta_dim: number of basis coefficients per block
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        hidden: int = 64,
        n_blocks: int = 2,
        theta_dim: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_blocks = n_blocks
        self.theta_dim = theta_dim

        # 4-layer FC stacks — one per block
        self.fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                for _ in range(n_blocks)
            ]
        )
        self.theta_b = nn.ModuleList(
            [nn.Linear(hidden, theta_dim) for _ in range(n_blocks)]
        )
        self.theta_f = nn.ModuleList(
            [nn.Linear(hidden, theta_dim) for _ in range(n_blocks)]
        )

        # Backcast basis (input grid) and forecast basis (output grid)
        t_b = torch.linspace(0.0, 1.0, seq_len)
        t_f = torch.linspace(0.0, 1.0, horizon)
        self.register_buffer(
            "basis_b",
            torch.stack([t_b ** (i + 1) for i in range(theta_dim)], dim=-1),
        )  # (T, theta_dim)
        self.register_buffer(
            "basis_f",
            torch.stack([t_f ** (i + 1) for i in range(theta_dim)], dim=-1),
        )  # (H, theta_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x_in = x.squeeze(-1)
        else:
            x_in = x
        residual = x_in
        forecast = torch.zeros(
            x_in.shape[0], self.horizon, device=x_in.device, dtype=x_in.dtype
        )
        for i in range(self.n_blocks):
            h = self.fc[i](residual)
            theta_b = self.theta_b[i](h)
            theta_f = self.theta_f[i](h)
            backcast = (theta_b.unsqueeze(1) * self.basis_b.unsqueeze(0)).sum(dim=-1)
            this_fwd = (theta_f.unsqueeze(1) * self.basis_f.unsqueeze(0)).sum(dim=-1)
            residual = residual - backcast
            forecast = forecast + this_fwd
        return forecast.unsqueeze(-1)  # (B, H, 1)


__all__ = [
    "NBEATSBackbone",
    "NBEATSPointHead",
    "NBEATSGaussianHead",
    "NBEATSGMMHead",
    "NBEATSForecaster",
]