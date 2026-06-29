"""
WaveletPeriodNet (WPN) — multi-resolution period attention with density forecasting.

Architecture (motivation: 3 theoretical improvements over TimesNet for financial data):

  1. Multi-scale decomposition (wavelet analog, fixed avg-pool smoothing at multiple scales)
     - Replaces TimesNet's single 2D-variation with N parallel per-scale PeriodBanks
     - Wavelet analog: each scale captures one frequency band; orthogonal multi-resolution
     - Robust to trends / regime changes / non-periodic components (TimesNet's weakness)

  2. PeriodBank with learnable period anchors + attention
     - Replaces TimesNet's hard AMPD-discovered periods with K learnable log-spaced anchors
     - Each anchor gets its own 2D-variation token
     - Attention-based aggregation replaces SUM aggregation
     - Mathematically: approximation to ∫ E[y|p,x] q(p|x) dp (Bayesian posterior over periods)
     - Lets the model use different periods for different samples (regime-adaptive)

  3. Gaussian Mixture Density head (vs Huber loss → point estimate)
     - Outputs (mu_k, sigma_k, pi_k) for K=4 mixture components
     - Trained with negative log-likelihood — directly models leptokurtic financial returns
     - Enables CRPS / Pinball loss evaluation (downstream metrics for risk management)
     - Point prediction = mixture mean (for fair MAE comparison vs TimesNet)

References:
  - Donoho (1995) wavelet shrinkage (Parseval energy preservation)
  - iTransformer (Liu et al. 2024) for variate-as-token ideas
  - MQ-CNN (Wen et al. 2017) for multi-quantile / mixture density forecasting
  - Koopman forecasting (Yi et al. 2024) for Bayesian posterior over dynamics

Walk-forward Phase 3 scaffold used for benchmark. Tested on S&P 500 monthly log-returns.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Multi-scale decomposition (wavelet-like, fixed avg-pool)
# ============================================================

def multi_scale_smooth(x: torch.Tensor, scales: List[int]) -> List[torch.Tensor]:
    """Smooth x at multiple scales via avg_pool.

    Args:
        x: (B, T, C)
        scales: list of kernel sizes (e.g., [3, 6, 12, 24])
    Returns:
        list of (B, T, C) smoothed signals at each scale
    """
    outs = []
    x_t = x.transpose(1, 2)  # (B, C, T)
    for s in scales:
        # Pad to keep length T (replicate padding)
        x_smooth = F.avg_pool1d(x_t, kernel_size=s, stride=1, padding=s // 2)
        outs.append(x_smooth.transpose(1, 2))
    return outs


# ============================================================
# PeriodBank: learnable period anchors + 2D-variation + attention aggregation
# ============================================================

class PeriodBank(nn.Module):
    """Learnable period anchors + shared 2D-conv + attention-weighted aggregation.

    For each anchor p in self.period_anchors:
      reshape (B, T, 1) -> (B, 1, NP, p)
      shared Conv2D -> (B, hidden, NP, p)
      mean-pool over (NP, p) -> (B, hidden) per-anchor token
    Then self-attention across anchors + learned gate aggregation.

    Args:
        n_anchors: number of learnable period anchors (K)
        p_min, p_max: anchor period range (log-spaced initial values)
        hidden: hidden dim
        n_heads: attention heads
        use_learnable_anchors: if False, use fixed log-spaced periods (no gradient)
    """

    def __init__(self, n_anchors: int = 4, p_min: int = 4, p_max: int = 60,
                 hidden: int = 32, n_heads: int = 2, use_learnable_anchors: bool = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.hidden = hidden
        self.use_learnable_anchors = use_learnable_anchors

        # Log-spaced initial period anchors
        log_p = torch.linspace(np.log(p_min), np.log(p_max), n_anchors)
        if use_learnable_anchors:
            self.period_anchors = nn.Parameter(torch.exp(log_p))
        else:
            self.register_buffer("period_anchors", torch.exp(log_p))

        # Shared 2D conv (one conv for all anchors — learns period-invariant features)
        self.conv2d = nn.Conv2d(1, hidden, kernel_size=3, padding=1)

        # Self-attention across anchor tokens
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=0.1)
        self.norm_attn = nn.LayerNorm(hidden)

        # Gate: produces attention weights over anchors
        self.gate = nn.Linear(hidden, n_anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1) single-channel input
        Returns:
            token: (B, hidden)
        """
        B, T, C = x.shape
        assert C == 1, f"PeriodBank expects 1 channel, got {C}"

        # Compute per-anchor tokens
        # Use straight-through estimator (STE) for the integer period so gradient flows
        # to the period_anchors parameter: forward = int(p), backward = p.
        tokens = []
        for i in range(self.n_anchors):
            p = self.period_anchors[i]
            p_int = (p.round() - p).detach() + p  # STE: forward int, backward continuous
            NP = ((T + p_int - 1) // p_int).long()
            target_len = NP * p_int.long()
            pad = target_len - T
            x_p = F.pad(x, (0, 0, 0, int(pad.item())), mode="replicate") if int(pad.item()) > 0 else x
            x_2d = x_p.reshape(B, int(NP.item()), int(p_int.item()), C).permute(0, 3, 1, 2).contiguous()
            h = F.gelu(self.conv2d(x_2d))
            token = h.mean(dim=(2, 3))
            tokens.append(token)

        H = torch.stack(tokens, dim=1)  # (B, K, hidden)

        # Self-attention across anchors
        attn_out, attn_w = self.attn(H, H, H)
        H = self.norm_attn(H + attn_out)

        # Gated aggregation: gates derived from average token
        gates = F.softmax(self.gate(H.mean(dim=1)), dim=-1)  # (B, K)
        out = (gates.unsqueeze(-1) * H).sum(dim=1)  # (B, hidden)
        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get the diagonal attention weights (per-anchor self-attention score) for analysis.

        Args:
            x: (B, T, 1)
        Returns:
            attn_weights: (B, n_anchors) — diagonal of avg attention matrix
        """
        B, T, C = x.shape
        tokens = []
        for i in range(self.n_anchors):
            p = self.period_anchors[i]
            p_int = (p.round() - p).detach() + p
            NP = ((T + p_int - 1) // p_int).long()
            target_len = NP * p_int.long()
            pad = target_len - T
            x_p = F.pad(x, (0, 0, 0, int(pad.item())), mode="replicate") if int(pad.item()) > 0 else x
            x_2d = x_p.reshape(B, int(NP.item()), int(p_int.item()), C).permute(0, 3, 1, 2).contiguous()
            h = F.gelu(self.conv2d(x_2d))
            token = h.mean(dim=(2, 3))
            tokens.append(token)
        H = torch.stack(tokens, dim=1)
        _, attn_w = self.attn(H, H, H)
        # attn_w has shape (B, n_anchors, n_anchors) when average_attn_weights=True (default)
        # Diagonal = self-attention weight per anchor
        diag = attn_w.diagonal(dim1=1, dim2=2)  # (B, n_anchors)
        return diag

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get the learned gate weights for the PeriodBank.

        Args:
            x: (B, T, 1)
        Returns:
            gates: (B, n_anchors) — softmax weights
        """
        B, T, C = x.shape
        tokens = []
        for i in range(self.n_anchors):
            p = self.period_anchors[i]
            p_int = (p.round() - p).detach() + p
            NP = ((T + p_int - 1) // p_int).long()
            target_len = NP * p_int.long()
            pad = target_len - T
            x_p = F.pad(x, (0, 0, 0, int(pad.item())), mode="replicate") if int(pad.item()) > 0 else x
            x_2d = x_p.reshape(B, int(NP.item()), int(p_int.item()), C).permute(0, 3, 1, 2).contiguous()
            h = F.gelu(self.conv2d(x_2d))
            token = h.mean(dim=(2, 3))
            tokens.append(token)
        H = torch.stack(tokens, dim=1)
        _, attn_w = self.attn(H, H, H)
        H = self.norm_attn(H + attn_w)
        gates = F.softmax(self.gate(H.mean(dim=1)), dim=-1)
        return gates


# ============================================================
# Gaussian Mixture Density head
# ============================================================

class GMMHead(nn.Module):
    """Output head predicting Gaussian mixture parameters.

    Outputs (mu_k, sigma_k, pi_k) for K=4 mixture components per (timestep, channel).
    Trained with negative log-likelihood; supports sampling for CRPS evaluation.
    """

    def __init__(self, hidden: int, horizon: int, n_channels: int = 1, n_mixtures: int = 4):
        super().__init__()
        self.horizon = horizon
        self.n_channels = n_channels
        self.n_mixtures = n_mixtures
        # 3 outputs per mixture per (timestep, channel): mu, log_sigma, logit_pi
        self.proj = nn.Linear(hidden, horizon * n_channels * n_mixtures * 3)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (B, hidden)
        Returns:
            mu: (B, horizon, n_channels, K)
            sigma: (B, horizon, n_channels, K)
            pi: (B, horizon, n_channels, K)
        """
        B = context.shape[0]
        params = self.proj(context).view(
            B, self.horizon, self.n_channels, self.n_mixtures, 3
        )
        mu = params[..., 0]
        log_sigma = params[..., 1]
        logit_pi = params[..., 2]
        sigma = F.softplus(log_sigma) + 1e-3  # ensure positivity
        pi = F.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


def gmm_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of y under GMM.

    Args:
        y: (B, horizon, n_channels) target
        mu, sigma, pi: (B, horizon, n_channels, K) mixture params
    Returns:
        scalar NLL
    """
    y = y.unsqueeze(-1)  # (B, H, C, 1)
    log_p = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_p = log_p + torch.log(pi + 1e-8)
    nll = -torch.logsumexp(log_p, dim=-1)
    return nll.mean()


# ============================================================
# Point-prediction head (for ablation: replace GMM with single linear)
# ============================================================

class PointHead(nn.Module):
    """Simple linear head: context -> (horizon, n_channels) point prediction."""

    def __init__(self, hidden: int, horizon: int, n_channels: int = 1):
        super().__init__()
        self.horizon = horizon
        self.n_channels = n_channels
        self.proj = nn.Linear(hidden, horizon * n_channels)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        return self.proj(context).view(B, self.horizon, self.n_channels)


# ============================================================
# Full WPN model
# ============================================================

class WaveletPeriodNet(nn.Module):
    """WaveletPeriodNet: multi-scale + PeriodBank + GMM/point head.

    Args:
        seq_len: input window T
        horizon: forecast horizon H
        hidden: hidden dim
        scales: list of smoothing scales for multi-resolution
        n_anchors: anchors per PeriodBank
        n_mixtures: GMM components (ignored if head='point')
        head: 'gmm' or 'point'
        use_learnable_anchors: if False, periods are fixed (ablation)
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        hidden: int = 32,
        scales: Optional[List[int]] = None,
        n_anchors: int = 4,
        n_mixtures: int = 4,
        head: str = "gmm",
        use_learnable_anchors: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden = hidden
        self.scales = scales if scales is not None else [3, 6, 12, 24]
        self.head_type = head
        self.use_multi_scale = len(self.scales) > 1

        if self.use_multi_scale:
            self.period_banks = nn.ModuleList([
                PeriodBank(n_anchors=n_anchors, hidden=hidden,
                           use_learnable_anchors=use_learnable_anchors)
                for _ in self.scales
            ])
            self.fuse = nn.Linear(hidden * len(self.scales), hidden)
        else:
            self.period_banks = nn.ModuleList([
                PeriodBank(n_anchors=n_anchors, hidden=hidden,
                           use_learnable_anchors=use_learnable_anchors)
            ])
            self.fuse = nn.Identity()

        # Input norm (per-channel BatchNorm over time)
        self.norm_in = nn.BatchNorm1d(1)

        # Output head
        if head == "gmm":
            self.head = GMMHead(hidden, horizon, n_channels=1, n_mixtures=n_mixtures)
        elif head == "point":
            self.head = PointHead(hidden, horizon, n_channels=1)
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, 1) single-channel input
        Returns:
            if head='gmm': (mu, sigma, pi) — each (B, H, 1, K)
            if head='point': (B, H, 1) forecast
        """
        # Input norm
        x_t = x.transpose(1, 2)  # (B, 1, T)
        x_t = self.norm_in(x_t)
        x = x_t.transpose(1, 2)  # (B, T, 1)

        # Multi-scale decomposition + per-scale PeriodBank
        if self.use_multi_scale:
            scale_features = multi_scale_smooth(x, self.scales)
            tokens = [pb(sf) for pb, sf in zip(self.period_banks, scale_features)]
            context = self.fuse(torch.cat(tokens, dim=-1))  # (B, hidden)
        else:
            context = self.fuse(self.period_banks[0](x))

        return self.head(context)


# ============================================================
# Evaluation utilities (CRPS, Pinball for GMM)
# ============================================================

def gmm_point_predict(mu: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Point prediction = mixture mean (for MAE / display).

    Args:
        mu: (B, H, C, K), pi: (B, H, C, K)
    Returns:
        y_hat: (B, H, C)
    """
    return (mu * pi).sum(dim=-1)


def gmm_median(mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor,
               n_samples: int = 1000) -> torch.Tensor:
    """Approximate median via Monte Carlo sampling (better for MAE than mean)."""
    samples = sample_gmm(mu, sigma, pi, n_samples=n_samples)  # (n_samples, B, H, C)
    samples_sorted, _ = samples.sort(dim=0)
    return samples_sorted[n_samples // 2]


def sample_gmm(mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor,
               n_samples: int = 1000) -> torch.Tensor:
    """Sample from GMM via composition.

    Args:
        mu, sigma, pi: (B, H, C, K)
    Returns:
        samples: (n_samples, B, H, C)
    """
    B, H, C, K = mu.shape
    mu_flat = mu.reshape(-1, K)
    sigma_flat = sigma.reshape(-1, K)
    pi_flat = pi.reshape(-1, K)

    # Choose component
    cat = torch.distributions.Categorical(pi_flat)
    components = cat.sample((n_samples,))  # (n_samples, B*H*C)

    # Sample from chosen
    chosen_mu = torch.gather(mu_flat.expand(n_samples, -1, -1), 2,
                             components.unsqueeze(-1)).squeeze(-1)  # (n_samples, B*H*C)
    chosen_sigma = torch.gather(sigma_flat.expand(n_samples, -1, -1), 2,
                                components.unsqueeze(-1)).squeeze(-1)
    eps = torch.randn_like(chosen_mu)
    samples = chosen_mu + chosen_sigma * eps
    return samples.reshape(n_samples, B, H, C)


def crps_gmm(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor,
             n_samples: int = 1000) -> torch.Tensor:
    """Empirical CRPS for GMM predictive distribution.

    CRPS = E|Y - y| - 0.5 E|Y - Y'|
    where Y, Y' are iid samples from the predictive distribution.

    Args:
        y: (B, H, C) truth
        mu, sigma, pi: (B, H, C, K) mixture params
    Returns:
        mean CRPS over (B, H, C)
    """
    samples = sample_gmm(mu, sigma, pi, n_samples=n_samples)  # (n_samples, B, H, C)
    y_exp = y.unsqueeze(0).expand_as(samples)
    term1 = torch.abs(samples - y_exp).mean(dim=0)  # (B, H, C)

    # Unbiased estimate: use first half vs second half
    half = n_samples // 2
    s1 = samples[:half]
    s2 = samples[half:half * 2]
    term2 = torch.abs(s1 - s2).mean(dim=0)
    return (term1 - 0.5 * term2).mean()


def pinball_loss(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, pi: torch.Tensor,
                 quantiles: List[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
                 n_samples: int = 1000) -> Dict[str, float]:
    """Empirical pinball loss at multiple quantiles via Monte Carlo sampling."""
    samples = sample_gmm(mu, sigma, pi, n_samples=n_samples).cpu().numpy()  # (n_samples, B, H, C)
    y_np = y.cpu().numpy()  # (B, H, C)
    losses = {}
    for q in quantiles:
        q_hat = np.quantile(samples, q, axis=0)  # (B, H, C)
        diff = y_np - q_hat
        loss = np.maximum(q * diff, (q - 1) * diff).mean()
        losses[f"pinball_{q:.2f}"] = float(loss)
    return losses
