"""iTransformer backbone (NeurIPS 2023, ICLR 2024 spotlight).

Liu et al. 2024. https://arxiv.org/abs/2310.06625

iTransformer inverts the canonical Transformer: time steps are
projected to a hidden representation, then attention is applied
ACROSS the time dimension (instead of across feature/variate
dimensions as in the canonical Transformer). For multivariate
input this treats each variate as a token; for univariate input
the inversion is degenerate --- each time step is a token, which
is exactly what the canonical Transformer does.

For our univariate financial-return setting the iTransformer
adaptation is therefore equivalent to a vanilla Transformer
encoder over the time dimension. We implement it here as
``ITransformerBackbone`` with the standard iTransformer interface
``(B, T, C_in) -> (B, hidden)``.

The implementation differs from ``PatchTSTBackbone`` in two ways:
(i) no patching --- each time step is its own token, (ii) mean
pooling over the time dimension after the encoder (instead of
flatten + linear), which keeps the head's parameter count
comparable to the other backbones (~4K vs PatchTST's ~29K).
"""
import math
import torch
import torch.nn as nn


class ITransformerBackbone(nn.Module):
    """iTransformer backbone (NeurIPS 2023 / ICLR 2024 spotlight),
    univariate adaptation.

    Architecture:
        input (B, T, 1)
        -> per-time-step linear embed: (B, T, hidden)
        -> add learnable positional encoding
        -> TransformerEncoder (n_layers, n_heads, ff_dim)
        -> mean-pool over time: (B, hidden)
        -> linear head: (B, hidden)
    """
    def __init__(self, seq_len, hidden=64, n_heads=4, n_layers=3,
                 ff_dim=128, dropout=0.1, max_pos=1024):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        # Per-time-step linear embed (scalar -> hidden)
        self.embed = nn.Linear(1, hidden)
        # Learnable positional encoding (capped at max_pos for safety)
        pe = torch.zeros(max_pos, hidden)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_pos, hidden)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Head: project pooled hidden -> hidden (for symmetry with other backbones)
        self.head = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (B, T, 1)
        B, T, C = x.shape
        h = self.embed(x)                 # (B, T, hidden)
        h = h + self.pe[:, :T, :]         # add positional encoding
        h = self.encoder(h)              # (B, T, hidden)
        h = h.mean(dim=1)                # (B, hidden) — mean-pool over time
        h = self.norm(h)
        return self.head(h)              # (B, hidden)
