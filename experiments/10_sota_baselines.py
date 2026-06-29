"""
Experiment 10: SOTA baseline benchmark on S&P 500 monthly log-returns.

Implements 5 strong baselines from the time-series forecasting literature:
  - DLinear (Zeng et al. AAAI 2023): simple linear + seasonal decomposition
  - N-BEATS (Oreshkin et al. ICLR 2020): basis expansion with backward/forward blocks
  - PatchTST (Nie et al. ICLR 2023): patched input + Transformer
  - iTransformer (Liu et al. ICLR 2024): inverted Transformer (variates as tokens)
  - TimesNet (Wu et al. ICLR 2024): 2D-variation block (the model we improve upon)

Plus our PeriodAttentionBlock (PAB) for direct comparison.

Walk-forward 70/30 split, 3 seeds for deep models, horizons 1/3/6/12.
"""
import sys
import json
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.ampd import AMPD
from src.models.period_attention import PeriodAttentionTimesNetLite

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] {DEVICE}")

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
TEST_FRAC = 0.30
HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]


# ============================================================
# Baseline models
# ============================================================

class DLinear(nn.Module):
    """DLinear: decompose input into trend + seasonal via moving average, then linear."""

    def __init__(self, seq_len, horizon, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # x: (B, T, 1)
        # Moving average trend
        x_t = x.transpose(1, 2)  # (B, 1, T)
        pad = (self.kernel_size - 1) // 2
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, self.kernel_size, stride=1)  # (B, 1, T)
        seasonal = x_t - trend  # (B, 1, T)
        trend = trend.squeeze(1)  # (B, T)
        seasonal = seasonal.squeeze(1)  # (B, T)
        y = self.linear_trend(trend) + self.linear_seasonal(seasonal)
        return y.unsqueeze(-1)


class NBEATSBlock(nn.Module):
    """One N-BEATS block: basis expansion + FC + backward/forward."""

    def __init__(self, seq_len, horizon, hidden=64, theta_dim=8, basis="trend"):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.basis = basis
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.theta_b = nn.Linear(hidden, theta_dim)
        self.theta_f = nn.Linear(hidden, theta_dim)
        # basis coefficients
        if basis == "trend":
            t_b = np.arange(seq_len) / seq_len
            t_f = np.arange(horizon) / horizon
            self.register_buffer("T_b", torch.tensor(t_b, dtype=torch.float32))
            self.register_buffer("T_f", torch.tensor(t_f, dtype=torch.float32))
        else:  # seasonality
            p = theta_dim // 2
            t_b = np.arange(seq_len) / seq_len
            t_f = np.arange(horizon) / horizon
            self.register_buffer("T_b", torch.tensor(t_b, dtype=torch.float32))
            self.register_buffer("T_f", torch.tensor(t_f, dtype=torch.float32))
            self.p = p

    def _basis_b(self, theta):
        # polynomial basis
        p = theta.shape[-1]
        basis = sum(theta[..., i:i+1] * (self.T_b ** i) for i in range(p))
        return basis

    def _basis_f(self, theta):
        p = theta.shape[-1]
        basis = sum(theta[..., i:i+1] * (self.T_f ** i) for i in range(p))
        return basis

    def forward(self, x):
        # x: (B, T)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        b = self._basis_b(theta_b)
        f = self._basis_f(theta_f)
        return b, f


class NBEATS(nn.Module):
    """N-BEATS with 2 stacked blocks (trend + seasonality)."""

    def __init__(self, seq_len, horizon, hidden=64, theta_dim=8, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(seq_len, horizon, hidden, theta_dim, basis=("trend" if i % 2 == 0 else "seasonality"))
            for i in range(n_blocks)
        ])

    def forward(self, x):
        # x: (B, T, 1)
        x_in = x.squeeze(-1)  # (B, T)
        residual_backcast = x_in
        forecast = torch.zeros(x.shape[0], self.blocks[0].horizon, device=x.device)
        for blk in self.blocks:
            b, f = blk(residual_backcast)
            residual_backcast = residual_backcast - b
            forecast = forecast + f
        return forecast.unsqueeze(-1)


class PatchTST(nn.Module):
    """Simplified PatchTST: patch input + per-patch linear + Transformer."""

    def __init__(self, seq_len, horizon, patch_len=8, stride=4, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        # Use non-overlapping patches for simplicity: n_patches = seq_len // patch_len
        self.n_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model * self.n_patches, horizon)

    def forward(self, x):
        # x: (B, T, 1)
        B, T, C = x.shape
        x = x.squeeze(-1)  # (B, T)
        # Non-overlapping patches
        n_patches = T // self.patch_len
        patches = x[:, :n_patches * self.patch_len].reshape(B, n_patches, self.patch_len)
        h = self.patch_proj(patches)  # (B, n_patches, d_model)
        h = h + self.pos_emb[:, :n_patches]
        h = self.encoder(h)
        h = h.reshape(B, -1)
        return self.head(h).unsqueeze(-1)


class iTransformer(nn.Module):
    """Simplified iTransformer: variates as tokens, attention across variates."""

    def __init__(self, seq_len, horizon, n_channels=1, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.proj_in = nn.Linear(seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_channels, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.proj_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (B, T, C) - variates as tokens
        h = x.transpose(1, 2)  # (B, C, T)
        h = self.proj_in(h)  # (B, C, d_model)
        h = h + self.pos_emb
        h = self.encoder(h)
        y = self.proj_out(h)  # (B, C, horizon)
        y = y.transpose(1, 2)  # (B, horizon, C)
        return y


class TimesNet(nn.Module):
    """Simplified TimesNet: 2D-variation block (FFT top-1 + 2D conv + flatten + linear)."""

    def __init__(self, seq_len, horizon, n_channels=1, top_k=2, hidden=64, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.top_k = top_k
        self.hidden = hidden
        self.periods = []
        # 2D Inception-like conv
        self.conv2d = nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        # proj input: hidden (we SUM across periods, not concat)
        self.proj = nn.Linear(hidden, horizon * n_channels)

    def fit_periods(self, x_train_np):
        amp = AMPD(top_k=self.top_k, max_period=min(360, self.seq_len), min_period=4)
        self.periods = [max(int(round(p)), 4) for p in amp.fit_discover(x_train_np)]

    def _reshape_2d(self, x, period):
        B, T, C = x.shape
        n_p = (T + period - 1) // period
        pad = n_p * period - T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        return x.reshape(B, n_p, period, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # x: (B, T, 1)
        B, T, C = x.shape
        if not self.periods:
            x_np = x.detach().squeeze(-1).cpu().numpy()
            self.fit_periods(x_np[0])
        agg = None
        for p in self.periods:
            x2d = self._reshape_2d(x, p)
            h2d = self.act(self.conv2d(x2d))  # (B, hidden, n_p, p)
            # Mean-pool over (n_p, p) to (B, hidden)
            h_pool = h2d.mean(dim=(2, 3))  # (B, hidden)
            agg = h_pool if agg is None else agg + h_pool
        if agg is None:
            agg = x.mean(dim=1)  # (B, 1)
        y = self.proj(agg)  # (B, horizon * C)
        return y.reshape(B, self.horizon, C)


# ============================================================
# Training utilities
# ============================================================

def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X), np.array(Y)


def train_model(model_class, X_train, Y_train, horizon, n_channels=1, seed=0, epochs=80, lr=1e-3, hidden=64, model_kwargs=None):
    """Generic training loop."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(-1)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
    n = X.shape[0]
    bs = min(128, n)

    if model_class is PeriodAttentionTimesNetLite:
        # PAB needs fit_periods
        kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels,
                                       top_k=2, hidden=hidden, num_heads=4, dropout=0.1)
        model = model_class(**kwargs)
        model.fit_periods(X_train[:, 0], max_period=360, min_period=4)
        model = model.to(DEVICE)
    else:
        # Different model classes have different signatures
        if model_class is DLinear:
            kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon)
        elif model_class is NBEATS:
            kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon, hidden=hidden)
        elif model_class is PatchTST:
            kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon, d_model=hidden, n_heads=4, n_layers=2)
        elif model_class is iTransformer:
            kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels, d_model=hidden)
        elif model_class is TimesNet:
            kwargs = model_kwargs or dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels, top_k=2, hidden=hidden)
        else:
            kwargs = model_kwargs or {}
        model = model_class(**kwargs).to(DEVICE)
        # TimesNet needs explicit fit_periods() to populate proj layer
        if model_class is TimesNet:
            model.fit_periods(X_train[:, 0])
            model.proj = model.proj.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss(delta=1.0)
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            yp = model(X[idx])
            loss = crit(yp, Y[idx])
            loss.backward()
            opt.step()
        sched.step()
    return model


def predict_model(model, X_test):
    model.eval()
    X = torch.tensor(X_test, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(-1)
    X = X.to(DEVICE)
    with torch.no_grad():
        yp = model(X).cpu().numpy().squeeze(-1)
    return yp


def predict_linear(X_train, Y_train, X_test):
    preds = []
    for k in range(Y_train.shape[1]):
        lr = LinearRegression()
        lr.fit(X_train, Y_train[:, k])
        preds.append(lr.predict(X_test))
    return np.array(preds).T


def evaluate(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 10: SOTA baselines on S&P 500 monthly log-returns")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values
    n = len(series)
    n_train = int(n * (1 - TEST_FRAC))
    print(f"[data] n={n}, n_train={n_train}")

    MODELS = {
        "Linear": None,
        "DLinear": DLinear,
        "NBEATS": NBEATS,
        "PatchTST": PatchTST,
        "iTransformer": iTransformer,
        "TimesNet": TimesNet,
        "PAB(ours)": PeriodAttentionTimesNetLite,
    }

    results = {}
    for h in HORIZONS:
        print(f"\n--- Horizon h = {h} ---")
        X_all, Y_all = make_supervised(series, SEQ_LEN, h)
        n_test_pairs = n - n_train
        X_train_p = X_all[:-n_test_pairs]
        Y_train_p = Y_all[:-n_test_pairs]
        X_test_p = X_all[-n_test_pairs:]
        Y_test_p = Y_all[-n_test_pairs:]

        h_results = {}
        for name, cls in MODELS.items():
            t0 = time.time()
            if cls is None:
                # Linear baseline
                yp = predict_linear(X_train_p, Y_train_p, X_test_p)
            else:
                preds = []
                for seed in SEEDS:
                    model = train_model(cls, X_train_p, Y_train_p, h, n_channels=1, seed=seed)
                    yp_seed = predict_model(model, X_test_p)
                    preds.append(yp_seed)
                yp = np.mean(preds, axis=0)
            mae, rmse = evaluate(Y_test_p, yp)
            t1 = time.time()
            h_results[name] = {"mae": mae, "rmse": rmse, "train_time_s": t1 - t0}
            print(f"  {name:<18}: MAE={mae:.5f} RMSE={rmse:.5f}  ({t1-t0:.1f}s)")
        results[f"h{h}"] = h_results

    # Summary
    print("\n" + "=" * 70)
    print(" SOTA benchmark: MAE on S&P 500 log-returns")
    print("=" * 70)
    header = f"{'Horizon':<8} "
    for name in MODELS.keys():
        header += f" {name[:10]:>10}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        row = f"  h={h:<4} "
        for name in MODELS.keys():
            row += f" {results[f'h{h}'][name]['mae']:>10.5f}"
        print(row)

    # Best per horizon
    print("\nBest model per horizon:")
    for h in HORIZONS:
        best = min(MODELS.keys(), key=lambda k: results[f"h{h}"][k]["mae"])
        print(f"  h={h}: {best}  MAE={results[f'h{h}'][best]['mae']:.5f}")

    out = RESULTS_DIR / "10_sota_baselines.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
