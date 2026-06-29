"""
Experiment 07: Multi-asset TimesBlock.

Use a multivariate input (8 channels: SP500_Return, DXY_Return, M2_Growth, FedFunds,
Copper_Return, Oil_WTI_Return, Wheat_Return, plus one more) and a TimesBlock that
operates channel-wise (n_channels > 1). Compare against single-asset TimesBlock.

Goal: determine if cross-asset periodicity helps forecast SP500 returns.
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
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.ampd import AMPD
from src.models.times_block import TimesBlock

warnings.filterwarnings("ignore")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
TEST_FRAC = 0.30
HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]
EPOCHS = 80
HIDDEN = 24

# Channels to use (all log-returns or growth, comparable scale)
CHANNELS = [
    "SP500_Return", "DXY_Return", "M2_Growth", "FedFunds",
    "Copper_Return", "Oil_WTI_Return", "Wheat_Return",
]


class MultiChannelTimesBlock(nn.Module):
    """TimesBlock with multi-channel input, but predicts only the target channel."""

    def __init__(self, seq_len, horizon, n_channels, top_k=2, hidden=24):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.top_k = top_k
        self.periods = []
        self.inceptions = nn.ModuleList()
        self.hidden = hidden
        # Output: only target channel
        self.proj = nn.Linear(seq_len, horizon)

    def init_periods(self, periods):
        self.periods = periods
        self.inceptions = nn.ModuleList(
            [nn.Conv2d(self.n_channels, self.hidden, kernel_size=3, padding=1) for _ in periods]
        )

    def fit(self, x_train, max_period=360, min_period=6):
        # Use first column (SP500_Return) for period discovery
        amp = AMPD(top_k=self.top_k, max_period=max_period, min_period=min_period)
        periods = amp.fit_discover(x_train[:, 0])
        self.init_periods([max(int(round(p)), min_period) for p in periods])

    def _reshape_2d(self, x, period):
        B, T, C = x.shape
        num_periods = (T + period - 1) // period
        pad = num_periods * period - T
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad), mode="replicate")
        return x.reshape(B, num_periods, period, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        B, T, C = x.shape
        outs = []
        for p, conv in zip(self.periods, self.inceptions):
            x2d = self._reshape_2d(x, p)
            y2d = conv(x2d)
            y_flat = y2d.mean(dim=1).reshape(B, -1)  # mean over hidden, flatten periods*p
            cur = y_flat.shape[1]
            if cur > T:
                y_flat = y_flat[:, :T]
            elif cur < T:
                y_flat = torch.nn.functional.pad(y_flat, (0, T - cur), mode="replicate")
            outs.append(y_flat)
        if outs:
            agg = torch.stack(outs, dim=0).sum(dim=0)
        else:
            agg = x[:, :, 0]
        h = agg + x[:, :, 0]  # residual only on target channel
        return self.proj(h).unsqueeze(-1)


class MultiChannelTimesNetLite(nn.Module):
    def __init__(self, seq_len=60, horizon=1, n_channels=1, top_k=2, hidden=24):
        super().__init__()
        self.block = MultiChannelTimesBlock(seq_len, horizon, n_channels, top_k, hidden)
        self.norm_in = nn.LayerNorm([seq_len, n_channels])
        self.norm_out = nn.LayerNorm([horizon, 1])

    def fit_periods(self, x_train, max_period=360, min_period=6):
        self.block.fit(x_train, max_period, min_period)

    def forward(self, x):
        x = self.norm_in(x)
        y = self.block(x)
        y = self.norm_out(y)
        return y


def make_multichannel_supervised(panel_df, channels, target_col, seq_len, horizon):
    """Build (X, Y) with X having all channels, Y having only target."""
    df = panel_df[channels].dropna()
    target = panel_df[target_col].reindex(df.index)
    X, Y = [], []
    vals = df.values
    tgt = target.values
    n = len(df)
    for t in range(n - seq_len - horizon + 1):
        X.append(vals[t:t + seq_len])
        Y.append(tgt[t + seq_len:t + seq_len + horizon])
    return np.array(X), np.array(Y)


def train_model(model_class, X_train, Y_train, horizon, n_channels, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
    model = model_class(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels, top_k=2, hidden=HIDDEN)
    if isinstance(model, MultiChannelTimesNetLite):
        model.fit_periods(X_train[:, :, 0])
    else:
        model.fit_periods(X_train[:, 0])
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    n = X.shape[0]
    bs = min(32, n)
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        return model(X).cpu().numpy().squeeze(-1)


def evaluate(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 70)
    print(" Experiment 07: Multi-asset TimesBlock on S&P 500 log-returns")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"[load] panel: {panel.shape}, date range: {panel.index.min()} to {panel.index.max()}")

    n = len(panel.dropna(subset=CHANNELS + ["SP500_Return"]))
    n_train = int(n * (1 - TEST_FRAC))

    results = {}
    for h in HORIZONS:
        print(f"\n--- Horizon h = {h} ---")
        X_all, Y_all = make_multichannel_supervised(panel, CHANNELS, "SP500_Return", SEQ_LEN, h)
        # Drop pairs with NaN
        valid = ~np.isnan(X_all).any(axis=(1, 2)) & ~np.isnan(Y_all).any(axis=1)
        X_all, Y_all = X_all[valid], Y_all[valid]
        n_total = X_all.shape[0]
        n_test = n_total - int(n_total * (1 - TEST_FRAC))
        X_tr, Y_tr = X_all[:-n_test], Y_all[:-n_test]
        X_te, Y_te = X_all[-n_test:], Y_all[-n_test:]
        print(f"  X_tr {X_tr.shape}, Y_tr {Y_tr.shape}, X_te {X_te.shape}")
        print(f"  Channels ({len(CHANNELS)}): {CHANNELS}")

        h_results = {}

        # Single-asset baseline
        t0 = time.time()
        preds = []
        for seed in SEEDS:
            from src.models.times_block import TimesNetLite
            model = train_model(TimesNetLite, X_tr[:, :, 0:1], Y_tr, h, n_channels=1, seed=seed)
            preds.append(predict(model, X_te[:, :, 0:1]))
        yp_single = np.mean(preds, axis=0)
        mae, rmse = evaluate(Y_te, yp_single)
        t1 = time.time()
        h_results["single"] = {"mae": mae, "rmse": rmse, "train_time_s": t1 - t0}
        print(f"  Single-asset   : MAE={mae:.5f} RMSE={rmse:.5f} ({t1-t0:.1f}s)")

        # Multi-asset (all channels)
        t0 = time.time()
        preds = []
        for seed in SEEDS:
            model = train_model(MultiChannelTimesNetLite, X_tr, Y_tr, h, n_channels=len(CHANNELS), seed=seed)
            preds.append(predict(model, X_te))
        yp_multi = np.mean(preds, axis=0)
        mae, rmse = evaluate(Y_te, yp_multi)
        t1 = time.time()
        h_results["multi"] = {"mae": mae, "rmse": rmse, "train_time_s": t1 - t0}
        print(f"  Multi-asset({len(CHANNELS)}) : MAE={mae:.5f} RMSE={rmse:.5f} ({t1-t0:.1f}s)")

        results[f"h{h}"] = h_results

    print("\n" + "=" * 70)
    print(" Multi-asset summary: MAE on S&P 500 log-returns")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Single':>10} {'Multi':>10} {'Delta':>10}")
    print("-" * 50)
    for h in HORIZONS:
        r = results[f"h{h}"]
        delta = r["multi"]["mae"] - r["single"]["mae"]
        sign = "↓" if delta < 0 else "↑"
        print(f"  h={h:<6} {r['single']['mae']:>10.5f} {r['multi']['mae']:>10.5f} {delta:>+10.5f} {sign}")

    out = RESULTS_DIR / "07_multiasset.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
