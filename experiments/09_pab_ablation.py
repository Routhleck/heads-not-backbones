"""
Experiment 09: Real-data ablation with the new PeriodAttentionBlock.

Goal: verify that the 4 theory prior sets NOW produce different MAE
(vs v1 TimesBlock where all 4 priors gave identical MAE because the
linear projection dominated).

If PAB differentiates priors: 4 priors give different MAE, with the
best matching the data (probably Kondratiev-Juglar or None).
If PAB still does not: need to investigate further.
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
from src.models.period_attention import PeriodAttentionTimesNetLite

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
EPOCHS = 100
HIDDEN = 64
NUM_HEADS = 4
DROPOUT = 0.1

PRIOR_SETS = {
    "None": [],
    "Zhou_Jintao": [30 * 12, 20 * 12, 7 * 12],
    "Kondratiev_Juglar": [25 * 12, 10 * 12, 4 * 12],
    "Kuznets": [20 * 12, 10 * 12, 5 * 12],
}


def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X), np.array(Y)


def train_pab(X_train, Y_train, horizon, periods_in, seed=0):
    """Train PAB with explicit period set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Z-score normalize
    x_mean, x_std = X_train.mean(), X_train.std() + 1e-8
    y_mean, y_std = Y_train.mean(), Y_train.std() + 1e-8
    X_n = (X_train - x_mean) / x_std
    Y_n = (Y_train - y_mean) / y_std
    X = torch.tensor(X_n, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y_n, dtype=torch.float32).unsqueeze(-1)
    n_periods = max(len(periods_in), 2) if periods_in else 2
    model = PeriodAttentionTimesNetLite(
        seq_len=SEQ_LEN, horizon=horizon, n_channels=1,
        top_k=n_periods, hidden=HIDDEN, num_heads=NUM_HEADS, dropout=DROPOUT
    )
    if periods_in and len(periods_in) > 0:
        rounded = [max(int(round(p)), 4) for p in periods_in]
        model.block.init_periods(rounded)
    else:
        # Use AMPD to discover
        amp = AMPD(top_k=n_periods, max_period=360, min_period=4)
        periods = amp.fit_discover(X_n[:, 0])
        model.block.init_periods([max(int(round(p)), 4) for p in periods])
    # Move all submodules to device
    device = DEVICE
    model = model.to(device)
    model.block.blocks = model.block.blocks.to(device)
    if model.block.proj is not None:
        model.block.proj = model.block.proj.to(device)
    model.norm_in = model.norm_in.to(device)
    model.norm_out = model.norm_out.to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.HuberLoss(delta=1.0)
    n = X.shape[0]
    bs = min(32, n)
    X = X.to(device)
    Y = Y.to(device)
    for epoch in range(EPOCHS):
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
    model._x_mean = x_mean
    model._x_std = x_std
    model._y_mean = y_mean
    model._y_std = y_std
    return model


def predict_pab(model, X_test):
    model.eval()
    X_n = (X_test - model._x_mean) / model._x_std
    with torch.no_grad():
        X = torch.tensor(X_n, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_n = model(X).cpu().numpy().squeeze(-1)
    return y_n * model._y_std + model._y_mean


def evaluate(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 70)
    print(" Experiment 09: Real-data PAB ablation (4 priors)")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values
    n = len(series)
    n_train = int(n * (1 - TEST_FRAC))
    print(f"[data] n={n}, n_train={n_train}")

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
        for prior_name, prior_periods in PRIOR_SETS.items():
            t0 = time.time()
            preds = []
            for seed in SEEDS:
                model = train_pab(X_train_p, Y_train_p, h, prior_periods, seed=seed)
                preds.append(predict_pab(model, X_test_p))
            yp = np.mean(preds, axis=0)
            mae, rmse = evaluate(Y_test_p, yp)
            t1 = time.time()
            h_results[prior_name] = {
                "periods_months": [float(p) for p in (prior_periods or [])],
                "mae": mae,
                "rmse": rmse,
                "train_time_s": t1 - t0,
            }
            print(f"  {prior_name:<22}: MAE={mae:.5f} RMSE={rmse:.5f}  ({t1-t0:.1f}s)")

        results[f"h{h}"] = h_results

    # Summary
    print("\n" + "=" * 70)
    print(" PAB Ablation: MAE on S&P 500 log-returns (4 priors x 4 horizons)")
    print("=" * 70)
    header = f"{'Horizon':<10} {'None':>10} {'Zhou':>10} {'Kondr-Jugl':>12} {'Kuznets':>10}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        r = results[f"h{h}"]
        row = f"  h={h:<6} "
        for k in ["None", "Zhou_Jintao", "Kondratiev_Juglar", "Kuznets"]:
            row += f" {r[k]['mae']:>10.5f}"
        print(row)
    print()
    # Verdict: do priors differ?
    print("Verdict: do priors now produce different MAE?")
    for h in HORIZONS:
        maes = [results[f"h{h}"][k]["mae"] for k in ["None", "Zhou_Jintao", "Kondratiev_Juglar", "Kuznets"]]
        spread = max(maes) - min(maes)
        rel_spread = spread / np.mean(maes) * 100
        verdict = "DIFFERENT" if rel_spread > 1.0 else "STILL IDENTICAL"
        print(f"  h={h}: spread={spread:.5f}  rel_spread={rel_spread:.2f}%  {verdict}")

    out = RESULTS_DIR / "09_pab_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
