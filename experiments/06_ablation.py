"""
Experiment 06: Ablation study on theory prior sets.

For each prior set (None / Zhou Jintao / Kondratiev-Juglar / Kuznets):
  - Train TimesBlock with the prior set periods
  - Compare against pure AMPD-discovered periods (None baseline)
  - Report MAE/RMSE on S&P 500 log-returns, horizons 1/3/6/12

Goal: determine whether adding theory priors helps forecasting.
If priors ≈ None, then the data speaks for itself; if priors ≪ None, priors help.
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

from src.models.times_block import TimesNetLite
from src.models.ampd import AMPD

warnings.filterwarnings("ignore")

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
TEST_FRAC = 0.30
HORIZONS = [1, 3, 6, 12]
SEEDS = [0]
EPOCHS = 40
HIDDEN = 16

PRIOR_SETS = {
    "None": [],  # AMPD discovers from data
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


def train_timesnet(X_train, Y_train, horizon, periods_in, seed=0):
    """Train TimesBlock with explicit period set (could be AMPD-discovered or prior)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
    model = TimesNetLite(seq_len=SEQ_LEN, horizon=horizon, n_channels=1, top_k=len(periods_in) or 2, hidden=HIDDEN)
    # Force init_periods with given periods (skip AMPD discovery)
    rounded = [max(int(round(p)), 6) for p in periods_in] if periods_in else None
    if rounded is None or len(rounded) == 0:
        # Fall back to AMPD discovery
        amp = AMPD(top_k=2, max_period=360, min_period=6)
        rounded = [max(int(round(p)), 6) for p in amp.fit_discover(X_train[:, 0])]
    model.block.init_periods(rounded)
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
        X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        return model(X).cpu().numpy().squeeze(-1)


def evaluate(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 70)
    print(" Experiment 06: Theory-prior ablation on S&P 500 log-returns")
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
        # Discover AMPD periods once per horizon (None baseline)
        amp = AMPD(top_k=2, max_period=360, min_period=6)
        ampd_periods = list(amp.fit_discover(X_train_p[:, 0]))
        print(f"  AMPD-discovered (None): {[round(p, 1) for p in ampd_periods]} months")

        for prior_name, prior_periods in PRIOR_SETS.items():
            # For "None", use AMPD-discovered. For priors, use the given set.
            periods_used = ampd_periods if prior_name == "None" else list(prior_periods)
            # The top_k is min(2, len(periods_used)) - so use up to 2 periods
            periods_for_block = periods_used[:2]
            t0 = time.time()
            preds = []
            for seed in SEEDS:
                model = train_timesnet(X_train_p, Y_train_p, h, periods_for_block, seed=seed)
                preds.append(predict(model, X_test_p))
            yp = np.mean(preds, axis=0)
            mae, rmse = evaluate(Y_test_p, yp)
            t1 = time.time()
            h_results[prior_name] = {
                "periods_months": [float(p) for p in periods_used],
                "periods_used_months": [float(p) for p in periods_for_block],
                "mae": mae,
                "rmse": rmse,
                "train_time_s": t1 - t0,
            }
            print(f"  {prior_name:<22}: MAE={mae:.5f} RMSE={rmse:.5f}  periods={[round(p,1) for p in periods_for_block]} ({t1-t0:.1f}s)")

        results[f"h{h}"] = h_results

    # Summary table
    print("\n" + "=" * 70)
    print(" Ablation summary: MAE on S&P 500 log-returns")
    print("=" * 70)
    header = f"{'Horizon':<10} {'None (AMPD)':>14} {'Zhou':>10} {'Kondr-Jugl':>12} {'Kuznets':>10}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        r = results[f"h{h}"]
        row = f"  h={h:<6} "
        for k in ["None", "Zhou_Jintao", "Kondratiev_Juglar", "Kuznets"]:
            row += f" {r[k]['mae']:>10.5f} "
        print(row)
    print()
    print("If 'None' is best: data-driven discovery wins (current result expected)")
    print("If a prior set is best: theory priors add value beyond data")
    print("If all within ~0.0005: no significant difference (conclusion: priors do not help here)")

    out = RESULTS_DIR / "06_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
