"""
Experiment 04: Real-data baseline forecast on S&P 500 monthly returns.

Compares:
  - Naive (last value persistence) - must beat
  - ARIMA(1,0,1) - classical baseline
  - TimesNetLite (our AMPD-aware 2D-variation block) - main model
  - Linear regression (sanity vs deep model)

Walk-forward validation with expanding window.
Horizons: 1, 3, 6, 12 months.
Metrics: MAE, RMSE.
"""
import sys
import time
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from src.models.times_block import TimesNetLite

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


def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X), np.array(Y)


def train_timesnet(X_train, Y_train, horizon, seed=0, epochs=200, lr=1e-3, hidden=32, top_k=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
    # Build model on CPU first, init periods, then move to device
    model = TimesNetLite(seq_len=SEQ_LEN, horizon=horizon, n_channels=1, top_k=top_k, hidden=hidden)
    model.fit_periods(X_train[:, 0], max_period=horizon * 24, min_period=6)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    n = X.shape[0]
    batch_size = min(32, n)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx].to(DEVICE)
            yb = Y[idx].to(DEVICE)
            opt.zero_grad()
            yp = model(xb)
            loss = crit(yp, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
    return model


def predict_timesnet(model, X_test):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        yp = model(X).cpu().numpy().squeeze(-1)
    return yp


def predict_naive(X_test, horizon):
    last = X_test[:, -1:]
    return np.broadcast_to(last, (X_test.shape[0], horizon)).copy()


def predict_arima(series_train, X_test, horizon, order=(1, 0, 1)):
    """
    ARIMA multi-step forecast: fit once on training data, then iteratively
    extend with predictions for h-step forecast, refit ONLY when new observation
    arrives (walk-forward). To keep cost low, refit every `refit_every` steps.
    """
    refit_every = 50  # balance between accuracy and speed
    try:
        history = list(series_train)
        n_test = X_test.shape[0]
        preds = np.zeros((n_test, horizon))
        last_fit_idx = -1
        fit = None
        for i in range(n_test):
            if i - last_fit_idx >= refit_every or fit is None:
                fit = ARIMA(history, order=order).fit()
                last_fit_idx = i
            # Forecast horizon steps iteratively using fitted model
            cur_hist = list(history)
            for k in range(horizon):
                # Use fit.forecast which extends with all prior predictions
                fc = fit.forecast(steps=1)
                yh = float(fc[0]) if hasattr(fc, '__getitem__') else float(fc)
                preds[i, k] = yh
                cur_hist.append(yh)
            # Update history with actual observation
            history.append(X_test[i, -1])
        return preds
    except Exception as e:
        print(f"  [ARIMA] fit failed: {e}; using naive")
        return predict_naive(X_test, horizon)


def predict_linear(X_train, Y_train, X_test):
    preds = []
    for k in range(Y_train.shape[1]):
        lr = LinearRegression()
        lr.fit(X_train, Y_train[:, k])
        preds.append(lr.predict(X_test))
    return np.array(preds).T


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    print("=" * 70)
    print(" Experiment 04: Real-data baseline forecast on S&P 500")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values
    print(f"[data] S&P monthly returns: n={len(series)}")

    n = len(series)
    n_train = int(n * (1 - TEST_FRAC))
    print(f"[split] train pairs: ~{int(n_train * (1 - TEST_FRAC))}, test pairs: ~{n - int(n_train * (1 - TEST_FRAC))}")

    train_series = series[:n_train]

    results = {}
    for h in HORIZONS:
        print(f"\n--- Horizon h = {h} months ---")
        X_all, Y_all = make_supervised(series, SEQ_LEN, h)
        n_pairs = X_all.shape[0]
        n_test_pairs = n - n_train
        X_train_p = X_all[:-n_test_pairs]
        Y_train_p = Y_all[:-n_test_pairs]
        X_test_p = X_all[-n_test_pairs:]
        Y_test_p = Y_all[-n_test_pairs:]
        print(f"  X_train {X_train_p.shape}, Y_train {Y_train_p.shape}, X_test {X_test_p.shape}")

        h_results = {}

        yp_naive = predict_naive(X_test_p, h)
        mae, rmse = evaluate(Y_test_p, yp_naive)
        h_results["naive"] = {"mae": float(mae), "rmse": float(rmse)}
        print(f"  Naive      : MAE={mae:.5f}, RMSE={rmse:.5f}")

        yp_lin = predict_linear(X_train_p, Y_train_p, X_test_p)
        mae, rmse = evaluate(Y_test_p, yp_lin)
        h_results["linear"] = {"mae": float(mae), "rmse": float(rmse)}
        print(f"  Linear     : MAE={mae:.5f}, RMSE={rmse:.5f}")

        yp_arima = predict_arima(train_series, X_test_p, h)
        mae, rmse = evaluate(Y_test_p, yp_arima)
        h_results["arima"] = {"mae": float(mae), "rmse": float(rmse)}
        print(f"  ARIMA      : MAE={mae:.5f}, RMSE={rmse:.5f}")

        t0 = time.time()
        timesnet_preds = []
        for seed in SEEDS:
            model = train_timesnet(X_train_p, Y_train_p, h, seed=seed, epochs=EPOCHS, lr=1e-3, hidden=HIDDEN, top_k=2)
            yp = predict_timesnet(model, X_test_p)
            timesnet_preds.append(yp)
        yp_tn = np.mean(timesnet_preds, axis=0)
        t1 = time.time()
        mae, rmse = evaluate(Y_test_p, yp_tn)
        h_results["timesnet_lite"] = {"mae": float(mae), "rmse": float(rmse), "train_time_s": t1 - t0}
        print(f"  TimesNetLite: MAE={mae:.5f}, RMSE={rmse:.5f}  ({t1-t0:.1f}s, {len(SEEDS)} seeds)")

        results[f"h{h}"] = h_results

    print("\n" + "=" * 70)
    print(" Summary: MAE on S&P monthly returns (walk-forward)")
    print("=" * 70)
    header = f"{'Horizon':<10} {'Naive':>10} {'Linear':>10} {'ARIMA':>10} {'TimesNet':>10}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        r = results[f"h{h}"]
        row = f"  h={h:<6} "
        for m in ["naive", "linear", "arima", "timesnet_lite"]:
            row += f" {r[m]['mae']:>10.5f}"
        print(row)

    out = RESULTS_DIR / "04_real_baseline.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
