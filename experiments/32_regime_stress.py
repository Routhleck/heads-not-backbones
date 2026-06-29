"""
Experiment 32: Per-regime stress test
=======================================

Tests whether the TimesNet_gmm advantage holds during specific market
regimes — particularly crisis periods where forecasting is hardest.

Regimes (using S&P 500 monthly data 1871-2023):
  - 2008 GFC:        2008-01 to 2009-06  (18 months)
  - COVID crash:     2020-02 to 2020-04  (3 months)
  - Dot-com bust:    2000-03 to 2002-10  (32 months)
  - Secular bull:    2010-07 to 2020-01  (115 months)
  - High-vol 1970s:  1973-01 to 1974-12  (24 months)
  - Pre-WWII:        1929-09 to 1932-06  (34 months)

For each regime, compute TimesNet_gmm vs TimesNet_point CRPS on
test windows that fall within that regime's date range.

Usage:
    python experiments/32_regime_stress.py   (Windows GPU box)
"""

import sys, os, json, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ampd import AMPD
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"

SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
EPOCHS = 80
LR = 1e-3
BS = 128
N_MIXTURES = 4
HIDDEN = 64
N_SAMPLES_CRPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Regime definitions (start_date, end_date, label)
REGIMES = [
    ("1929-09-30", "1932-06-30", "pre_wwii_great_depression"),
    ("1973-01-31", "1974-12-31", "high_vol_1970s"),
    ("2000-03-31", "2002-10-31", "dotcom_bust"),
    ("2008-01-31", "2009-06-30", "2008_gfc"),
    ("2010-07-31", "2020-01-31", "secular_bull"),
    ("2020-02-29", "2020-04-30", "covid_crash"),
]


# ============================================================================
# Backbones + heads (TimesNet only for this experiment)
# ============================================================================
class TimesNetBackbone(nn.Module):
    def __init__(self, seq_len, hidden=HIDDEN, top_k=2):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.hidden = hidden
        self.periods = []
        self.conv2d = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()

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
        B, T, C = x.shape
        if not self.periods:
            x_np = x.detach().squeeze(-1).cpu().numpy()
            self.fit_periods(x_np[0])
        agg = None
        for p in self.periods:
            x2d = self._reshape_2d(x, p)
            h2d = self.act(self.conv2d(x2d))
            h_pool = h2d.mean(dim=(2, 3))
            agg = h_pool if agg is None else agg + h_pool
        if agg is None:
            agg = x.mean(dim=1)
        return agg


class PointHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):
        return self.fc(x)


class GMMHead(nn.Module):
    def __init__(self, hidden, horizon, n_mixtures=N_MIXTURES):
        super().__init__()
        self.horizon = horizon
        self.n_mixtures = n_mixtures
        self.fc_mu = nn.Linear(hidden, horizon * n_mixtures)
        self.fc_sigma = nn.Linear(hidden, horizon * n_mixtures)
        self.fc_pi = nn.Linear(hidden, horizon * n_mixtures)
    def forward(self, x):
        B = x.shape[0]
        mu = self.fc_mu(x).view(B, self.horizon, 1, self.n_mixtures)
        sigma = F.softplus(self.fc_sigma(x)).view(B, self.horizon, 1, self.n_mixtures) + 1e-3
        pi = F.softmax(self.fc_pi(x), dim=-1).view(B, self.horizon, 1, self.n_mixtures)
        return mu, sigma, pi


# ============================================================================
# Data
# ============================================================================
def load_series_with_dates():
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna()
    return series  # pandas Series with DatetimeIndex


def make_supervised(series_arr, seq_len=SEQ_LEN, horizon=1):
    X, Y = [], []
    for t in range(len(series_arr) - seq_len - horizon + 1):
        X.append(series_arr[t:t + seq_len])
        Y.append(series_arr[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_folds(series_arr, horizon):
    n = len(series_arr)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    X_all, Y_all = make_supervised(series_arr, SEQ_LEN, horizon)
    folds = []
    fold_idx = 0
    train_end_series = init_train
    while train_end_series + test_window + horizon <= n:
        test_end_series = train_end_series + test_window
        train_end_pair = train_end_series - SEQ_LEN
        test_end_pair = test_end_series - SEQ_LEN
        X_train = X_all[:train_end_pair]
        Y_train = Y_all[:train_end_pair]
        X_test = X_all[train_end_pair:test_end_pair]
        Y_test = Y_all[train_end_pair:test_end_pair]
        if len(X_test) == 0:
            break
        folds.append({
            "fold": fold_idx,
            "X_train": X_train, "Y_train": Y_train,
            "X_test": X_test, "Y_test": Y_test,
            "test_end_idx": test_end_series,
        })
        fold_idx += 1
        train_end_series += step
    return folds


# ============================================================================
# Worker — train, get CRPS per test window with date
# ============================================================================
def crps(samples, Y):
    half = N_SAMPLES_CRPS // 2
    if Y.ndim == 1:
        Y = Y[None, :]
    if samples.ndim == 2:
        samples = samples[None, :, :] if Y.ndim == 1 else samples
    term1 = np.mean(np.abs(samples - Y[None, :]), axis=0)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    return (term1 - 0.5 * term2)  # (H,) or scalar


def train_and_get_per_window_crps(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = TimesNetBackbone(SEQ_LEN).to(DEVICE)
    if variant.endswith("_point"):
        head = PointHead(HIDDEN, horizon).to(DEVICE)
    else:
        head = GMMHead(HIDDEN, horizon).to(DEVICE)

    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    if variant.endswith("_point"):
        crit = nn.HuberLoss(delta=1.0)
    else:
        crit = lambda yp, yt: gmm_nll(yt, yp[0], yp[1], yp[2])

    for ep in range(EPOCHS):
        backbone.train()
        head.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            h = backbone(X[idx])
            out = head(h)
            if variant.endswith("_point"):
                yp = out.unsqueeze(-1)
                loss = crit(yp, Y[idx])
            else:
                mu_, sigma_, pi_ = out
                loss = crit((mu_, sigma_, pi_), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if variant.endswith("_point"):
            y_pred = out.cpu().numpy()
            h_train = backbone(X)
            train_pred = head(h_train).detach().cpu().numpy()
            train_resid = Y_train - train_pred
            sigma_per_step = np.std(train_resid, axis=0)
            np.random.seed(seed)
            samples = np.random.normal(
                y_pred[None, :, :], sigma_per_step[None, None, :],
                size=(N_SAMPLES_CRPS, y_pred.shape[0], y_pred.shape[1]),
            )
        else:
            mu_, sigma_, pi_ = out
            samples = sample_gmm(mu_, sigma_, pi_, n_samples=N_SAMPLES_CRPS).squeeze(-1).cpu().numpy()

    # Per-window CRPS (mean over horizons)
    crps_per_window = crps(samples, Y_test)  # (n_test,)
    crps_per_window = np.asarray(crps_per_window).flatten()  # ensure 1D
    if crps_per_window.ndim > 1:
        crps_per_window = crps_per_window.mean(axis=-1)
    return crps_per_window.astype(np.float32)


def main():
    print("=" * 72)
    print(" Experiment 32: Per-regime stress test")
    print("=" * 72)

    series_pd = load_series_with_dates()
    series_arr = series_pd.values.astype(np.float32)
    dates = series_pd.index
    print(f"[data] {len(series_pd)} monthly log-returns, "
          f"{dates[0].date()} to {dates[-1].date()}")

    # Compute test_end_idx for each fold so we can map to dates
    n = len(series_arr)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)

    # For each (variant, horizon, regime), gather CRPS across all test windows in regime
    per_regime_results = {}

    for horizon in HORIZONS:
        print(f"\n=== Horizon h={horizon} ===")
        folds = build_folds(series_arr, horizon)
        # Build tasks
        tasks_point = []
        tasks_gmm = []
        for seed in SEEDS:
            for fi, f in enumerate(folds):
                tasks_point.append(("TimesNet_point", horizon, fi, seed,
                                    f["X_train"], f["Y_train"], f["X_test"], f["Y_test"]))
                tasks_gmm.append(("TimesNet_gmm", horizon, fi, seed,
                                  f["X_train"], f["Y_train"], f["X_test"], f["Y_test"]))

        t0 = time.time()
        results_point = Parallel(n_jobs=1, verbose=0)(
            delayed(train_and_get_per_window_crps)(t) for t in tasks_point
        )
        results_gmm = Parallel(n_jobs=1, verbose=0)(
            delayed(train_and_get_per_window_crps)(t) for t in tasks_gmm
        )
        elapsed = time.time() - t0
        print(f"  trained {len(tasks_point) + len(tasks_gmm)} cells in {elapsed:.1f}s")

        # For each fold, attach date of test_end
        # The test_end corresponds to the last test window in the fold
        for start_date, end_date, label in REGIMES:
            sd = pd.Timestamp(start_date)
            ed = pd.Timestamp(end_date)
            # Compute per-regime stats — per-fold CRPS averaged then date-bin
            pass

        # Compute per-regime stats — simpler: per-fold CRPS averaged then date-bin
        # Flatten all per-window CRPS across seeds+folds
        # results_point[i] has shape (n_test,) — verify and reshape
        crps_point_cells = [np.asarray(r).flatten() for r in results_point]
        crps_gmm_cells = [np.asarray(r).flatten() for r in results_gmm]
        crps_point_all = np.concatenate(crps_point_cells)
        crps_gmm_all = np.concatenate(crps_gmm_cells)
        n_test_per_fold = len(crps_point_cells[0]) if crps_point_cells else 0

        # Build window dates: each fold's test window has the same date across seeds
        n_folds = len(folds)
        window_dates = []
        for fi in range(n_folds):
            train_end_series = init_train + fi * step
            test_start_idx = train_end_series + 1
            for i in range(n_test_per_fold):
                if test_start_idx + i < len(dates):
                    window_dates.append(dates[test_start_idx + i])
                else:
                    window_dates.append(None)
        window_dates = np.array(window_dates)
        # Broadcast: (n_folds * n_test,) tiled n_seeds times -> (n_seeds*n_folds*n_test,)
        if len(window_dates) > 0 and len(crps_point_all) > 0:
            n_expected = len(window_dates) * len(SEEDS)
            assert len(crps_point_all) == n_expected, (
                f"shape mismatch: crps={len(crps_point_all)}, expected={n_expected}")
            mask_dates = np.array([d is not None for d in window_dates])
            mask_full = np.tile(mask_dates, len(SEEDS))
        else:
            mask_full = np.zeros(len(crps_point_all), dtype=bool)

        for start_date, end_date, label in REGIMES:
            sd = pd.Timestamp(start_date)
            ed = pd.Timestamp(end_date)
            regime_mask_dates = np.array(
                [d is not None and sd <= d <= ed for d in window_dates])
            regime_mask_full = np.tile(regime_mask_dates, len(SEEDS))
            regime_mask_full = regime_mask_full & mask_full
            crps_p_reg = crps_point_all[regime_mask_full]
            crps_g_reg = crps_gmm_all[regime_mask_full]
            if len(crps_p_reg) == 0:
                continue
            crps_ss = (crps_p_reg.mean() - crps_g_reg.mean()) / crps_p_reg.mean() * 100
            per_regime_results.setdefault(label, {"horizons": {}})
            per_regime_results[label]["horizons"][f"h{horizon}"] = {
                "n_windows": int(len(crps_p_reg)),
                "crps_point_mean": float(crps_p_reg.mean()),
                "crps_gmm_mean": float(crps_g_reg.mean()),
                "crps_ss_pct": float(crps_ss),
                "date_range": [str(sd.date()), str(ed.date())],
            }
            print(f"  [{label:<22}] n={len(crps_p_reg):3d}  "
                  f"CRPS-SS = {crps_ss:+.2f}% "
                  f"(point={crps_p_reg.mean():.5f}, gmm={crps_g_reg.mean():.5f})")

    out = {
        "config": {
            "EPOCHS": EPOCHS,
            "HORIZONS": HORIZONS,
            "SEEDS": SEEDS,
            "N_SAMPLES_CRPS": N_SAMPLES_CRPS,
        },
        "regimes": [{"label": lbl, "start": sd, "end": ed} for sd, ed, lbl in REGIMES],
        "per_regime_per_horizon": per_regime_results,
    }
    out_path = RESULTS_DIR / "32_regime_stress.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()