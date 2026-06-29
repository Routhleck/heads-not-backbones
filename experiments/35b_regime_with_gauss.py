"""
Experiment 35b: Regime stress test extension — add TimesNet_gauss.

Trains TimesNet with the single-Gaussian density head (jointly trained with
Gaussian NLL, exp 34's head) on the same 5 anchored walk-forward folds as
exp 32. Computes per-window CRPS, then aggregates by regime.

Output: results/35b_regime_with_gauss.json — extends exp 32's regime stress
data with the TimesNet_gauss column.

Tasks: 1 backbone × 1 head (gauss) × 5 folds × 4 horizons × 3 seeds = 60.
Wall clock on RTX 4060: ~1-2 min.
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Resolve ROOT relative to this file (cross-platform)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ampd import AMPD

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
HIDDEN = 64
N_SAMPLES_CRPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Same regime definitions as exp 32
REGIMES = [
    ("1929-09-30", "1932-06-30", "pre_wwii_great_depression"),
    ("1973-01-31", "1974-12-31", "high_vol_1970s"),
    ("2000-03-31", "2002-10-31", "dotcom_bust"),
    ("2008-01-31", "2009-06-30", "2008_gfc"),
    ("2010-07-31", "2020-01-31", "secular_bull"),
    ("2020-02-29", "2020-04-30", "covid_crash"),
]


# ============= Backbone (TimesNet) + Gaussian head =============
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


class GaussianHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(hidden, horizon * 2)

    def forward(self, h):
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, 2)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    log_p = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    return -log_p.mean()


def sample_gaussian(mu, sigma, n_samples, generator):
    eps = torch.randn((n_samples,) + mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps


# ============= Data =============
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
        folds.append({
            "fold": fold_idx,
            "X_train": X_all[:train_end_pair],
            "Y_train": Y_all[:train_end_pair],
            "X_test": X_all[train_end_pair:test_end_pair],
            "Y_test": Y_all[train_end_pair:test_end_pair],
        })
        fold_idx += 1
        train_end_series += step
    return folds


def crps(samples, Y):
    half = N_SAMPLES_CRPS // 2
    if Y.ndim == 1:
        Y = Y[None, :]
    term1 = np.mean(np.abs(samples - Y[None, :]), axis=0)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    return (term1 - 0.5 * term2).flatten()


def train_and_get_per_window_crps(args):
    horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = TimesNetBackbone(SEQ_LEN).to(DEVICE)
    head = GaussianHead(HIDDEN, horizon).to(DEVICE)

    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        backbone.train()
        head.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            h = backbone(X[idx])
            mu, sigma = head(h)
            loss = gaussian_nll(Y[idx], mu, sigma)
            loss.backward()
            opt.step()
        sched.step()

    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        mu, sigma = head(h_test)
        g = torch.Generator(device=DEVICE)
        g.manual_seed(seed)
        samples = sample_gaussian(mu, sigma, N_SAMPLES_CRPS, g).squeeze(-1).cpu().numpy()

    return crps(samples, Y_test).astype(np.float32)


def main():
    print("=" * 72)
    print(" Experiment 35b: Regime stress test — add TimesNet_gauss")
    print("=" * 72)

    series_pd = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)["SP500_Return"].dropna()
    series_arr = series_pd.values.astype(np.float32)
    dates = series_pd.index
    n = len(series_arr)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    print(f"[data] {len(series_pd)} obs, {dates[0].date()} to {dates[-1].date()}")

    per_regime_results = {}

    for horizon in HORIZONS:
        print(f"\n=== Horizon h={horizon} ===")
        folds = build_folds(series_arr, horizon)
        tasks = []
        for seed in SEEDS:
            for fi, f in enumerate(folds):
                tasks.append((horizon, fi, seed, f["X_train"], f["Y_train"],
                              f["X_test"], f["Y_test"]))

        t0 = time.time()
        results_gauss = Parallel(n_jobs=1, verbose=0)(
            delayed(train_and_get_per_window_crps)(t) for t in tasks
        )
        elapsed = time.time() - t0
        print(f"  trained {len(tasks)} cells in {elapsed:.1f}s")

        crps_gauss_cells = [np.asarray(r).flatten() for r in results_gauss]
        crps_gauss_all = np.concatenate(crps_gauss_cells)
        n_test_per_fold = len(crps_gauss_cells[0]) if crps_gauss_cells else 0

        # Build window dates
        window_dates = []
        for fi in range(len(folds)):
            train_end_series = init_train + fi * step
            test_start_idx = train_end_series + 1
            for i in range(n_test_per_fold):
                if test_start_idx + i < len(dates):
                    window_dates.append(dates[test_start_idx + i])
                else:
                    window_dates.append(None)
        window_dates = np.array(window_dates)
        mask_dates = np.array([d is not None for d in window_dates])
        mask_full = np.tile(mask_dates, len(SEEDS))

        for start_date, end_date, label in REGIMES:
            sd = pd.Timestamp(start_date)
            ed = pd.Timestamp(end_date)
            regime_mask_dates = np.array(
                [d is not None and sd <= d <= ed for d in window_dates])
            regime_mask_full = np.tile(regime_mask_dates, len(SEEDS)) & mask_full
            crps_g_reg = crps_gauss_all[regime_mask_full]
            if len(crps_g_reg) == 0:
                continue
            per_regime_results.setdefault(label, {"horizons": {}})
            per_regime_results[label]["horizons"][f"h{horizon}"] = {
                "n_windows": int(len(crps_g_reg)),
                "crps_gauss_mean": float(crps_g_reg.mean()),
                "date_range": [str(sd.date()), str(ed.date())],
            }
            print(f"  [{label:<22}] n={len(crps_g_reg):3d}  gauss CRPS = {crps_g_reg.mean():.5f}")

    # Merge with exp 32's existing data (point + gmm)
    exp32_path = RESULTS_DIR / "32_regime_stress.json"
    if exp32_path.exists():
        with open(exp32_path) as f:
            exp32 = json.load(f)
        for label, hdata in exp32["per_regime_per_horizon"].items():
            if label in per_regime_results:
                for h, hvals in hdata["horizons"].items():
                    if h in per_regime_results[label]["horizons"]:
                        # Add point and gmm from exp 32
                        per_regime_results[label]["horizons"][h].update({
                            "crps_point_mean": hvals.get("crps_point_mean"),
                            "crps_gmm_mean": hvals.get("crps_gmm_mean"),
                            "crps_ss_pct_gmm_vs_point": hvals.get("crps_ss_pct"),
                        })
                        # Compute GMM vs Gauss incremental
                        crps_g = per_regime_results[label]["horizons"][h]["crps_gauss_mean"]
                        crps_gmm = hvals.get("crps_gmm_mean")
                        if crps_gmm is not None and crps_g is not None and crps_g > 0:
                            per_regime_results[label]["horizons"][h]["crps_ss_pct_gmm_vs_gauss"] = (
                                (crps_g - crps_gmm) / crps_g * 100
                            )

    out = {
        "config": {
            "EPOCHS": EPOCHS, "HORIZONS": HORIZONS, "SEEDS": SEEDS,
            "N_SAMPLES_CRPS": N_SAMPLES_CRPS, "device": str(DEVICE),
        },
        "regimes": [{"label": lbl, "start": sd, "end": ed} for sd, ed, lbl in REGIMES],
        "per_regime_per_horizon": per_regime_results,
    }
    out_path = RESULTS_DIR / "35b_regime_with_gauss.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
