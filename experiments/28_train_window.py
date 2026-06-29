"""
Experiment 28: Training window sensitivity
==============================================

Tests whether the TimesNet + GMM density head advantage is robust to
the initial training fraction.

Protocol:
  - INITIAL_TRAIN_FRAC ∈ {0.40, 0.50, 0.60, 0.70}
  - For each frac, run exp 22's protocol: 4 horizons × 3 seeds × 5 folds ×
    2 variants (TimesNet_point, TimesNet_gmm) = 120 tasks per frac.
  - Compute CRPS-Skill-Score vs TimesNet_point baseline (per frac).

Total: 4 fracs × 120 = 480 tasks. ~12 min on RTX 4060.

Usage:
    python experiments/28_train_window.py
"""

import sys, os, json, time, warnings
from pathlib import Path

import numpy as np
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
TEST_FRAC = 0.07
STEP_FRAC = 0.10
EPOCHS = 80
LR = 1e-3
BS = 128
N_MIXTURES = 4
HIDDEN = 64
N_SAMPLES_CRPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

INITIAL_TRAIN_FRACS = [0.40, 0.50, 0.60, 0.70]
VARIANTS = ["TimesNet_point", "TimesNet_gmm"]


# ============================================================================
# Backbones + heads (verbatim from exp 22)
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


def build_model(variant, horizon):
    backbone = TimesNetBackbone(SEQ_LEN).to(DEVICE)
    if variant.endswith("_point"):
        head = PointHead(HIDDEN, horizon).to(DEVICE)
    else:
        head = GMMHead(HIDDEN, horizon).to(DEVICE)
    return backbone, head


# ============================================================================
# Data
# ============================================================================
def load_series():
    import pandas as pd
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return panel["SP500_Return"].dropna().values.astype(np.float32)


def make_supervised(series, seq_len=SEQ_LEN, horizon=1):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_folds(series, horizon, init_train_frac):
    n = len(series)
    init_train = int(n * init_train_frac)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    X_all, Y_all = make_supervised(series, SEQ_LEN, horizon)
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
        })
        fold_idx += 1
        train_end_series += step
    return folds


# ============================================================================
# Worker — train + CRPS-Skill-Score
# ============================================================================
def crps_skill_score(samples, Y_test):
    """CRPS-SS vs deterministic baseline (mean prediction)."""
    half = N_SAMPLES_CRPS // 2
    # samples: (N, n_test, H); Y_test: (n_test, H). Need (1, n_test, H)
    if Y_test.ndim == 2:
        Y_test_b = Y_test[None, :, :]                  # (1, n_test, H)
    else:
        Y_test_b = Y_test[None, :, :, :] if Y_test.ndim == 3 else Y_test[None]
    term1 = np.mean(np.abs(samples - Y_test_b), axis=0)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    crps_gmm = float((term1 - 0.5 * term2).mean())
    return crps_gmm


def train_and_score(args):
    variant, horizon, fold_idx, seed, init_train_frac, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone, head = build_model(variant, horizon)

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
                mu, sigma, pi = out
                loss = crit((mu, sigma, pi), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    # Predict + compute CRPS
    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if variant.endswith("_point"):
            y_pred = out.cpu().numpy()  # (n_test, H)
            # Gaussian samples for CRPS
            h_train = backbone(X)
            train_pred = head(h_train).detach().cpu().numpy()
            train_resid = Y_train - train_pred
            sigma_per_step = np.std(train_resid, axis=0)  # (H,)
            np.random.seed(seed)
            samples = np.random.normal(
                y_pred[None, :, :], sigma_per_step[None, None, :],
                size=(N_SAMPLES_CRPS, y_pred.shape[0], y_pred.shape[1]),
            )
        else:
            mu, sigma, pi = out
            samples = sample_gmm(mu, sigma, pi, n_samples=N_SAMPLES_CRPS)
            samples = samples.squeeze(-1).cpu().numpy()  # (N, n_test, H)
    crps = crps_skill_score(samples, Y_test)
    return crps


def main():
    print("=" * 72)
    print(" Experiment 28: Training window sensitivity")
    print("=" * 72)
    print(f"[setup] INITIAL_TRAIN_FRAC in {INITIAL_TRAIN_FRACS}")
    print(f"[setup] per frac: {len(HORIZONS)} horizons x {len(SEEDS)} seeds x 5 folds x 2 variants")

    series = load_series()
    print(f"[data] {len(series)} monthly log-returns")

    out_per_frac = {}

    for init_train_frac in INITIAL_TRAIN_FRACS:
        print(f"\n=== init_train_frac = {init_train_frac} ===")
        # Compute CRPS-SS for both variants per horizon
        per_horizon = {h: {"TimesNet_point": [], "TimesNet_gmm": []} for h in HORIZONS}

        for horizon in HORIZONS:
            folds = build_folds(series, horizon, init_train_frac)
            print(f"  h={horizon}: {len(folds)} folds")

            tasks = []
            for seed in SEEDS:
                for fi, f in enumerate(folds):
                    for v in VARIANTS:
                        tasks.append((v, horizon, fi, seed, init_train_frac,
                                      f["X_train"], f["Y_train"], f["X_test"], f["Y_test"]))
            t0 = time.time()
            results = Parallel(n_jobs=1, verbose=0)(
                delayed(train_and_score)(t) for t in tasks
            )
            print(f"    trained {len(tasks)} cells in {time.time() - t0:.1f}s")

            # Bucket per variant
            for v in VARIANTS:
                v_idx = [i for i, t in enumerate(tasks) if t[0] == v]
                per_horizon[horizon][v] = [float(results[i]) for i in v_idx]

        # CRPS-SS = (CRPS_point - CRPS_gmm) / CRPS_point
        crps_ss = {}
        for h in HORIZONS:
            crps_p = np.mean(per_horizon[h]["TimesNet_point"])
            crps_g = np.mean(per_horizon[h]["TimesNet_gmm"])
            ss = (crps_p - crps_g) / crps_p if crps_p > 0 else 0.0
            crps_ss[h] = {
                "crps_point_mean": float(crps_p),
                "crps_gmm_mean": float(crps_g),
                "crps_ss": float(ss),
                "crps_ss_pct": float(ss * 100),
                "n_cells": len(per_horizon[h]["TimesNet_point"]),
            }
            print(f"  h={h}: CRPS-SS = {ss * 100:+.2f}% "
                  f"(point={crps_p:.6f}, gmm={crps_g:.6f})")

        out_per_frac[f"{init_train_frac:.2f}"] = {
            "init_train_frac": init_train_frac,
            "n_folds": len(build_folds(series, 1, init_train_frac)),
            "crps_ss_per_horizon": crps_ss,
        }

    # Summary table
    print("\n" + "=" * 72)
    print(" Summary: CRPS-Skill-Score (%) of TimesNet_gmm vs TimesNet_point")
    print("=" * 72)
    print(f"{'init_train_frac':<18} {'h=1':>8} {'h=3':>8} {'h=6':>8} {'h=12':>8}")
    for frac, data in out_per_frac.items():
        row = [f"{data['crps_ss_per_horizon'][h]['crps_ss_pct']:+.2f}%" for h in HORIZONS]
        print(f"{frac:<18} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    out = {
        "config": {
            "INITIAL_TRAIN_FRACS": INITIAL_TRAIN_FRACS,
            "HORIZONS": HORIZONS,
            "SEEDS": SEEDS,
            "EPOCHS": EPOCHS,
        },
        "per_init_train_frac": out_per_frac,
    }
    out_path = RESULTS_DIR / "28_train_window.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()