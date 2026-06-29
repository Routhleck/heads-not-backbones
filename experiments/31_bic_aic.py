"""
Experiment 31: BIC/AIC for GMM component selection
=====================================================

Fits GaussianMixture with K = 1..8 components on the standardized
forecast residuals from TimesNet_gmm across folds. Computes BIC and AIC
to determine whether K=4 (the paper's default) is supported by
information criteria.

Protocol:
  - Reuse exp 22's TIMESNET_GMM training (one fold, seed 0) to obtain
    per-test-window residuals for each horizon.
  - Standardize residuals (subtract mean, divide by std).
  - Fit GaussianMixture(K) for K = 1..8.
  - Record BIC, AIC, log-likelihood, convergence.

Output:
  results/31_bic_aic.json  -- per-K BIC/AIC table; BIC-selected K

Usage:
    python experiments/31_bic_aic.py   (Windows GPU box)
"""

import sys, os, json, time, warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
EPOCHS = 80
LR = 1e-3
BS = 128
N_MIXTURES = 4
HIDDEN = 64
SEEDS = [0]
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

K_RANGE = list(range(1, 9))


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
        from src.models.ampd import AMPD
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


def build_folds(series, horizon):
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
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
# Train TimesNet_gmm, return per-test-window residuals
# ============================================================================
def train_and_get_residuals(args):
    horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    from src.models.wpn import gmm_nll, gmm_point_predict
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = TimesNetBackbone(SEQ_LEN).to(DEVICE)
    head = GMMHead(HIDDEN, horizon).to(DEVICE)

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
            mu, sigma, pi = head(h)
            loss = gmm_nll(Y[idx], mu, sigma, pi)
            loss.backward()
            opt.step()
        sched.step()

    # Predict
    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        mu, sigma, pi = head(h_test)
        # Point forecast = weighted mean across components (matches exp 22)
        y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, H)

    # Per-window residuals
    resid = (Y_test - y_pred).astype(np.float32)  # (n_test, H)
    return resid


# ============================================================================
# Fit GMM(K) on standardized residuals
# ============================================================================
def fit_gmm_per_k(resid_flat: np.ndarray, k: int, n_init: int = 5, seed: int = 0):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=n_init, random_state=seed,
                          max_iter=200, reg_covar=1e-4)
    gmm.fit(resid_flat)
    bic = gmm.bic(resid_flat)
    aic = gmm.aic(resid_flat)
    ll = gmm.score(resid_flat) * resid_flat.shape[0]   # total log-likelihood
    return {"k": k, "bic": float(bic), "aic": float(aic),
            "log_likelihood": float(ll), "converged": bool(gmm.converged_)}


def main():
    print("=" * 72)
    print(" Experiment 31: BIC/AIC for GMM component selection")
    print("=" * 72)
    print(f"[setup] TimesNet_gmm, SEEDS={SEEDS}, all folds, horizons={HORIZONS}")
    print(f"[setup] K range: {K_RANGE}")

    series = load_series()
    print(f"[data] {len(series)} monthly log-returns")

    # Per-horizon: train TimesNet_gmm on each fold, collect residuals,
    # then fit GMM(K) for K=1..8 on pooled standardized residuals.
    results_per_horizon = {}

    for horizon in HORIZONS:
        print(f"\n=== Horizon h={horizon} ===")
        folds = build_folds(series, horizon)
        tasks = [(horizon, fi, s, f["X_train"], f["Y_train"], f["X_test"], f["Y_test"])
                 for s in SEEDS for fi, f in enumerate(folds)]
        print(f"[tasks] {len(tasks)} (seed x fold) cells")

        t0 = time.time()
        all_resid = Parallel(n_jobs=1, verbose=0)(
            delayed(train_and_get_residuals)(t) for t in tasks
        )
        print(f"[trained] {len(all_resid)} cells in {time.time() - t0:.1f}s")

        # Concatenate residuals
        resid = np.concatenate(all_resid, axis=0)              # (N, H)
        resid_flat = resid.flatten()                           # (N*H,)
        # Standardize
        mu_r, std_r = resid_flat.mean(), resid_flat.std()
        resid_std = (resid_flat - mu_r) / (std_r + 1e-12)
        print(f"[residuals] shape={resid.shape}, N*H={resid_flat.shape[0]}, "
              f"std={std_r:.5f}")

        # Fit GMM(K) for each K
        gmm_results = []
        for k in K_RANGE:
            r = fit_gmm_per_k(resid_std.reshape(-1, 1), k, n_init=5)
            print(f"  K={k}: BIC={r['bic']:.1f}  AIC={r['aic']:.1f}  "
                  f"LL={r['log_likelihood']:.1f}  conv={r['converged']}")
            gmm_results.append(r)

        # Pick K by BIC and AIC
        k_bic = min(gmm_results, key=lambda r: r["bic"])["k"]
        k_aic = min(gmm_results, key=lambda r: r["aic"])["k"]
        print(f"[selected] BIC K={k_bic}, AIC K={k_aic}")

        results_per_horizon[f"h{horizon}"] = {
            "n_residuals": int(resid_flat.shape[0]),
            "residual_std": float(std_r),
            "residual_mean": float(mu_r),
            "gmm_per_k": gmm_results,
            "selected_K_bic": int(k_bic),
            "selected_K_aic": int(k_aic),
        }

    out = {
        "config": {
            "EPOCHS": EPOCHS,
            "HORIZONS": HORIZONS,
            "SEEDS": SEEDS,
            "K_RANGE": K_RANGE,
            "n_init": 5,
            "covariance_type": "full",
        },
        "per_horizon": results_per_horizon,
    }
    out_path = RESULTS_DIR / "31_bic_aic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] saved {out_path}")


if __name__ == "__main__":
    main()