"""
Experiment 34: Single-Gaussian density head × 4 backbones.

For each of the 4 backbones (TimesNet, DLinear, N-BEATS, iTransformer), train
a single-Gaussian density head (Linear -> (mu, sigma), trained with NLL) on
the exact same protocol as Experiment 22 (SOTA comparison).

iTransformer replaces PatchTST (see DECISION_PATCHTST_DROPPED.md).

This complements exp 22 (point head + GMM head) by adding the missing middle
case: a properly trained single-Gaussian density head. Together with exp 22,
this gives a clean 4 backbones x 3 heads = 12 variant comparison.

Critical protocol parity with exp 22:
  - Same walk-forward folds (5 anchored folds)
  - Same horizons {1, 3, 6, 12}
  - Same seeds {0, 1, 2}
  - Same backbone hyperparameters
  - Same Adam + cosine, lr=1e-3, weight_decay=1e-3, batch=128, 80 epochs
  - Same N=500 CRPS samples

Tasks: 4 backbones x 1 head x 5 folds x 4 horizons x 3 seeds = 240.

The single-Gaussian head outputs (mu, sigma) per (h, channel) and is trained
by the standard Gaussian NLL loss. It is NOT the empirical-Gaussian
approximation used in exp 22's "point head" CRPS evaluation — that was a
post-hoc construct from training residuals. Here sigma is learned jointly
with mu via gradient descent on NLL.
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
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from scipy import stats as sp_stats

from src.models.ampd import AMPD
from src.models.nbeats import NBEATSBackbone

warnings.filterwarnings("ignore")

# ---------------- Config (mirrors exp 22 exactly) ----------------
DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

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
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
WINKLER_LEVELS = [0.50, 0.80, 0.90, 0.95]


# ============================================================
# Backbones — identical to exp 22 (replicated to keep this script self-contained)
# ============================================================

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


class DLinearBackbone(nn.Module):
    def __init__(self, seq_len, hidden=HIDDEN, kernel=25):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        self.trend_feat = nn.Linear(seq_len, hidden)
        self.seasonal_feat = nn.Linear(seq_len, hidden)
        self.combine = nn.Linear(2 * hidden, hidden)

    def forward(self, x):
        x = x.squeeze(-1)
        trend = F.avg_pool1d(x.unsqueeze(1), self.kernel, stride=1,
                              padding=self.kernel // 2).squeeze(1)
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1]]
        seasonal = x - trend
        t = self.trend_feat(trend)
        s = self.seasonal_feat(seasonal)
        return self.combine(torch.cat([t, s], dim=-1))


class ITransformerBackboneAdapter(nn.Module):
    """Wrapper for src.models.itransformer.ITransformerBackbone
    to match the experiment script's (B, T, 1) -> (B, hidden) interface.
    """
    def __init__(self, seq_len, hidden=HIDDEN, n_heads=4, n_layers=3,
                 ff_dim=128, dropout=0.1):
        super().__init__()
        from src.models.itransformer import ITransformerBackbone
        self.impl = ITransformerBackbone(
            seq_len=seq_len, hidden=hidden, n_heads=n_heads,
            n_layers=n_layers, ff_dim=ff_dim, dropout=dropout,
        )

    def forward(self, x):
        return self.impl(x)


# ============================================================
# Single-Gaussian density head — joint NLL training
# ============================================================

class GaussianHead(nn.Module):
    """Linear projection: hidden -> (mu, sigma) per (h, channel).

    Trained with Gaussian NLL:
        L = -log N(y | mu, sigma^2)
    Both mu and sigma are learned jointly via gradient descent.
    """
    def __init__(self, hidden, horizon):
        super().__init__()
        self.horizon = horizon
        # 2 outputs per (h, channel): mu, log_sigma
        self.proj = nn.Linear(hidden, horizon * 2)

    def forward(self, h):
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, 2)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    """Gaussian negative log-likelihood (mean over batch).

    Args:
        y: (B, H, 1) target
        mu, sigma: (B, H, 1) predicted mean and std
    Returns:
        scalar NLL
    """
    # mu, sigma, y are all (B, H, 1). Compute log-density directly without unsqueezing.
    log_p = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    return -log_p.mean()


def sample_gaussian(mu, sigma, n_samples=500, generator=None):
    """Sample from a Gaussian predictive distribution.

    Args:
        mu, sigma: (B, H, 1)
        n_samples: int
    Returns:
        samples: (n_samples, B, H, 1)
    """
    eps = torch.randn((n_samples,) + mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps


# ============================================================
# Model registry
# ============================================================

def build_backbone(backbone_name, seq_len):
    if backbone_name == "TimesNet":
        return TimesNetBackbone(seq_len=seq_len, hidden=HIDDEN, top_k=2)
    if backbone_name == "DLinear":
        return DLinearBackbone(seq_len=seq_len, hidden=HIDDEN, kernel=25)
    if backbone_name == "NBEATS":
        return NBEATSBackbone(seq_len=seq_len, hidden=HIDDEN, n_blocks=2, theta_dim=8)
    if backbone_name == "iTransformer":
        return ITransformerBackboneAdapter(seq_len=seq_len, hidden=HIDDEN,
                                           n_heads=4, n_layers=3, ff_dim=128)
    raise ValueError(f"Unknown backbone: {backbone_name}")


# ============================================================
# Data
# ============================================================

def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_folds(series, seq_len, horizon):
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    X_all, Y_all = make_supervised(series, seq_len, horizon)
    folds = []
    fold_idx = 0
    train_end_series = init_train
    while train_end_series + test_window + horizon <= n:
        test_end_series = train_end_series + test_window
        train_end_pair = train_end_series - seq_len
        test_end_pair = test_end_series - seq_len
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


# ============================================================
# Worker
# ============================================================

def train_and_save_predictions(args):
    backbone_name, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    backbone = build_backbone(backbone_name, SEQ_LEN)
    head = GaussianHead(hidden=HIDDEN, horizon=horizon)
    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)

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

    # Predict
    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        mu, sigma = head(h_test)
        mu_np = mu.cpu().numpy().squeeze(-1)   # (n_test, H)
        sigma_np = sigma.cpu().numpy().squeeze(-1)  # (n_test, H)
        # Point prediction = mu (Bayes-optimal under MAE for symmetric Gaussian)
        y_pred = mu_np
        # Draw N=500 samples per test point
        N_SAMPLES = 500
        g = torch.Generator(device=DEVICE)
        g.manual_seed(seed)
        samples_t = sample_gaussian(mu, sigma, n_samples=N_SAMPLES, generator=g)
        samples = samples_t.squeeze(-1).cpu().numpy()  # (N, n_test, H)

    return {
        "backbone": backbone_name,
        "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": y_pred.astype(np.float32),
        "samples": samples.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "mu": mu_np.astype(np.float32),
        "sigma": sigma_np.astype(np.float32),
        "train_seconds": time.time() - t0,
    }


# ============================================================
# Metrics (same as exp 22)
# ============================================================

def pinball_loss_array(truth, q_pred, q_level):
    diff = truth - q_pred
    return float(np.maximum(q_level * diff, (q_level - 1) * diff).mean())


def winkler_interval_score(truth, lower, upper, alpha):
    score = (upper - lower)
    score += (2.0 / alpha) * np.maximum(0, lower - truth)
    score += (2.0 / alpha) * np.maximum(0, truth - upper)
    return float(score.mean())


def coverage_at_level(truth, lower, upper):
    return float(np.mean((truth >= lower) & (truth <= upper)))


def compute_calibration_data(samples, truth, quantiles):
    pred_levels, emp_levels = [], []
    for q in quantiles:
        lower = np.quantile(samples, (1 - q) / 2, axis=0)
        upper = np.quantile(samples, (1 + q) / 2, axis=0)
        cov = np.mean((truth >= lower) & (truth <= upper))
        pred_levels.append(q)
        emp_levels.append(cov)
    return np.array(pred_levels), np.array(emp_levels)


def diebold_mariano_test(truth, pred1, pred2, horizon):
    e1 = (truth - pred1) ** 2
    e2 = (truth - pred2) ** 2
    d = (e1 - e2).flatten()
    d = d - d.mean()
    n = len(d)
    h_lag = max(horizon - 1, 1)
    gamma0 = np.sum(d * d) / n
    var_d = gamma0
    for k in range(1, h_lag + 1):
        gamma_k = np.sum(d[k:] * d[:-k]) / n
        var_d += 2.0 * (1.0 - k / (h_lag + 1)) * gamma_k
    var_d = max(var_d, 1e-12)
    mean_d = float(e1.mean() - e2.mean())
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(dm_stat), df=n - 1))
    return {
        "dm_stat": float(dm_stat), "p_value": float(p_value),
        "mean_diff_se": mean_d, "n": int(n), "h_lag": int(h_lag),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 34: Single-Gaussian density head × 4 backbones (240 tasks)")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)}")

    folds_by_h = {h: build_folds(series, SEQ_LEN, h) for h in HORIZONS}
    for h in HORIZONS:
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
    print(f"[backbones] {BACKBONES}")
    n_tasks = len(BACKBONES) * 5 * 4 * 3
    print(f"[tasks] {n_tasks} total")

    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for bb in BACKBONES:
                for seed in SEEDS:
                    tasks.append((bb, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    print(f"[parallel] n_jobs={n_jobs}, device={DEVICE}")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_save_predictions)(t) for t in tasks
    )
    print(f"[parallel] all done in {time.time() - t_start:.1f}s")

    # Aggregate by (backbone, horizon, fold)
    by_vhf = {}
    for r in results_list:
        key = (r["backbone"], r["horizon"], r["fold"])
        if key not in by_vhf:
            by_vhf[key] = {
                "preds_sum": np.zeros_like(r["preds"], dtype=np.float64),
                "samples_list": [],
                "truth": r["truth"],
                "n_seeds": 0,
            }
        by_vhf[key]["preds_sum"] += r["preds"]
        by_vhf[key]["samples_list"].append(r["samples"])
        by_vhf[key]["n_seeds"] += 1
    for key in by_vhf:
        by_vhf[key]["preds_mean"] = by_vhf[key]["preds_sum"] / by_vhf[key]["n_seeds"]
        by_vhf[key]["samples_all"] = np.concatenate(by_vhf[key]["samples_list"], axis=0)

    # Compute metrics
    print("\n[metrics] Computing extended metrics per (backbone, horizon, fold)...")
    metrics = {f"h{h}": {} for h in HORIZONS}
    calibration_data = {f"h{h}": {} for h in HORIZONS}

    # CRPS using energy form
    for h in HORIZONS:
        for bb in BACKBONES:
            fold_metrics = []
            for fi in range(len(folds_by_h[h])):
                key = (bb, h, fi)
                if key not in by_vhf:
                    continue
                data = by_vhf[key]
                preds = data["preds_mean"]
                truth = data["truth"]
                samples = data["samples_all"]

                mae = float(mean_absolute_error(truth, preds))
                half = samples.shape[0] // 2
                crps = float(
                    np.mean(np.abs(samples - truth[None]), axis=0).mean() -
                    0.5 * np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0).mean()
                )
                pinball = {}
                for q in QUANTILES:
                    q_pred = np.quantile(samples, q, axis=0)
                    pinball[f"pinball_{q:.2f}"] = pinball_loss_array(truth, q_pred, q)
                winkler, coverage = {}, {}
                for alpha in WINKLER_LEVELS:
                    q_lo = (1 - alpha) / 2
                    q_hi = (1 + alpha) / 2
                    lower = np.quantile(samples, q_lo, axis=0)
                    upper = np.quantile(samples, q_hi, axis=0)
                    winkler[f"winkler_{alpha:.2f}"] = winkler_interval_score(truth, lower, upper, 1 - alpha)
                    coverage[f"coverage_{alpha:.2f}"] = coverage_at_level(truth, lower, upper)
                pred_lev, emp_lev = compute_calibration_data(samples, truth, [0.5, 0.8, 0.9, 0.95])

                fold_metrics.append({
                    "fold": fi, "mae": mae, "crps": crps,
                    "pinball": pinball, "winkler": winkler, "coverage": coverage,
                })
            n_f = len(fold_metrics)
            if n_f == 0:
                continue
            agg = {
                "mae": np.mean([m["mae"] for m in fold_metrics]),
                "crps": np.mean([m["crps"] for m in fold_metrics]),
                "crps_per_fold": [m["crps"] for m in fold_metrics],
                "mae_per_fold": [m["mae"] for m in fold_metrics],
                "pinball": {},
                "winkler": {},
                "coverage": {},
                "n_folds": n_f,
            }
            for q in QUANTILES:
                k = f"pinball_{q:.2f}"
                agg["pinball"][k] = np.mean([m["pinball"][k] for m in fold_metrics])
            for lev in WINKLER_LEVELS:
                k1, k2 = f"winkler_{lev:.2f}", f"coverage_{lev:.2f}"
                agg["winkler"][k1] = np.mean([m["winkler"][k1] for m in fold_metrics])
                agg["coverage"][k2] = np.mean([m["coverage"][k2] for m in fold_metrics])
            metrics[f"h{h}"][bb] = agg
            calibration_data[f"h{h}"][bb] = {
                "pred_levels": pred_lev.tolist(),
                "emp_levels": emp_lev.tolist(),
            }

    # CRPS-Skill-Score vs TimesNet-point baseline (load exp 22 for baseline)
    print("\n[CRPS-Skill-Score] loading exp 22 baseline (TimesNet_point CRPS)...")
    exp22_path = RESULTS_DIR / "22_sota_comparison.json"
    with open(exp22_path) as f:
        exp22 = json.load(f)
    baseline_crps = {f"h{h}": exp22["metrics"][f"h{h}"]["TimesNet_point"]["crps"] for h in HORIZONS}

    for h in HORIZONS:
        for bb in BACKBONES:
            model_crps = metrics[f"h{h}"][bb]["crps"]
            skill = 1.0 - model_crps / baseline_crps[f"h{h}"]
            metrics[f"h{h}"][bb]["crps_skill_score_vs_TimesNet_point"] = float(skill)

    # DM test: each Gaussian backbone vs TimesNet-point (load exp 22 preds for TimesNet_point)
    # We don't have TimesNet_point preds here — DM test against TimesNet_gmm as proxy for now,
    # which is the most useful comparison for the paper.
    # (Loading exp 22 preds per-fold is heavy; we compute a CRPS-Skill-Score test instead.)

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "quantiles": QUANTILES, "winkler_levels": WINKLER_LEVELS,
            "epochs": EPOCHS, "lr": LR, "device": DEVICE,
            "n_jobs": n_jobs, "n_total_tasks": n_tasks,
            "wall_clock_s": time.time() - t_start,
            "backbones": BACKBONES,
            "head": "gaussian",
        },
        "metrics": metrics,
        "calibration": calibration_data,
        "baseline_crps_TimesNet_point": baseline_crps,
    }
    out_path = RESULTS_DIR / "34_gaussian_head.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" Single-Gaussian head summary (CRPS-Skill-Score vs TimesNet_point)")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n--- h={h} ---")
        print(f"  {'Backbone':<15} {'MAE':>10} {'CRPS':>10} {'CRPS-SS':>10} {'Coverage@0.90':>15}")
        for bb in BACKBONES:
            v = metrics[f"h{h}"].get(bb, {})
            if not v:
                continue
            mae = v.get("mae", float("nan"))
            crps = v.get("crps", float("nan"))
            ss = v.get("crps_skill_score_vs_TimesNet_point", 0.0) * 100
            cov = v.get("coverage", {}).get("coverage_0.90", float("nan"))
            print(f"  {bb:<15} {mae:>10.5f} {crps:>10.5f} {ss:>+10.2f}% {cov:>15.4f}")


if __name__ == "__main__":
    main()
