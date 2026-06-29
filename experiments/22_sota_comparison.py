"""
Experiment 22: SOTA comparison on S&P 500 monthly log-returns.

Compares 4 backbones x 2 heads on the same walk-forward protocol as exp 19/21:
  Backbones: TimesNet, DLinear, N-BEATS, iTransformer
  Heads:     Huber (point forecast), GMM (density forecast)
  -> 8 model variants total.

iTransformer replaces PatchTST (see DECISION_PATCHTST_DROPPED.md
for the architectural incompatibility that motivated the swap).

Architecture (mirrors exp 19's clean design):
  - Backbone: x -> hidden features (B, hidden)
  - Head:     hidden -> output (Point: (B, H); GMM: (B, H, 1, K))
  - For *_point variants: head = Linear(hidden, H), loss = Huber
  - For *_gmm variants:   head = Linear(hidden, H*K*3) -> (mu, sigma, pi), loss = NLL

This is the missing mainline SOTA comparison: does the TimesNet + GMM story
hold up against modern forecasting baselines, or is it a fluke?

Metrics (same as exp 21):
  MAE, MASE, CRPS, 9-quantile Pinball, Winkler @ 4 levels, Coverage @ 4 levels,
  CRPS-Skill-Score vs TimesNet baseline, Diebold-Mariano p-values.

Protocol: 5 anchored walk-forward folds, 4 horizons (1, 3, 6, 12), 3 seeds.
Total tasks: 8 models x 5 folds x 4 horizons x 3 seeds = 480 tasks.

References:
  - TimesNet (Wu et al. 2023)
  - DLinear (Zeng et al. 2023)
  - N-BEATS (Oreshkin et al. 2020)
  - iTransformer (Liu et al. 2024)
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
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
N_MIXTURES = 4
HIDDEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
WINKLER_LEVELS = [0.50, 0.80, 0.90, 0.95]


# ============================================================
# Backbones — unified interface:
#   forward(x: (B, T, 1)) -> hidden: (B, hidden)
# ============================================================

class TimesNetBackbone(nn.Module):
    """TimesNet with AMPD-driven 2D-variation. Returns (B, hidden)."""

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
        return agg  # (B, hidden)


class DLinearBackbone(nn.Module):
    """DLinear: trend + seasonal decomposition, then linear per component.

    Zeng et al. 2023. https://arxiv.org/abs/2205.13502

    Returns (B, hidden) by combining trend+seasonal info through a small MLP.
    """
    def __init__(self, seq_len, hidden=HIDDEN, kernel=25):
        super().__init__()
        self.seq_len = seq_len
        self.kernel = kernel
        # Two feature extractors: one for trend, one for seasonal
        self.trend_feat = nn.Linear(seq_len, hidden)
        self.seasonal_feat = nn.Linear(seq_len, hidden)
        # Combine
        self.combine = nn.Linear(2 * hidden, hidden)

    def forward(self, x):
        # x: (B, T, 1)
        x = x.squeeze(-1)  # (B, T)
        # Moving average for trend
        trend = F.avg_pool1d(x.unsqueeze(1), self.kernel, stride=1,
                              padding=self.kernel // 2).squeeze(1)
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1]]
        seasonal = x - trend
        t = self.trend_feat(trend)  # (B, hidden)
        s = self.seasonal_feat(seasonal)  # (B, hidden)
        return self.combine(torch.cat([t, s], dim=-1))  # (B, hidden)


class ITransformerBackboneAdapter(nn.Module):
    """Thin adapter to wrap src.models.itransformer.ITransformerBackbone
    for the experiment script's (B, T, 1) -> (B, hidden) interface.
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
# Heads (mirror exp 19's design)
# ============================================================

class PointHead(nn.Module):
    """Linear projection: hidden -> (B, H)."""
    def __init__(self, hidden, horizon):
        super().__init__()
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, h):
        return self.proj(h)  # (B, H)


class GMMHead(nn.Module):
    """Linear projection: hidden -> (mu, sigma, pi) for K mixtures per (h, 1)."""
    def __init__(self, hidden, horizon, n_mixtures=N_MIXTURES):
        super().__init__()
        self.horizon = horizon
        self.n_mixtures = n_mixtures
        self.proj = nn.Linear(hidden, horizon * n_mixtures * 3)

    def forward(self, h):
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, self.n_mixtures, 3)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        logit_pi = params[..., 2]
        sigma = F.softplus(log_sigma) + 1e-3
        pi = F.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


# ============================================================
# Model registry
# ============================================================

def build_model(variant, seq_len, horizon):
    """Returns (backbone, head) where head is PointHead or GMMHead."""
    backbone_name, head_type = variant.split("_")
    if backbone_name == "TimesNet":
        backbone = TimesNetBackbone(seq_len=seq_len, hidden=HIDDEN, top_k=2)
    elif backbone_name == "DLinear":
        backbone = DLinearBackbone(seq_len=seq_len, hidden=HIDDEN, kernel=25)
    elif backbone_name == "NBEATS":
        backbone = NBEATSBackbone(seq_len=seq_len, hidden=HIDDEN, n_blocks=2, theta_dim=8)
    elif backbone_name == "iTransformer":
        backbone = ITransformerBackboneAdapter(seq_len=seq_len, hidden=HIDDEN,
                                               n_heads=4, n_layers=3, ff_dim=128)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if head_type == "point":
        head = PointHead(hidden=HIDDEN, horizon=horizon)
    elif head_type == "gmm":
        head = GMMHead(hidden=HIDDEN, horizon=horizon, n_mixtures=N_MIXTURES)
    else:
        raise ValueError(f"Unknown head: {head_type}")
    return backbone, head, head_type


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
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    backbone, head, head_type = build_model(variant, SEQ_LEN, horizon)
    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)

    # Move data
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    if head_type == "point":
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
            if head_type == "point":
                yp = out.unsqueeze(-1)  # (B, H, 1)
                loss = crit(yp, Y[idx])
            else:
                mu, sigma, pi = out  # (B, H, 1, K)
                loss = crit((mu, sigma, pi), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    # Predict
    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if head_type == "point":
            y_pred = out.cpu().numpy()  # (n_test, H)
            # Compute training residuals for Gaussian sample approximation
            h_train = backbone(X)
            train_pred = head(h_train).detach().cpu().numpy()  # (n_train, H)
            train_resid = Y_train - train_pred
            sigma_per_step = np.std(train_resid, axis=0)  # (H,)
            N_SAMPLES = 500
            np.random.seed(seed)
            samples = np.random.normal(
                y_pred[None, :, :], sigma_per_step[None, None, :],
                size=(N_SAMPLES, y_pred.shape[0], y_pred.shape[1]),
            )
        else:  # gmm
            mu, sigma, pi = out  # mu: (n_test, H, 1, K)
            y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, H)
            N_SAMPLES = 500
            samples_t = sample_gmm(mu, sigma, pi, n_samples=N_SAMPLES)
            samples = samples_t.squeeze(-1).cpu().numpy()  # (N, n_test, H)

    return {
        "variant": variant, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": y_pred.astype(np.float32),
        "samples": samples.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "head_type": head_type,
        "train_seconds": time.time() - t0,
    }


# ============================================================
# Extended metrics
# ============================================================

def mase_per_window(truth, pred, X_test):
    n_test, h = truth.shape
    naive_pred = X_test[:, -1:].repeat(h, axis=1)
    mae_model = np.mean(np.abs(truth - pred))
    mae_naive = np.mean(np.abs(truth - naive_pred))
    if mae_naive < 1e-8:
        return float("nan")
    return float(mae_model / mae_naive)


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
    print(" Experiment 22: SOTA comparison (4 backbones x 2 heads = 8 models)")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)}")

    folds_by_h = {h: build_folds(series, SEQ_LEN, h) for h in HORIZONS}
    for h in HORIZONS:
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
    HEADS = ["point", "gmm"]
    VARIANTS = [f"{b}_{h}" for b in BACKBONES for h in HEADS]
    print(f"[variants] {VARIANTS}")
    print(f"[tasks] {len(VARIANTS) * 5 * 4 * 3} total")

    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for variant in VARIANTS:
                for seed in SEEDS:
                    tasks.append((variant, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    print(f"[parallel] n_jobs={n_jobs}")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_save_predictions)(t) for t in tasks
    )
    print(f"[parallel] all done in {time.time() - t_start:.1f}s")

    # Aggregate
    by_vhf = {}
    for r in results_list:
        key = (r["variant"], r["horizon"], r["fold"])
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
    print("\n[metrics] Computing extended metrics per variant...")
    metrics = {f"h{h}": {} for h in HORIZONS}
    calibration_data = {f"h{h}": {} for h in HORIZONS}

    for h in HORIZONS:
        for variant in VARIANTS:
            fold_metrics = []
            for fi in range(len(folds_by_h[h])):
                key = (variant, h, fi)
                if key not in by_vhf:
                    continue
                data = by_vhf[key]
                preds = data["preds_mean"]
                truth = data["truth"]
                samples = data["samples_all"]
                X_test = folds_by_h[h][fi]["X_test"]

                mae = float(mean_absolute_error(truth, preds))
                mase = mase_per_window(truth, preds, X_test)
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
                    "fold": fi, "mae": mae, "mase": mase, "crps": crps,
                    "pinball": pinball, "winkler": winkler, "coverage": coverage,
                })
            n_f = len(fold_metrics)
            if n_f == 0:
                continue
            agg = {
                "mae": np.mean([m["mae"] for m in fold_metrics]),
                "mase": np.mean([m["mase"] for m in fold_metrics]),
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
            metrics[f"h{h}"][variant] = agg
            calibration_data[f"h{h}"][variant] = {
                "pred_levels": pred_lev.tolist(),
                "emp_levels": emp_lev.tolist(),
            }

    # CRPS-Skill-Score vs TimesNet_point
    print("\n[CRPS-Skill-Score] vs TimesNet_point baseline")
    for h in HORIZONS:
        baseline = metrics[f"h{h}"]["TimesNet_point"]["crps"]
        for variant in VARIANTS:
            model_crps = metrics[f"h{h}"][variant]["crps"]
            skill = 1.0 - model_crps / baseline
            metrics[f"h{h}"][variant]["crps_skill_score"] = float(skill)

    # DM test: each variant vs TimesNet_point
    print("\n[DM test] each variant vs TimesNet_point")
    dm_results = {f"h{h}": {} for h in HORIZONS}
    for h in HORIZONS:
        for variant in VARIANTS:
            if variant == "TimesNet_point":
                continue
            all_truth, all_p1, all_p2 = [], [], []
            for fi in range(len(folds_by_h[h])):
                k1 = ("TimesNet_point", h, fi)
                k2 = (variant, h, fi)
                if k1 in by_vhf and k2 in by_vhf:
                    all_truth.append(by_vhf[k1]["truth"])
                    all_p1.append(by_vhf[k1]["preds_mean"])
                    all_p2.append(by_vhf[k2]["preds_mean"])
            if all_truth:
                truth_c = np.concatenate(all_truth, axis=0)
                p1_c = np.concatenate(all_p1, axis=0)
                p2_c = np.concatenate(all_p2, axis=0)
                dm = diebold_mariano_test(truth_c, p1_c, p2_c, horizon=h)
                dm_results[f"h{h}"][variant] = dm

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "quantiles": QUANTILES, "winkler_levels": WINKLER_LEVELS,
            "epochs": EPOCHS, "lr": LR, "device": DEVICE,
            "n_jobs": n_jobs, "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start,
            "variants": VARIANTS,
        },
        "metrics": metrics,
        "dm_test": dm_results,
        "calibration": calibration_data,
    }
    out_path = RESULTS_DIR / "22_sota_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Summary print
    print("\n" + "=" * 70)
    print(" SOTA comparison summary (MAE + CRPS-Skill-Score vs TimesNet_point)")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n--- h={h} ---")
        print(f"  {'Variant':<20} {'MAE':>10} {'CRPS':>10} {'MASE':>8} {'CRPS-SS':>10} {'Coverage@0.90':>15}")
        for variant in VARIANTS:
            v = metrics[f"h{h}"].get(variant, {})
            if not v:
                continue
            mae = v.get("mae", float("nan"))
            crps = v.get("crps", float("nan"))
            mase = v.get("mase", float("nan"))
            ss = v.get("crps_skill_score", 0.0) * 100
            cov = v.get("coverage", {}).get("coverage_0.90", float("nan"))
            print(f"  {variant:<20} {mae:>10.5f} {crps:>10.5f} {mase:>8.4f} {ss:>+10.2f}% {cov:>15.4f}")


if __name__ == "__main__":
    main()
