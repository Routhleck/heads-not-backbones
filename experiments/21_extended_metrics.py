"""
Experiment 21: Extended metrics suite for paper polish.

Adds the standard probabilistic-forecasting evaluation metrics that mainstream papers
(DeepAR, MQ-RNN, TFT, M-competitions) use. Re-uses the same walk-forward scaffold as exp 19
but saves per-window predictions for post-hoc analysis.

New metrics (vs. exp 19):
- MASE: scale-free, comparable across series (N-BEATS / M-competition standard)
- Winkler interval score @ 50/80/90/95% coverage (CRPS origin papers)
- Empirical coverage @ each nominal level (MQ-RNN / TFT standard)
- CRPS-Skill-Score: 1 - CRPS_model/CRPS_baseline (interpretable relative metric)
- Diebold-Mariano test p-values (finance/econometrics standard)
- 9-quantile Pinball: 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99 (MQ-RNN standard)
- Calibration data: predicted vs empirical coverage (for reliability diagram)
- PIT data: probability integral transform samples (for PIT histogram)

Figures (generated after run):
- Calibration plot (reliability diagram)
- PIT histogram
- Forecast fan chart (P10/P50/P90 with truth)
- Model comparison heatmap (models × metrics)
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
from src.models.wpn import gmm_nll, gmm_point_predict, crps_gmm, sample_gmm

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
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
HIDDEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Quantile spectrum (MQ-RNN standard)
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

# Winkler interval score coverage levels
WINKLER_LEVELS = [0.50, 0.80, 0.90, 0.95]


# ============================================================
# Models (same as exp 19)
# ============================================================

class TimesNetGMMHead(nn.Module):
    def __init__(self, seq_len, horizon, n_mixtures=4, top_k=2, hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.top_k = top_k
        self.hidden = hidden
        self.n_mixtures = n_mixtures
        self.periods = []
        self.conv2d = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.gmm_proj = nn.Linear(hidden, horizon * n_mixtures * 3)

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
        params = self.gmm_proj(agg).view(B, self.horizon, 1, self.n_mixtures, 3)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        logit_pi = params[..., 2]
        sigma = F.softplus(log_sigma) + 1e-3
        pi = F.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


class TimesNetBaseline(nn.Module):
    def __init__(self, seq_len, horizon, top_k=2, hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.top_k = top_k
        self.hidden = hidden
        self.periods = []
        self.conv2d = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden, horizon)

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
        return self.proj(agg).unsqueeze(-1)


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
        folds.append({
            "fold": fold_idx,
            "X_train": X_all[:train_end_pair], "Y_train": Y_all[:train_end_pair],
            "X_test": X_all[train_end_pair:test_end_pair], "Y_test": Y_all[train_end_pair:test_end_pair],
        })
        fold_idx += 1
        train_end_series += step
    return folds


# ============================================================
# Worker: train + save predictions + GMM samples
# ============================================================

def train_and_save_predictions(args):
    model_name, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    use_gmm = model_name == "TimesNet_GMM"
    if model_name == "TimesNet":
        model = TimesNetBaseline(seq_len=SEQ_LEN, horizon=horizon, top_k=2, hidden=64).to(DEVICE)
    else:
        model = TimesNetGMMHead(seq_len=SEQ_LEN, horizon=horizon, n_mixtures=4,
                                 top_k=2, hidden=64).to(DEVICE)

    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    if use_gmm:
        crit = lambda yp, yt: gmm_nll(yt, yp[0], yp[1], yp[2])
    else:
        crit = nn.HuberLoss(delta=1.0)

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(X[idx])
            loss = crit(out, Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        out = model(Xt)

    # Compute predictions and samples
    if use_gmm:
        mu, sigma, pi = out
        y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, H)
        # Proper GMM sampling: per-(test, h) mixture weights via composition
        N_SAMPLES = 500
        with torch.no_grad():
            samples_t = sample_gmm(mu, sigma, pi, n_samples=N_SAMPLES)  # (N, B, H, 1)
        # sample_gmm returns (N, B, H, C); squeeze C -> (N, n_test, H)
        samples = samples_t.squeeze(-1).cpu().numpy()
    else:
        y_pred = out.cpu().numpy().squeeze(-1)  # (n_test, H)
        # Compute proper training residuals for honest Gaussian approximation
        with torch.no_grad():
            train_pred = model(X).squeeze(-1).cpu().numpy()  # (n_train, H)
        train_resid = Y_train - train_pred  # (n_train, H)
        sigma_per_step = np.std(train_resid, axis=0)  # (H,)
        # Broadcast: (N, n_test, H) draws from N(pred, sigma_per_step)
        N_SAMPLES = 500
        np.random.seed(seed)
        samples = np.random.normal(
            y_pred[None, :, :],
            sigma_per_step[None, None, :],
            size=(N_SAMPLES, y_pred.shape[0], y_pred.shape[1]),
        )

    return {
        "model": model_name, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": y_pred.astype(np.float32),  # (n_test, H)
        "samples": samples.astype(np.float32),  # (N_SAMPLES, n_test, H)
        "truth": Y_test.astype(np.float32),  # (n_test, H)
        "train_seconds": time.time() - t0,
    }


# ============================================================
# Extended metric computations
# ============================================================

def mase_per_window(truth, pred, X_test):
    """MASE = MAE_model / MAE_naive, where naive = last value of each X_test window.

    Standard M-competition / N-BEATS / MQ-RNN definition:
      naive_pred[i, h] = X_test[i, -1]  (last input value held forward)
      mae_naive = mean |truth - naive_pred|
    Args:
        truth: (n_test, H)
        pred: (n_test, H) model forecast
        X_test: (n_test, T) raw input windows
    Returns:
        MASE = mae_model / mae_naive
    """
    n_test, h = truth.shape
    naive_pred = X_test[:, -1:].repeat(h, axis=1)  # (n_test, H)
    mae_model = np.mean(np.abs(truth - pred))
    mae_naive = np.mean(np.abs(truth - naive_pred))
    if mae_naive < 1e-8:
        return float("nan")
    return float(mae_model / mae_naive)


def pinball_loss_array(truth, q_pred, q_level):
    """Pinball loss for a single quantile level. truth, q_pred: (n, H)."""
    diff = truth - q_pred
    return np.maximum(q_level * diff, (q_level - 1) * diff).mean()


def winkler_interval_score(truth, lower, upper, alpha):
    """Winkler interval score at coverage level 1-alpha.

    IS = (u - l) + (2/alpha) * max(0, l - y) + (2/alpha) * max(0, y - u)
    Lower IS = better.
    """
    score = (upper - lower)
    score += (2.0 / alpha) * np.maximum(0, lower - truth)
    score += (2.0 / alpha) * np.maximum(0, truth - upper)
    return float(score.mean())


def coverage_at_level(truth, lower, upper):
    """Empirical coverage: fraction of truth points inside [lower, upper]."""
    return float(np.mean((truth >= lower) & (truth <= upper)))


def diebold_mariano_test(truth, pred1, pred2, horizon):
    """Diebold-Mariano test: are forecast 1 and forecast 2 significantly different?"""
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
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff_se": mean_d,
        "n": int(n),
        "h_lag": int(h_lag),
    }


def compute_calibration_data(samples, truth, quantiles):
    """Compute predicted vs empirical coverage for a set of nominal quantile levels.

    Returns:
        pred_levels: nominal quantile levels
        emp_levels: empirical coverage
    For a well-calibrated model, emp_levels == pred_levels.
    """
    n_samples, n_test, H = samples.shape
    pred_levels = []
    emp_levels = []
    for q in quantiles:
        lower = np.quantile(samples, (1 - q) / 2, axis=0)
        upper = np.quantile(samples, (1 + q) / 2, axis=0)
        cov = np.mean((truth >= lower) & (truth <= upper))
        pred_levels.append(q)
        emp_levels.append(cov)
    return np.array(pred_levels), np.array(emp_levels)


def pit_values(samples, truth):
    """PIT (Probability Integral Transform) values.

    For each truth point, PIT = fraction of samples <= truth.
    A well-calibrated model has uniform PIT distribution.
    """
    n_samples, n_test, H = samples.shape
    pit = np.zeros((n_test, H))
    for i in range(n_test):
        for h in range(H):
            pit[i, h] = np.mean(samples[:, i, h] <= truth[i, h])
    return pit.flatten()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 21: Extended metrics suite (MASE, Winkler, Coverage, DM, PIT)")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)}")

    folds_by_h = {h: build_folds(series, SEQ_LEN, h) for h in HORIZONS}
    for h in HORIZONS:
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    MODELS = ["TimesNet", "TimesNet_GMM"]
    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for model_name in MODELS:
                for seed in SEEDS:
                    tasks.append((model_name, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))
    print(f"[tasks] {len(tasks)}")

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    print(f"[parallel] n_jobs={n_jobs}")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_save_predictions)(t) for t in tasks
    )
    print(f"[parallel] all done in {time.time() - t_start:.1f}s")

    # Aggregate by (model, h, fold):
    #   - point predictions: average across seeds (ensemble of point forecasts)
    #   - samples: concatenate across seeds (gives 3x more samples for stable quantile estimates)
    by_mhf = {}
    for r in results_list:
        key = (r["model"], r["horizon"], r["fold"])
        if key not in by_mhf:
            by_mhf[key] = {
                "preds_sum": np.zeros_like(r["preds"], dtype=np.float64),
                "samples_list": [],
                "truth": r["truth"],
                "n_seeds": 0,
            }
        by_mhf[key]["preds_sum"] += r["preds"]
        by_mhf[key]["samples_list"].append(r["samples"])
        by_mhf[key]["n_seeds"] += 1

    # Average predictions across seeds; concatenate samples for distribution metrics
    for key in by_mhf:
        by_mhf[key]["preds_mean"] = by_mhf[key]["preds_sum"] / by_mhf[key]["n_seeds"]
        by_mhf[key]["samples_all"] = np.concatenate(by_mhf[key]["samples_list"], axis=0)

    # ============================================================
    # Compute extended metrics per (model, h)
    # ============================================================
    print("\n[metrics] Computing extended metrics...")

    metrics = {f"h{h}": {} for h in HORIZONS}
    calibration_data = {f"h{h}": {} for h in HORIZONS}
    pit_data = {f"h{h}": {} for h in HORIZONS}

    for h in HORIZONS:
        for model_name in MODELS:
            fold_metrics = []
            for fi in range(len(folds_by_h[h])):
                key = (model_name, h, fi)
                if key not in by_mhf:
                    continue
                data = by_mhf[key]
                preds = data["preds_mean"]
                truth = data["truth"]
                samples = data["samples_all"]  # (3*N, n_test, H) — concatenated across seeds

                # Standard MAE
                mae = float(mean_absolute_error(truth, preds))
                # MASE (naive = last X_test value held forward)
                X_test = folds_by_h[h][fi]["X_test"]
                mase = mase_per_window(truth, preds, X_test)

                # Pinball at 9 quantiles (from samples)
                pinball_full = {}
                for q in QUANTILES:
                    q_pred = np.quantile(samples, q, axis=0)  # (n_test, H)
                    pinball_full[f"pinball_{q:.2f}"] = pinball_loss_array(truth, q_pred, q)

                # Winkler + coverage at 4 levels
                winkler = {}
                coverage = {}
                for alpha in WINKLER_LEVELS:
                    q_lo = (1 - alpha) / 2
                    q_hi = (1 + alpha) / 2
                    lower = np.quantile(samples, q_lo, axis=0)
                    upper = np.quantile(samples, q_hi, axis=0)
                    winkler[f"winkler_{alpha:.2f}"] = winkler_interval_score(truth, lower, upper, 1 - alpha)
                    coverage[f"coverage_{alpha:.2f}"] = coverage_at_level(truth, lower, upper)

                # CRPS: E|X - y| - 0.5 * E|X - X'|  (first-half vs second-half estimator)
                half = samples.shape[0] // 2
                crps_samples = float(
                    np.mean(np.abs(samples - truth[None]), axis=0).mean() -
                    0.5 * np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0).mean()
                )

                # Calibration data
                pred_lev, emp_lev = compute_calibration_data(samples, truth,
                                                              [0.5, 0.8, 0.9, 0.95])

                # PIT values
                pit = pit_values(samples, truth)

                fold_metrics.append({
                    "fold": fi,
                    "mae": mae, "mase": mase, "crps": float(crps_samples),
                    "pinball": pinball_full, "winkler": winkler, "coverage": coverage,
                })

            # Aggregate across folds
            n_f = len(fold_metrics)
            agg = {
                "mae": np.mean([m["mae"] for m in fold_metrics]),
                "mase": np.mean([m["mase"] for m in fold_metrics]),
                "crps": np.mean([m["crps"] for m in fold_metrics]),
                "pinball": {},
                "winkler": {},
                "coverage": {},
                "n_folds": n_f,
            }
            for q in QUANTILES:
                key = f"pinball_{q:.2f}"
                agg["pinball"][key] = np.mean([m["pinball"][key] for m in fold_metrics])
            for lev in WINKLER_LEVELS:
                k1, k2 = f"winkler_{lev:.2f}", f"coverage_{lev:.2f}"
                agg["winkler"][k1] = np.mean([m["winkler"][k1] for m in fold_metrics])
                agg["coverage"][k2] = np.mean([m["coverage"][k2] for m in fold_metrics])
            metrics[f"h{h}"][model_name] = agg
            calibration_data[f"h{h}"][model_name] = {"pred_levels": pred_lev.tolist(),
                                                     "emp_levels": emp_lev.tolist()}
            pit_data[f"h{h}"][model_name] = pit.tolist()

    # CRPS-Skill-Score: relative improvement vs TimesNet baseline
    print("\n[CRPS-Skill-Score] TimesNet+GMM vs TimesNet baseline")
    for h in HORIZONS:
        crps_baseline = metrics[f"h{h}"]["TimesNet"]["crps"]
        crps_model = metrics[f"h{h}"]["TimesNet_GMM"]["crps"]
        skill = 1.0 - crps_model / crps_baseline
        metrics[f"h{h}"]["TimesNet_GMM"]["crps_skill_score"] = float(skill)
        print(f"  h={h}: baseline CRPS={crps_baseline:.6f}, +GMM CRPS={crps_model:.6f}, "
              f"skill score={skill*100:+.2f}%")

    # Diebold-Mariano test: TimesNet vs TimesNet_GMM
    print("\n[DM test] TimesNet vs TimesNet+GMM (lower = +GMM better)")
    dm_results = {f"h{h}": {} for h in HORIZONS}
    for h in HORIZONS:
        # Concatenate per-fold predictions and truths
        all_truth = []
        all_p1, all_p2 = [], []
        for fi in range(len(folds_by_h[h])):
            k1 = ("TimesNet", h, fi)
            k2 = ("TimesNet_GMM", h, fi)
            if k1 in by_mhf and k2 in by_mhf:
                all_truth.append(by_mhf[k1]["truth"])
                all_p1.append(by_mhf[k1]["preds_mean"])
                all_p2.append(by_mhf[k2]["preds_mean"])
        if all_truth:
            truth_c = np.concatenate(all_truth, axis=0)
            p1_c = np.concatenate(all_p1, axis=0)
            p2_c = np.concatenate(all_p2, axis=0)
            dm = diebold_mariano_test(truth_c, p1_c, p2_c, horizon=h)
            dm_results[f"h{h}"] = dm
            sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else ("*" if dm["p_value"] < 0.10 else ""))
            print(f"  h={h}: DM stat={dm['dm_stat']:.3f}, p={dm['p_value']:.4f} {sig}")

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "quantiles": QUANTILES, "winkler_levels": WINKLER_LEVELS,
            "epochs": EPOCHS, "lr": LR, "device": DEVICE,
            "n_jobs": n_jobs, "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start,
        },
        "metrics": metrics,
        "dm_test": dm_results,
        "calibration": calibration_data,
    }
    out_path = RESULTS_DIR / "21_extended_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Save PIT values as npz for plotting
    pit_arrays = {}
    for h in HORIZONS:
        for m in MODELS:
            pit_arrays[f"pit_h{h}_{m}"] = np.array(pit_data[f"h{h}"][m])
    np.savez(RESULTS_DIR / "21_pit_values.npz", **pit_arrays)
    print(f"Saved {RESULTS_DIR / '21_pit_values.npz'}")

    # Save sample data for fan chart (h=12 fold 4 is most interesting — recent)
    fan_data = {}
    for h in [1, 3, 6, 12]:
        fi = min(4, len(folds_by_h[h]) - 1)  # last fold
        for m in MODELS:
            key = (m, h, fi)
            if key in by_mhf:
                fan_data[f"h{h}_{m}_samples"] = by_mhf[key]["samples_all"]
                fan_data[f"h{h}_{m}_truth"] = by_mhf[key]["truth"]
                fan_data[f"h{h}_{m}_preds"] = by_mhf[key]["preds_mean"]
    np.savez(RESULTS_DIR / "21_fan_chart_data.npz", **fan_data)
    print(f"Saved {RESULTS_DIR / '21_fan_chart_data.npz'}")

    # Print summary
    print("\n" + "=" * 70)
    print(" Extended metrics summary (mean over folds)")
    print("=" * 70)
    print(f"\n{'Horizon':<8} {'Metric':<14} {'TimesNet':>12} {'+GMM':>12}")
    for h in HORIZONS:
        for mname, mkey in [("MAE", "mae"), ("MASE", "mase"), ("CRPS", "crps"),
                              ("Pinball 0.05", "pinball"), ("Pinball 0.5", "pinball"),
                              ("Pinball 0.95", "pinball"), ("Winkler 0.90", "winkler"),
                              ("Coverage 0.90", "coverage")]:
            if mname == "MAE":
                v1 = metrics[f"h{h}"]["TimesNet"]["mae"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["mae"]
            elif mname == "MASE":
                v1 = metrics[f"h{h}"]["TimesNet"]["mase"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["mase"]
            elif mname == "CRPS":
                v1 = metrics[f"h{h}"]["TimesNet"]["crps"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["crps"]
            elif mname == "Pinball 0.05":
                v1 = metrics[f"h{h}"]["TimesNet"]["pinball"]["pinball_0.05"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["pinball"]["pinball_0.05"]
            elif mname == "Pinball 0.5":
                v1 = metrics[f"h{h}"]["TimesNet"]["pinball"]["pinball_0.50"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["pinball"]["pinball_0.50"]
            elif mname == "Pinball 0.95":
                v1 = metrics[f"h{h}"]["TimesNet"]["pinball"]["pinball_0.95"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["pinball"]["pinball_0.95"]
            elif mname == "Winkler 0.90":
                v1 = metrics[f"h{h}"]["TimesNet"]["winkler"]["winkler_0.90"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["winkler"]["winkler_0.90"]
            elif mname == "Coverage 0.90":
                v1 = metrics[f"h{h}"]["TimesNet"]["coverage"]["coverage_0.90"]
                v2 = metrics[f"h{h}"]["TimesNet_GMM"]["coverage"]["coverage_0.90"]
            print(f"  h={h:<4} {mname:<14} {v1:>12.5f} {v2:>12.5f}")


if __name__ == "__main__":
    main()
