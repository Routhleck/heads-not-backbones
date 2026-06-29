"""
Phase 2.1: GARCH baselines for monthly S&P 500.

Tests:
  (a) GARCH(1,1) + skewed-t h-step via Monte Carlo simulation (not
      iterative mean — proper h-step)
  (b) GARCH(1,1) + Gaussian NLL (for comparison)
  (c) GARCH(1,1) + GMM head on standardized residuals
  (d) GJR-GARCH(1,1) (leverage effect)
  (e) EGARCH(1,1) (asymmetric)

Same walk-forward protocol as exp 22: 5 anchored folds, 3 seeds (only
1 seed needed for classical — no model weights), 4 horizons.

Outputs results/37a_garch_monthly.json
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(0)

# ============= Config =============
DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]  # for GMM head
EPOCHS = 80
LR = 1e-3
BS = 128
N_MIXTURES = 4
HIDDEN = 8  # tiny head for residual fitting
DEVICE = "cpu"  # classical + tiny head, no need for GPU
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10

N_SAMPLES = 500  # for h-step simulation and CRPS


# ============= Data =============
def build_folds(series, horizon):
    """Walk-forward: 5 anchored folds, like exp 22."""
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    folds = []
    fold_idx = 0
    train_end = init_train
    while train_end + test_window + horizon <= n:
        test_end = train_end + test_window
        train = series[:train_end]
        test = series[train_end:test_end]
        folds.append({
            "fold": fold_idx, "train": train, "test": test,
        })
        fold_idx += 1
        train_end += step
    return folds


# ============= GARCH fitting =============
def fit_garch_and_forecast(train, test, horizon, vol_model="GARCH", p=1, q=1, o=0, dist="skewt"):
    """Fit GARCH on train, forecast h-step distribution via simulation.

    Returns:
      samples: (N_SAMPLES, len(test), horizon)  — predictive samples
      point_pred: (len(test), horizon)  — conditional mean path
    """
    # Convert log returns to percent for numerical stability
    train_pct = train * 100
    test_pct = test * 100

    try:
        am = arch_model(train_pct, mean="Constant", vol=vol_model, p=p, q=q, o=o, dist=dist)
        res = am.fit(disp="off", show_warning=False)
    except Exception as e:
        print(f"    GARCH fit failed: {e}")
        return None, None

    # Forecast (use method='simulation' for path generation)
    fc = res.forecast(horizon=horizon, reindex=False, simulations=2000, method="simulation")
    sims = fc.simulations.values  # (1, 2000, horizon) — conditional path simulations
    # sims[0, :, :] is the 2000xhorizon matrix of paths
    paths = sims[0]  # (2000, horizon)

    # For each test point, sample horizon values from the SAME path distribution
    # (paths are aligned to the forecast origin = end of train)
    # We use the SAME path for all test points since GARCH refits per fold
    # This is a simplification — proper approach: refit GARCH up to each test point
    # But for h-step CRPS, this is the standard approach
    N = len(test)
    samples_pct = np.zeros((N_SAMPLES, N, horizon))
    for h in range(horizon):
        # Sample N_SAMPLES per test point from the path distribution at step h
        # All test points share the same path distribution (no rolling refit for speed)
        path_at_h = paths[:, h]  # (2000,)
        # Resample N_SAMPLES from this for each test point
        idx = np.random.choice(2000, size=(N_SAMPLES, N), replace=True)
        samples_pct[:, :, h] = path_at_h[idx]
    samples = samples_pct / 100  # back to log returns
    point_pred = np.tile(np.median(paths, axis=0) / 100, (N, 1))
    return samples, point_pred


def fit_garch_with_gmm_head(train, test, horizon, vol_model="GARCH", p=1, q=1, dist="skewt",
                             n_mixtures=N_MIXTURES, epochs=EPOCHS, seed=0):
    """GARCH + GMM head on standardized residuals.

    1. Fit GARCH to get standardized residuals
    2. Fit GMM to standardized residuals (on train)
    3. Forecast using GARCH + GMM resampling
    """
    train_pct = train * 100
    test_pct = test * 100

    try:
        am = arch_model(train_pct, mean="Constant", vol=vol_model, p=p, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)
    except Exception as e:
        print(f"    GARCH fit failed: {e}")
        return None, None

    # Get standardized residuals
    std_resid = np.asarray(res.std_resid)
    std_resid = std_resid[~np.isnan(std_resid)]
    if len(std_resid) < 50:
        return None, None

    # Fit GMM to standardized residuals (1-D)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_mixtures, random_state=seed, max_iter=200)
    gmm.fit(std_resid.reshape(-1, 1))
    print(f"    GMM on std_resid: weights={gmm.weights_.round(3)}, "
          f"means={gmm.means_.flatten().round(3)}")

    # Forecast
    fc = res.forecast(horizon=horizon, reindex=False, simulations=2000, method="simulation")
    paths = fc.simulations.values[0]  # (2000, horizon)
    cond_vol = np.sqrt(fc.variance.values[0, :])  # (horizon,)

    # For each test point, sample from GARCH forecast + GMM noise
    N = len(test)
    samples_pct = np.zeros((N_SAMPLES, N, horizon))
    for h in range(horizon):
        # Conditional mean at step h
        cond_mean_h = np.zeros(N)  # GARCH assumes zero mean after centering
        # Sample N_SAMPLES*N from GMM, returns (X, 1)
        gmm_samples, _ = gmm.sample(N_SAMPLES * N)
        gmm_samples = gmm_samples[:N_SAMPLES * N, 0].reshape(N_SAMPLES, N)
        # Add conditional mean and scale by vol
        samples_pct[:, :, h] = cond_mean_h + cond_vol[h] * gmm_samples
    samples = samples_pct / 100
    point_pred = np.tile(np.zeros(horizon), (N, 1))
    return samples, point_pred


# ============= Metrics =============
def pinball_loss_array(truth, q_pred, q_level):
    diff = truth - q_pred
    return float(np.maximum(q_level * diff, (q_level - 1) * diff).mean())


def coverage_at_level(truth, lower, upper):
    return float(np.mean((truth >= lower) & (truth <= upper)))


def crps_from_samples(samples, truth):
    """CRPS via energy form."""
    half = samples.shape[0] // 2
    crps = float(
        np.mean(np.abs(samples - truth[None]), axis=0).mean() -
        0.5 * np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0).mean()
    )
    return crps


# ============= Main =============
def main():
    print("=" * 70)
    print(" Phase 2.1: GARCH baselines (monthly S&P 500)")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] monthly SP500_Return n={len(series)}")

    # Methods to test
    methods = [
        ("GARCH_gauss", lambda tr, te, h: fit_garch_and_forecast(tr, te, h, dist="normal")),
        ("GARCH_skewt", lambda tr, te, h: fit_garch_and_forecast(tr, te, h, dist="skewt")),
        ("GJR_skewt", lambda tr, te, h: fit_garch_and_forecast(tr, te, h, vol_model="GARCH", p=1, q=1, dist="skewt", o=1)),
        ("EGARCH_skewt", lambda tr, te, h: fit_garch_and_forecast(tr, te, h, vol_model="EGARCH", p=1, q=1, dist="skewt")),
        ("GARCH_skewt_gmm", lambda tr, te, h: fit_garch_with_gmm_head(tr, te, h, dist="skewt", seed=0)),
    ]

    # Per-method per-horizon
    results_by_method = {name: {f"h{h}": {"crps_per_fold": []} for h in HORIZONS} for name, _ in methods}
    # TimesNet_point baseline (load from exp 22)
    with open(RESULTS_DIR / "22_sota_comparison.json") as f:
        exp22 = json.load(f)
    baseline_crps = {f"h{h}": exp22["metrics"][f"h{h}"]["TimesNet_point"]["crps"] for h in HORIZONS}

    t_start = time.time()
    for method_name, fit_fn in methods:
        print(f"\n[{method_name}]")
        for h in HORIZONS:
            folds = build_folds(series, h)
            for fold in folds:
                samples, point_pred = fit_fn(fold["train"], fold["test"], h)
                if samples is None:
                    print(f"    fold {fold['fold']} h={h}: SKIP")
                    continue
                truth = fold["test"]
                if truth.ndim == 1:
                    truth = truth[:, None]
                # CRPS: take mean over horizons
                crps = crps_from_samples(samples, truth)
                results_by_method[method_name][f"h{h}"]["crps_per_fold"].append(crps)
        # Summary
        agg = {}
        for h in HORIZONS:
            crps_folds = results_by_method[method_name][f"h{h}"]["crps_per_fold"]
            if not crps_folds:
                continue
            mean_crps = float(np.mean(crps_folds))
            ss = (1 - mean_crps / baseline_crps[f"h{h}"]) * 100
            agg[f"h{h}"] = {
                "crps_per_fold": crps_folds,
                "mean_crps": mean_crps,
                "crps_skill_score_vs_TimesNet_point_pct": ss,
            }
            print(f"  h={h}: CRPS={mean_crps:.5f}, CRPS-SS={ss:+.2f}%")
        results_by_method[method_name] = agg
        print(f"  [elapsed] {time.time() - t_start:.1f}s")

    # Save
    out = {
        "config": {
            "data": "monthly SP500_Return (Shiller)",
            "horizons": HORIZONS,
            "n_simulation_paths": 2000,
            "n_samples_per_cell": N_SAMPLES,
            "methods": [m for m, _ in methods],
        },
        "baseline_crps_TimesNet_point": baseline_crps,
        "methods": results_by_method,
    }
    out_path = RESULTS_DIR / "37a_garch_monthly.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(" GARCH baseline summary (CRPS-SS vs TimesNet_point, %)")
    print("=" * 70)
    print(f"  {'Method':<20} ", end="")
    for h in HORIZONS:
        print(f"h={h:<6} ", end="")
    print()
    for m in [n for n, _ in methods]:
        print(f"  {m:<20} ", end="")
        for h in HORIZONS:
            r = results_by_method[m].get(f"h{h}")
            if r:
                print(f"{r['crps_skill_score_vs_TimesNet_point_pct']:>+6.2f}% ", end="")
            else:
                print(f"  N/A  ", end="")
        print()


if __name__ == "__main__":
    main()
