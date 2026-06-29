"""
Experiment 26: Bootstrap confidence intervals for SOTA CRPS-Skill-Score.

Re-runs exp 22's protocol but saves per-fold per-seed CRPS, then bootstraps
to compute 95% CI on CRPS-SS for each variant × horizon.

Protocol (same as exp 22 / 23 / 25):
- 5 anchored walk-forward folds, 4 horizons {1, 3, 6, 12}, 3 seeds
- 8 variants (4 backbones × 2 heads) = 480 tasks
- ~10 min on RTX 4060, ~8 hours on Mac CPU

Bootstrap: sample (variant × horizon) observations with replacement from
the 5 folds × 3 seeds = 15 observations per cell. Compute CRPS-SS vs
TimesNet_point for each resample. 95% CI = [2.5%, 97.5%] percentiles.
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

# Import exp 22 model classes via importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp22", str(ROOT / "experiments" / "22_sota_comparison.py")
)
exp22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp22)

from src.models.wpn import crps_gmm, sample_gmm

warnings.filterwarnings("ignore")

# ---- Config (identical to exp 22) ----
DATA_PATH = exp22.DATA_PATH
RESULTS_DIR = exp22.RESULTS_DIR
SEQ_LEN = exp22.SEQ_LEN
HORIZONS = exp22.HORIZONS
SEEDS = exp22.SEEDS
INITIAL_TRAIN_FRAC = exp22.INITIAL_TRAIN_FRAC
TEST_FRAC = exp22.TEST_FRAC
STEP_FRAC = exp22.STEP_FRAC
EPOCHS = exp22.EPOCHS
LR = exp22.LR
BS = exp22.BS
HIDDEN = exp22.HIDDEN
DEVICE = exp22.DEVICE
N_MIXTURES = exp22.N_MIXTURES
N_SAMPLES_CRPS = 500  # samples for CRPS computation (matches exp 22)

# CRPS-Skill-Score baseline = TimesNet_point
BASELINE_VARIANT = "TimesNet_point"


def train_and_save_crps(variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test):
    """Train one task, return CRPS computed from samples (mirrors exp 22)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    use_gmm = variant.endswith("_gmm")
    backbone, head, _ = exp22.build_model(variant, SEQ_LEN, horizon)
    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    if use_gmm:
        crit = lambda yp, yt: exp22.gmm_nll(yt, yp[0], yp[1], yp[2])
    else:
        crit = nn.HuberLoss(delta=1.0)

    for ep in range(EPOCHS):
        backbone.train(); head.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            h = backbone(X[idx])
            out = head(h)
            if use_gmm:
                mu, sigma, pi = out
                loss = exp22.gmm_nll(Y[idx], mu, sigma, pi)
            else:
                yp = out.unsqueeze(-1)  # (B, H, 1)
                loss = nn.HuberLoss(delta=1.0)(yp, Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    # Inference: collect samples
    backbone.eval(); head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Yt = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n_test = Xt.shape[0]
    with torch.no_grad():
        h = backbone(Xt)
        out = head(h)
    if use_gmm:
        mu, sigma, pi = out
        # Sample from GMM: (B, H, 1, K) -> (n_samples, B, H, 1)
        samples = sample_gmm(mu, sigma, pi, n_samples=N_SAMPLES_CRPS).cpu().numpy()
        # samples shape: (N_SAMPLES, n_test, H, 1)
    else:
        # Gaussian-approx with train residual std
        train_resid_std = float(np.std(Y_train - 0))
        preds = out.cpu().numpy()  # (n_test, H)
        # Sample N_SAMPLES_CRPS from N(preds, train_resid_std) per test point
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((N_SAMPLES_CRPS, n_test, horizon, 1)) * train_resid_std
        samples = preds[None, :, :, None] + noise  # (N, n_test, H, 1)
    # Compute CRPS using exp 22's formula (np-based, robust)
    # Y_test is (n_test, H), need to add channel dim: (1, n_test, H, 1)
    Y_test_4d = Y_test[None, :, :, None] if Y_test.ndim == 2 else Y_test[None]
    half = N_SAMPLES_CRPS // 2
    term1 = np.mean(np.abs(samples - Y_test_4d), axis=0)         # (n_test, H, 1)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    crps = float((term1 - 0.5 * term2).mean())
    return {
        "variant": variant, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "crps": crps, "elapsed_sec": time.time() - t0,
    }


def main():
    print("=" * 70)
    print(" Experiment 26: Bootstrap CI for SOTA CRPS-Skill-Score")
    print("=" * 70)
    panel = pd.read_csv(DATA_PATH)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    n = len(series)

    # Build folds per horizon
    folds_by_h = {}
    for h in HORIZONS:
        folds_by_h[h] = exp22.build_folds(series, SEQ_LEN, h)
    print(f"[folds] {len(folds_by_h[HORIZONS[0]])} folds × {len(HORIZONS)} horizons × "
          f"{len(exp22.build_model.__globals__.get('VARIANTS', [])) or 8} variants × "
          f"{len(SEEDS)} seeds")

    # Hardcoded variants list (same as exp 22)
    VARIANTS = ["TimesNet_point", "TimesNet_gmm", "DLinear_point", "DLinear_gmm",
                "NBEATS_point", "NBEATS_gmm", "PatchTST_point", "PatchTST_gmm"]

    task_args = []
    for variant in VARIANTS:
        for h in HORIZONS:
            for fold_info in folds_by_h[h]:
                fid = fold_info["fold"]
                for seed in SEEDS:
                    task_args.append(
                        (variant, h, fid, seed,
                         fold_info["X_train"], fold_info["Y_train"],
                         fold_info["X_test"], fold_info["Y_test"])
                    )

    print(f"[tasks] {len(task_args)} total")
    t_start = time.time()
    if DEVICE == "cuda":
        results_flat = [train_and_save_crps(*a) for a in task_args]
    else:
        n_jobs = 4
        results_flat = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
            delayed(train_and_save_crps)(*a) for a in task_args
        )
    print(f"[done] {len(results_flat)} tasks in {time.time()-t_start:.1f}s")

    # Aggregate into (variant, horizon) -> list of crps over (fold, seed)
    crps_grid = {}  # (variant, horizon) -> list of crps
    for r in results_flat:
        key = (r["variant"], r["horizon"])
        crps_grid.setdefault(key, []).append(r["crps"])

    # Compute CRPS-Skill-Score = 1 - crps_variant / crps_baseline (per fold-seed pair)
    # Then bootstrap
    n_boot = 1000
    rng = np.random.default_rng(42)
    bootstrap_results = {}
    for variant in VARIANTS:
        bootstrap_results[variant] = {}
        for h in HORIZONS:
            crps_var = np.array(crps_grid[(variant, h)])
            crps_base = np.array(crps_grid[(BASELINE_VARIANT, h)])
            assert len(crps_var) == len(crps_base) == 15
            crps_ss = 1 - crps_var / crps_base
            # Bootstrap
            boot = np.zeros(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, len(crps_ss), size=len(crps_ss))
                boot[b] = crps_ss[idx].mean()
            bootstrap_results[variant][f"h{h}"] = {
                "mean_crps_ss": float(crps_ss.mean()),
                "std_crps_ss": float(crps_ss.std()),
                "ci_95_low": float(np.percentile(boot, 2.5)),
                "ci_95_high": float(np.percentile(boot, 97.5)),
                "ci_90_low": float(np.percentile(boot, 5.0)),
                "ci_90_high": float(np.percentile(boot, 95.0)),
                "n_obs": len(crps_ss),
                "all_crps_ss": crps_ss.tolist(),
            }

    out = {
        "config": {
            "BASELINE_VARIANT": BASELINE_VARIANT,
            "VARIANTS": VARIANTS,
            "HORIZONS": HORIZONS, "SEEDS": SEEDS, "EPOCHS": EPOCHS,
            "n_bootstrap": n_boot, "device": DEVICE,
        },
        "bootstrap_crps_ss": bootstrap_results,
    }
    out_path = RESULTS_DIR / "26_bootstrap_crps_ss.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Pretty-print headline
    print(f"\n{'Variant':<20} {'h':<4} {'CRPS-SS':<10} {'95% CI':<25} {'Signif?'}")
    print("-" * 75)
    for variant in VARIANTS:
        for h in HORIZONS:
            r = bootstrap_results[variant][f"h{h}"]
            ci = f"[{r['ci_95_low']*100:+.2f}%, {r['ci_95_high']*100:+.2f}%]"
            sig = "***" if r['ci_95_low'] > 0 else (
                  "**" if r['ci_95_low'] > -0.01 else "ns")
            print(f"{variant:<20} {h:<4} {r['mean_crps_ss']*100:+7.2f}%  {ci:<25} {sig}")


if __name__ == "__main__":
    main()