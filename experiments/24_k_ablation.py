"""
Experiment 24: GMM K-component ablation.

Justification: the review of exp 22 flagged "为什么是4分量GMM？没有解释".
This experiment runs NBEATS+GMM and TimesNet+GMM with K ∈ {2, 4, 6, 8}
on the same 5-fold walk-forward protocol as exp 19/22, and computes
the standard suite of metrics (MAE, CRPS, Pinball, Coverage). The aim
is to (1) justify K=4 as the default (lowest CRPS or simplest model
within ±0.5% MAE), and (2) check that the headline GMM-head benefit
(+1.7% to +6.4% CRPS-SS) is not specific to K=4 — i.e. it holds
across reasonable K choices.

Total tasks: 2 backbones x 4 K values x 5 folds x 4 horizons x 3 seeds
            = 480 tasks (same as exp 22).

Wall clock estimate: ~40 min on RTX 4060, ~3 hours on Mac CPU.
Push to Windows box (`13107@192.168.1.17`) before running.

Output: results/24_k_ablation.json — schema:
{
  "config": {...},
  "metrics": {
    "NBEATS_gmm_K2": {"h1": {...}, "h3": {...}, "h6": {...}, "h12": {...}},
    "NBEATS_gmm_K4": {...},
    "NBEATS_gmm_K6": {...},
    "NBEATS_gmm_K8": {...},
    "TimesNet_gmm_K2": {...},
    "TimesNet_gmm_K4": {...},
    "TimesNet_gmm_K6": {...},
    "TimesNet_gmm_K8": {...},
  }
}
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
import torch.optim as optim

# Import model classes from exp 22 (verified importable; no side effects in module body).
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp22", str(ROOT / "experiments" / "22_sota_comparison.py")
)
exp22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp22)

# Pull in CRPS / Pinball / sample helpers from src.models.wpn (not re-exported by exp22).
from src.models.wpn import crps_gmm, pinball_loss, sample_gmm

warnings.filterwarnings("ignore")

# ---------------- Config (identical to exp 22) ----------------
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

# Ablation grid
K_VALUES = [2, 4, 6, 8]
BACKBONES = ["NBEATS", "TimesNet"]  # strongest + the paper's primary


def make_model(backbone_name: str, horizon: int, k: int):
    """Build (backbone, GMMHead) with k components."""
    if backbone_name == "TimesNet":
        bb = exp22.TimesNetBackbone(seq_len=SEQ_LEN, hidden=HIDDEN, top_k=2)
    elif backbone_name == "NBEATS":
        bb = exp22.NBEATSBackbone(seq_len=SEQ_LEN, hidden=HIDDEN, n_blocks=2, theta_dim=8)
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")
    head = exp22.GMMHead(hidden=HIDDEN, horizon=horizon, n_mixtures=k)
    return bb, head


def train_one(args):
    """Train one (backbone, k, horizon, fold, seed) and return MAE + CRPS samples."""
    backbone_name, k, horizon, fold, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    bb, head = make_model(backbone_name, horizon, k)
    bb = bb.to(DEVICE)
    head = head.to(DEVICE)
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    params = list(bb.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        bb.train(); head.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            h = bb(X[idx])
            mu, sigma, pi = head(h)
            loss = exp22.gmm_nll(Y[idx], mu, sigma, pi)
            loss.backward()
            opt.step()
        sched.step()

    bb.eval(); head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Yt = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h = bb(Xt)
        mu, sigma, pi = head(h)
    # MAE: |mixture_mean - truth|
    y_pred = exp22.gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, H)
    truth = Y_test  # (n_test, H)
    mae = float(np.mean(np.abs(y_pred - truth)))
    # CRPS: empirical via GMM samples
    try:
        crps = float(crps_gmm(Yt, mu, sigma, pi, n_samples=500))
    except Exception:
        crps = float("nan")
    elapsed = time.time() - t0
    return {
        "backbone": backbone_name, "k": k, "horizon": horizon,
        "fold": fold, "seed": seed, "mae": mae, "crps": crps,
        "elapsed_sec": elapsed,
    }


def build_folds(series, seq_len, horizon):
    """Identical to exp 22 build_folds."""
    return exp22.build_folds(series, seq_len, horizon)


def main():
    print("=" * 70)
    print(" Experiment 24: GMM K-component ablation (K ∈ {2,4,6,8} × 2 backbones)")
    print("=" * 70)
    panel = pd.read_csv(DATA_PATH)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)}")

    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)

    # Pre-build supervised arrays per horizon
    X_by_h = {}
    Y_by_h = {}
    folds_by_h = {}
    for h in HORIZONS:
        X, Y = exp22.make_supervised(series, SEQ_LEN, h)
        X_by_h[h] = X
        Y_by_h[h] = Y
        folds_by_h[h] = build_folds(series, SEQ_LEN, h)

    n_tasks = len(BACKBONES) * len(K_VALUES) * len(HORIZONS) * sum(
        len(f) for f in folds_by_h.values()
    ) * len(SEEDS)
    print(f"[tasks] ~{n_tasks} train/eval tasks")
    print(f"[device] {DEVICE}")

    # Build all task args
    task_args = []
    for backbone_name in BACKBONES:
        for k in K_VALUES:
            for h in HORIZONS:
                for fold_info in folds_by_h[h]:
                    fid = fold_info["fold"]
                    X_train = fold_info["X_train"]; Y_train = fold_info["Y_train"]
                    X_test = fold_info["X_test"]; Y_test = fold_info["Y_test"]
                    for seed in SEEDS:
                        task_args.append(
                            (backbone_name, k, h, fid, seed,
                             X_train, Y_train, X_test, Y_test)
                        )

    print(f"[tasks] {len(task_args)} total, running with joblib loky parallelism...")
    t_start = time.time()
    # GPU: sequential (joblib workers compete for VRAM). CPU: 4 workers.
    if DEVICE == "cuda":
        results_flat = [train_one(a) for a in task_args]
    else:
        n_jobs = 4
        results_flat = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
            delayed(train_one)(a) for a in task_args
        )
    print(f"\n[done] {len(results_flat)} tasks in {time.time()-t_start:.1f}s")

    # Aggregate: mean MAE/CRPS over (fold, seed) for each (backbone, k, h)
    metrics = {}
    for r in results_flat:
        key = f"{r['backbone']}_gmm_K{r['k']}"
        metrics.setdefault(key, {})
        h_key = f"h{r['horizon']}"
        metrics[key].setdefault(h_key, {"mae": [], "crps": []})
        metrics[key][h_key]["mae"].append(r["mae"])
        metrics[key][h_key]["crps"].append(r["crps"])

    out_metrics = {}
    for key, h_dict in metrics.items():
        out_metrics[key] = {}
        for h_key, vals in h_dict.items():
            out_metrics[key][h_key] = {
                "mean_mae": float(np.mean(vals["mae"])),
                "std_mae": float(np.std(vals["mae"])),
                "mean_crps": float(np.mean(vals["crps"])),
                "std_crps": float(np.std(vals["crps"])),
                "n": len(vals["mae"]),
            }

    out = {
        "config": {
            "K_VALUES": K_VALUES, "BACKBONES": BACKBONES,
            "SEQ_LEN": SEQ_LEN, "HORIZONS": HORIZONS, "SEEDS": SEEDS,
            "EPOCHS": EPOCHS, "HIDDEN": HIDDEN, "DEVICE": DEVICE,
            "INITIAL_TRAIN_FRAC": INITIAL_TRAIN_FRAC, "TEST_FRAC": TEST_FRAC,
            "STEP_FRAC": STEP_FRAC,
        },
        "metrics": out_metrics,
    }
    out_path = RESULTS_DIR / "24_k_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Pretty-print headline table
    print(f"\n{'Variant':<24} {'h':<4} {'MAE':<9} {'CRPS':<9}")
    print("-" * 50)
    for key in sorted(out_metrics.keys()):
        for h_key in sorted(out_metrics[key].keys()):
            m = out_metrics[key][h_key]
            print(f"{key:<24} {h_key:<4} {m['mean_mae']:<9.5f} {m['mean_crps']:<9.5f}")


if __name__ == "__main__":
    main()