"""
Experiment 15: WPN ablation study — does the math actually work?

Compares TimesNet (baseline) vs 4 WPN variants on the Phase 3 univariate walk-forward:
  M0 = TimesNet (existing TimesBlock, Huber loss, point forecast)
  M1 = WPN full (multi-scale + learnable anchors + GMM)
  M2 = WPN w/o wavelet (single scale, learnable anchors + GMM)
  M3 = WPN w/o learnable anchors (multi-scale + fixed anchors + GMM)
  M4 = WPN w/ Huber (multi-scale + learnable anchors + point head)

Metrics:
  MAE (point prediction)
  CRPS (probabilistic forecast quality; GMM models + TimesNet-treated-as-Gaussian)
  Pinball @ 0.05 / 0.95 (tail-event forecasting — risk management)

Wall-clock target: <15 min on Windows CUDA 4060.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats as sp_stats

from src.models.ampd import AMPD
from src.models.wpn import (
    WaveletPeriodNet, gmm_nll, gmm_point_predict, crps_gmm,
    sample_gmm, pinball_loss,
)

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
HIDDEN = 32  # smaller than TimesNet to keep WPN parameter count comparable
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Baseline: TimesNet (for fair comparison)
# ============================================================

class TimesNet(nn.Module):
    """TimesNet baseline (matches exp 11/13 implementation)."""

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
        return self.proj(agg).unsqueeze(-1)  # (B, H, 1)


# ============================================================
# Model registry: 5 variants
# ============================================================

def build_model(variant: str, seq_len: int, horizon: int):
    """Build one of the 5 model variants."""
    if variant == "TimesNet":
        return TimesNet(seq_len=seq_len, horizon=horizon, top_k=2, hidden=64)
    elif variant == "WPN_full":
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=True)
    elif variant == "WPN_no_wavelet":
        # Single scale = no multi-resolution
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[60], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=True)
    elif variant == "WPN_no_learnable":
        # Fixed log-spaced periods (ablation: anchor gradients off)
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=False)
    elif variant == "WPN_huber":
        # Full architecture but point head (Huber loss instead of NLL)
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="point", use_learnable_anchors=True)
    else:
        raise ValueError(variant)


VARIANTS = ["TimesNet", "WPN_full", "WPN_no_wavelet", "WPN_no_learnable", "WPN_huber"]


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
# Worker: train + predict + metrics
# ============================================================

def train_and_predict(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    model = build_model(variant, SEQ_LEN, horizon).to(DEVICE)

    # Move input data to device
    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)  # higher wd for regularization
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Loss depends on variant
    use_gmm = "WPN" in variant and variant != "WPN_huber"

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
            if use_gmm:
                loss = crit(out, Y[idx])
            else:
                loss = crit(out, Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    # Predict
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        out = model(Xt)

    # Compute outputs
    if use_gmm:
        mu, sigma, pi = out
        y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, horizon)
        # CRPS / Pinball — convert Y_test to torch tensor on device
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (n_test, H, 1)
        try:
            crps = float(crps_gmm(Y_test_t, mu, sigma, pi, n_samples=500))
        except Exception as e:
            crps = float("nan")
        pinball = pinball_loss(Y_test_t, mu, sigma, pi,
                               quantiles=[0.05, 0.5, 0.95], n_samples=500)
    else:
        # TimesNet or WPN_huber: point forecast
        y_pred = out.cpu().numpy().squeeze(-1)  # (n_test, horizon)
        # Approximate CRPS / Pinball using training residual std as σ
        train_resid_std = float(np.std(Y_train - 0))  # Y is mean-zero
        # shapes: y_pred (n_test, H), Y_test (n_test, H)
        sigma_approx = np.full((y_pred.shape[0], y_pred.shape[1], 1), train_resid_std)
        mu_approx = y_pred[..., None]  # (n_test, H, 1)
        pi_approx = np.ones((y_pred.shape[0], y_pred.shape[1], 1))
        # Convert to torch for crps_gmm
        mu_t = torch.tensor(mu_approx, dtype=torch.float32, device=DEVICE)
        sigma_t = torch.tensor(sigma_approx, dtype=torch.float32, device=DEVICE)
        pi_t = torch.tensor(pi_approx, dtype=torch.float32, device=DEVICE)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (n_test, H, 1)
        try:
            crps = float(crps_gmm(Y_test_t, mu_t, sigma_t, pi_t, n_samples=500))
        except Exception:
            crps = float("nan")
        # Pinball (single Gaussian) — squeeze q_hat to match Y_test shape
        from scipy.stats import norm
        quantiles = [0.05, 0.5, 0.95]
        pinball = {}
        for q in quantiles:
            q_hat = (mu_approx + sigma_approx * norm.ppf(q)).squeeze(-1)  # (n_test, H)
            diff = Y_test - q_hat
            loss = np.maximum(q * diff, (q - 1) * diff).mean()
            pinball[f"pinball_{q:.2f}"] = float(loss)

    elapsed = time.time() - t0
    return {
        "variant": variant,
        "horizon": horizon,
        "fold": fold_idx,
        "seed": seed,
        "preds": y_pred.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "crps": crps,
        "pinball": pinball,
        "train_seconds": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 15: WPN ablation study (Phase 3 walk-forward)")
    print("=" * 70)
    print(f"[device workers] {DEVICE}  [cpus] {os.cpu_count()}")

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    n = len(series)
    print(f"[data] n={n}, hidden={HIDDEN}")

    folds_by_h = {}
    for h in HORIZONS:
        folds_by_h[h] = build_folds(series, SEQ_LEN, h)
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    # Tasks: 5 variants × 4 horizons × 5 folds × 3 seeds = 300 deep tasks
    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for variant in VARIANTS:
                for seed in SEEDS:
                    tasks.append((variant, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))
    print(f"[tasks] {len(tasks)} total")

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_predict)(t) for t in tasks
    )
    print(f"[parallel] all tasks done in {time.time() - t_start:.1f}s")

    # Aggregate: average predictions across seeds per (variant, h, fold)
    by_vhf: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for r in results_list:
        key = (r["variant"], r["horizon"], r["fold"])
        if key not in by_vhf:
            by_vhf[key] = {
                "preds_sum": np.zeros_like(r["preds"], dtype=np.float64),
                "truth": r["truth"],
                "crps_sum": 0.0, "crps_n": 0,
                "pinball_sum": {k: 0.0 for k in r["pinball"]},
                "pinball_n": 0,
                "n_seeds": 0,
            }
        by_vhf[key]["preds_sum"] += r["preds"]
        if not np.isnan(r["crps"]):
            by_vhf[key]["crps_sum"] += r["crps"]
            by_vhf[key]["crps_n"] += 1
        for k, v in r["pinball"].items():
            by_vhf[key]["pinball_sum"][k] += v
        by_vhf[key]["pinball_n"] += 1
        by_vhf[key]["n_seeds"] += 1

    # Compute mean metrics per (variant, h, fold)
    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (variant, h, fold), data in by_vhf.items():
        preds_mean = data["preds_sum"] / data["n_seeds"]
        mae = float(mean_absolute_error(data["truth"], preds_mean))
        crps_mean = data["crps_sum"] / data["crps_n"] if data["crps_n"] > 0 else float("nan")
        pinball_mean = {k: v / data["pinball_n"] for k, v in data["pinball_sum"].items()}

        aggregate.setdefault(f"h{h}", {}).setdefault(variant, {"mae_sum": 0.0, "crps_sum": 0.0,
                                                                "pinball_sum": {k: 0.0 for k in pinball_mean},
                                                                "folds": 0})
        aggregate[f"h{h}"][variant]["mae_sum"] += mae
        aggregate[f"h{h}"][variant]["crps_sum"] += crps_mean
        for k, v in pinball_mean.items():
            aggregate[f"h{h}"][variant]["pinball_sum"][k] += v
        aggregate[f"h{h}"][variant]["folds"] += 1

    # Average across folds
    final: Dict[str, Dict[str, Dict[str, float]]] = {}
    for hkey, variants in aggregate.items():
        final[hkey] = {}
        for variant, agg in variants.items():
            n_f = agg["folds"]
            final[hkey][variant] = {
                "mae": agg["mae_sum"] / n_f,
                "crps": agg["crps_sum"] / n_f,
                "pinball": {k: v / n_f for k, v in agg["pinball_sum"].items()},
                "n_folds": n_f,
            }

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "initial_train_frac": INITIAL_TRAIN_FRAC, "test_frac": TEST_FRAC,
            "step_frac": STEP_FRAC, "epochs": EPOCHS, "lr": LR,
            "hidden_wpn": HIDDEN, "device_workers": DEVICE, "n_jobs": n_jobs,
            "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start, "n_series": int(n),
            "variants": VARIANTS,
        },
        "aggregate": final,
    }
    out_path = RESULTS_DIR / "15_wpn_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(" MAE comparison (mean over folds) — lower is better")
    print("=" * 70)
    header = f"{'Horizon':<8}"
    for v in VARIANTS:
        header += f" {v[:14]:>14}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for v in VARIANTS:
            mae = final[f"h{h}"][v]["mae"]
            row += f" {mae:>14.5f}"
        print(row)

    print("\n" + "=" * 70)
    print(" CRPS comparison (mean over folds) — lower is better")
    print("=" * 70)
    header = f"{'Horizon':<8}"
    for v in VARIANTS:
        header += f" {v[:14]:>14}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for v in VARIANTS:
            crps = final[f"h{h}"][v]["crps"]
            row += f" {crps:>14.5f}"
        print(row)

    print("\n" + "=" * 70)
    print(" Pinball loss @ 0.05 (left-tail) — lower is better")
    print("=" * 70)
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for v in VARIANTS:
            p = final[f"h{h}"][v]["pinball"].get("pinball_0.05", float("nan"))
            row += f" {p:>14.5f}"
        print(row)

    # Markdown summary
    md = ["# WPN ablation — Phase 3 walk-forward", ""]
    cfg = out["config"]
    md.append(f"**Variants**: {', '.join(VARIANTS)}")
    md.append("")
    md.append(f"**Setup**: anchored expanding-window walk-forward, 5 folds, SEQ_LEN=60, "
              f"{len(cfg['seeds'])} seeds, {cfg['epochs']} epochs, lr={cfg['lr']}, "
              f"WPN hidden={cfg['hidden_wpn']}, device={cfg['device_workers']}, n_jobs={cfg['n_jobs']}.")
    md.append("")
    md.append(f"**Wall clock**: {cfg['wall_clock_s']:.1f}s  ({cfg['wall_clock_s']/60:.1f} min)")
    md.append("")
    md.append("## MAE (point prediction)")
    md.append("")
    md.append("| Horizon | " + " | ".join(VARIANTS) + " |")
    md.append("|---" * (len(VARIANTS) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for v in VARIANTS:
            cells.append(f"{final[f'h{h}'][v]['mae']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## CRPS (probabilistic forecast quality)")
    md.append("")
    md.append("Lower = better predictive distribution. GMM models use mixture; TimesNet uses Gaussian(μ, σ=resid_std).")
    md.append("")
    md.append("| Horizon | " + " | ".join(VARIANTS) + " |")
    md.append("|---" * (len(VARIANTS) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for v in VARIANTS:
            cells.append(f"{final[f'h{h}'][v]['crps']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Pinball @ 0.05 (left-tail)")
    md.append("")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for v in VARIANTS:
            cells.append(f"{final[f'h{h}'][v]['pinball'].get('pinball_0.05', float('nan')):.5f}")
        md.append("| " + " | ".join([f"h={h}"] + cells[1:]) + " |")
    md.append("")
    md.append("## Ablation interpretation")
    md.append("")
    md.append("- **WPN_full vs TimesNet**: Does the math beat the baseline?")
    md.append("- **WPN_no_wavelet vs WPN_full**: Is multi-scale decomposition worth it?")
    md.append("- **WPN_no_learnable vs WPN_full**: Do learnable period anchors help?")
    md.append("- **WPN_huber vs WPN_full**: Does the GMM head + NLL beat Huber loss for point prediction?")
    md.append("- **WPN_huber vs TimesNet**: Is the WPN architecture alone (without GMM) better than TimesNet?")
    md_path = RESULTS_DIR / "15_wpn_ablation_summary.md"
    md_path.write_text("\n".join(md))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
