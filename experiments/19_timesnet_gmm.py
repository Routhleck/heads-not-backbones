"""
Experiment 19: TimesNet + GMM head (minimal modification).

Hypothesis: replacing TimesNet's linear projection + Huber loss with a Gaussian mixture
density head + NLL training should give similar tail-event forecasting improvement as
WPN (which has the same GMM head), since the ablation showed GMM is the dominant contributor.

This isolates the contribution: is the gain from the GMM head, or from the multi-scale +
PeriodBank architecture? If TimesNet+GMM matches WPN, then the architectural complexity
of WPN adds little.

Setup: same as exp 15 — 5 anchored walk-forward folds, 4 horizons, 3 seeds, 8 models:
  - TimesNet (Huber, point forecast)  [re-run for fair comparison with same hyperparameters]
  - TimesNet_GMM (Huber→NLL, linear→GMM head)  [the proposed minimal modification]
  - WPN_full (existing baseline from exp 15)  [for comparison]
  - WPN_huber (Huber instead of NLL — confirms GMM contribution)
  - WPN_no_wavelet, WPN_no_learnable  [context]

Models trained on S&P 500 monthly log-returns.
Metrics: MAE, Pinball @ 0.05 / 0.5 / 0.95.
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
from src.models.wpn import (
    WaveletPeriodNet, gmm_nll, gmm_point_predict, crps_gmm, pinball_loss,
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
HIDDEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# TimesNet + GMM head (the minimal modification)
# ============================================================

class TimesNetGMMHead(nn.Module):
    """TimesNet backbone with GMM density head (replaces linear projection + Huber loss).

    Differences from exp 11 TimesNet:
    - Final layer: nn.Linear(hidden, H) -> nn.Linear(hidden, H*K*3) where K = n_mixtures
    - Output: (mu, sigma, pi) per (batch, horizon, 1, K)
    - Loss: NLL under Gaussian mixture (handled in train_and_predict)

    Everything else (AMPD-driven 2D variation, SUM aggregation) is unchanged.
    """

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
        # GMM head: 3 outputs per mixture per (timestep, channel)
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

        # GMM head
        params = self.gmm_proj(agg).view(B, self.horizon, 1, self.n_mixtures, 3)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        logit_pi = params[..., 2]
        sigma = F.softplus(log_sigma) + 1e-3
        pi = F.softmax(logit_pi, dim=-1)
        return mu, sigma, pi


def build_model(variant, seq_len, horizon):
    if variant == "TimesNet":
        return None  # Special-cased: Huber loss + linear projection
    elif variant == "TimesNet_GMM":
        return TimesNetGMMHead(seq_len=seq_len, horizon=horizon, n_mixtures=4, top_k=2, hidden=64)
    elif variant == "WPN_full":
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=True)
    elif variant == "WPN_no_wavelet":
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[60], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=True)
    elif variant == "WPN_no_learnable":
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="gmm", use_learnable_anchors=False)
    elif variant == "WPN_huber":
        return WaveletPeriodNet(seq_len=seq_len, horizon=horizon, hidden=HIDDEN,
                                scales=[3, 6, 12, 24], n_anchors=4, n_mixtures=4,
                                head="point", use_learnable_anchors=True)
    else:
        raise ValueError(variant)


class TimesNetBaseline(nn.Module):
    """Original TimesNet for fair comparison (Huber loss, point forecast)."""

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

def train_and_predict(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    use_gmm = variant in ("TimesNet_GMM", "WPN_full", "WPN_no_wavelet", "WPN_no_learnable")
    use_huber_only = variant in ("TimesNet", "WPN_huber")

    # Build model
    if variant == "TimesNet":
        model = TimesNetBaseline(seq_len=SEQ_LEN, horizon=horizon, top_k=2, hidden=64).to(DEVICE)
    else:
        model = build_model(variant, SEQ_LEN, horizon).to(DEVICE)

    # Move data
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

    # Predict
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        out = model(Xt)

    # Compute metrics
    if use_gmm:
        mu, sigma, pi = out
        y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        try:
            crps = float(crps_gmm(Y_test_t, mu, sigma, pi, n_samples=500))
        except Exception:
            crps = float("nan")
        pinball = pinball_loss(Y_test_t, mu, sigma, pi,
                               quantiles=[0.05, 0.5, 0.95], n_samples=500)
    else:
        y_pred = out.cpu().numpy().squeeze(-1)
        train_resid_std = float(np.std(Y_train - 0))
        sigma_approx = np.full((y_pred.shape[0], y_pred.shape[1], 1), train_resid_std)
        mu_approx = y_pred[..., None]
        pi_approx = np.ones((y_pred.shape[0], y_pred.shape[1], 1))
        mu_t = torch.tensor(mu_approx, dtype=torch.float32, device=DEVICE)
        sigma_t = torch.tensor(sigma_approx, dtype=torch.float32, device=DEVICE)
        pi_t = torch.tensor(pi_approx, dtype=torch.float32, device=DEVICE)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        try:
            crps = float(crps_gmm(Y_test_t, mu_t, sigma_t, pi_t, n_samples=500))
        except Exception:
            crps = float("nan")
        from scipy.stats import norm
        quantiles = [0.05, 0.5, 0.95]
        pinball = {}
        for q in quantiles:
            q_hat = (mu_approx + sigma_approx * norm.ppf(q)).squeeze(-1)
            diff = Y_test - q_hat
            loss = np.maximum(q * diff, (q - 1) * diff).mean()
            pinball[f"pinball_{q:.2f}"] = float(loss)

    return {
        "variant": variant, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": y_pred.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "crps": crps, "pinball": pinball,
        "train_seconds": time.time() - t0,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 19: TimesNet + GMM head (minimal modification)")
    print("=" * 70)
    print(f"[device] {DEVICE}, hidden={HIDDEN}")

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    print(f"[data] n={len(series)}")

    folds_by_h = {}
    for h in HORIZONS:
        folds_by_h[h] = build_folds(series, SEQ_LEN, h)
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    VARIANTS = ["TimesNet", "TimesNet_GMM", "WPN_full", "WPN_no_wavelet",
                "WPN_no_learnable", "WPN_huber"]

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
    print(f"[parallel] n_jobs={n_jobs}")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_predict)(t) for t in tasks
    )
    print(f"[parallel] all tasks done in {time.time() - t_start:.1f}s")

    # Aggregate
    by_vhf = {}
    for r in results_list:
        key = (r["variant"], r["horizon"], r["fold"])
        if key not in by_vhf:
            by_vhf[key] = {
                "preds_sum": np.zeros_like(r["preds"], dtype=np.float64),
                "truth": r["truth"], "crps_sum": 0.0, "crps_n": 0,
                "pinball_sum": {k: 0.0 for k in r["pinball"]}, "pinball_n": 0,
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

    final = {}
    for (variant, h, fold), data in by_vhf.items():
        preds_mean = data["preds_sum"] / data["n_seeds"]
        mae = float(mean_absolute_error(data["truth"], preds_mean))
        crps_mean = data["crps_sum"] / data["crps_n"] if data["crps_n"] > 0 else float("nan")
        pinball_mean = {k: v / data["pinball_n"] for k, v in data["pinball_sum"].items()}
        final.setdefault(f"h{h}", {}).setdefault(variant, {"mae_sum": 0.0, "crps_sum": 0.0,
                                                              "pinball_sum": {k: 0.0 for k in pinball_mean},
                                                              "folds": 0})
        final[f"h{h}"][variant]["mae_sum"] += mae
        final[f"h{h}"][variant]["crps_sum"] += crps_mean
        for k, v in pinball_mean.items():
            final[f"h{h}"][variant]["pinball_sum"][k] += v
        final[f"h{h}"][variant]["folds"] += 1

    aggregate = {}
    for hkey, variants in final.items():
        aggregate[hkey] = {}
        for variant, agg in variants.items():
            n_f = agg["folds"]
            aggregate[hkey][variant] = {
                "mae": agg["mae_sum"] / n_f,
                "crps": agg["crps_sum"] / n_f,
                "pinball": {k: v / n_f for k, v in agg["pinball_sum"].items()},
                "n_folds": n_f,
            }

    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "epochs": EPOCHS, "lr": LR, "hidden": HIDDEN, "device": DEVICE,
            "n_jobs": n_jobs, "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start, "variants": VARIANTS,
        },
        "aggregate": aggregate,
    }
    out_path = RESULTS_DIR / "19_timesnet_gmm_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

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
            row += f" {aggregate[f'h{h}'][v]['mae']:>14.5f}"
        print(row)

    print("\n" + "=" * 70)
    print(" Pinball @ 0.05 (left-tail, lower is better)")
    print("=" * 70)
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for v in VARIANTS:
            row += f" {aggregate[f'h{h}'][v]['pinball']['pinball_0.05']:>14.5f}"
        print(row)

    # Key comparison: TimesNet vs TimesNet_GMM
    print("\n" + "=" * 70)
    print(" KEY COMPARISON: TimesNet vs TimesNet_GMM (the proposed minimal modification)")
    print("=" * 70)
    print(f"{'Horizon':<8} {'Metric':<14} {'TimesNet':>12} {'TimesNet_GMM':>14} {'Change':>10}")
    print("-" * 70)
    for h in HORIZONS:
        for metric_name, metric_key in [("MAE", "mae"), ("Pinball 0.05", "pinball_0.05"),
                                          ("Pinball 0.5", "pinball_0.50"), ("Pinball 0.95", "pinball_0.95")]:
            if metric_key == "mae":
                t_val = aggregate[f"h{h}"]["TimesNet"]["mae"]
                tg_val = aggregate[f"h{h}"]["TimesNet_GMM"]["mae"]
            else:
                t_val = aggregate[f"h{h}"]["TimesNet"]["pinball"][metric_key]
                tg_val = aggregate[f"h{h}"]["TimesNet_GMM"]["pinball"][metric_key]
            change = (tg_val - t_val) / t_val * 100 if t_val > 0 else 0
            change_str = f"{change:+.2f}%"
            if h == HORIZONS[0]:
                print(f"  h={h:<4} {metric_name:<14} {t_val:>12.5f} {tg_val:>14.5f} {change_str:>10}")
            else:
                print(f"  h={h:<4} {metric_name:<14} {t_val:>12.5f} {tg_val:>14.5f} {change_str:>10}")

    # MD summary
    md = ["# TimesNet + GMM Head Ablation", ""]
    md.append(f"**Setup**: same as exp 15 — anchored expanding-window walk-forward, 5 folds, "
              f"{len(SEEDS)} seeds, device={DEVICE}, hidden={HIDDEN}.")
    md.append("")
    md.append(f"**Wall clock**: {time.time() - t_start:.1f}s")
    md.append("")
    md.append("## MAE (mean over folds)")
    md.append("")
    md.append("| Horizon | " + " | ".join(VARIANTS) + " |")
    md.append("|---" * (len(VARIANTS) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for v in VARIANTS:
            cells.append(f"{aggregate[f'h{h}'][v]['mae']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Pinball @ 0.05 (left-tail)")
    md.append("")
    md.append("| Horizon | " + " | ".join(VARIANTS) + " |")
    md.append("|---" * (len(VARIANTS) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for v in VARIANTS:
            cells.append(f"{aggregate[f'h{h}'][v]['pinball']['pinball_0.05']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md_path = RESULTS_DIR / "19_timesnet_gmm_summary.md"
    md_path.write_text("\n".join(md))
    print(f"\nSaved {md_path}")


if __name__ == "__main__":
    main()
