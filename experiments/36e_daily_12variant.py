"""
12-variant experiment on daily data (Phase 1.1).

Mirrors exp 22 + exp 34, but for daily frequency:
  - 60-day lookback
  - Horizons: {1, 5, 10, 20} days (≈ 1day, 1wk, 2wk, 1mo, 3mo)
  - 5 anchored walk-forward folds
  - 3 seeds × 12 variants × 5 folds × 4 horizons = 720 training runs

Reads from data/raw/panel_daily.csv, with a specific column
(configurable via PANEL_COL).

Saves to results/36e_daily_<column>.json with same schema as
exp 22 + 34 (metrics, dm_test, per-fold CRPS arrays).
"""
import sys
import os
import json
import time
import warnings
import argparse
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
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

warnings.filterwarnings("ignore")

# ============= Config (mirrors exp 22) =============
DATA_PATH = ROOT / "data" / "raw" / "panel_daily.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
HORIZONS = [1, 5, 10, 20]
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


# ============= Backbones (replicated from exp 22) =============
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
        amp = AMPD(top_k=self.top_k, max_period=min(60, self.seq_len), min_period=2)
        self.periods = [max(int(round(p)), 2) for p in amp.fit_discover(x_train_np)]

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
    def __init__(self, seq_len, hidden=HIDDEN, kernel=10):
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


# ============= Heads =============
class PointHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, h):
        return self.proj(h)


class GMMHead(nn.Module):
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


class GaussianHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(hidden, horizon * 2)

    def forward(self, h):
        B = h.shape[0]
        params = self.proj(h).view(B, self.horizon, 1, 2)
        mu = params[..., 0]
        log_sigma = params[..., 1]
        sigma = F.softplus(log_sigma) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    log_p = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    return -log_p.mean()


def sample_gaussian(mu, sigma, n_samples=500, generator=None):
    eps = torch.randn((n_samples,) + mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps


def build_model(variant, seq_len, horizon):
    backbone_name, head_type = variant.split("_")
    if backbone_name == "TimesNet":
        backbone = TimesNetBackbone(seq_len=seq_len, hidden=HIDDEN, top_k=2)
    elif backbone_name == "DLinear":
        backbone = DLinearBackbone(seq_len=seq_len, hidden=HIDDEN, kernel=10)
    elif backbone_name == "NBEATS":
        backbone = NBEATSBackbone(seq_len=seq_len, hidden=HIDDEN, n_blocks=2, theta_dim=8)
    elif backbone_name == "iTransformer":
        backbone = ITransformerBackboneAdapter(seq_len=seq_len, hidden=HIDDEN,
                                               n_heads=4, n_layers=3, ff_dim=128)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if head_type == "point":
        head = PointHead(hidden=HIDDEN, horizon=horizon)
    elif head_type == "gauss":
        head = GaussianHead(hidden=HIDDEN, horizon=horizon)
    elif head_type == "gmm":
        head = GMMHead(hidden=HIDDEN, horizon=horizon, n_mixtures=N_MIXTURES)
    else:
        raise ValueError(f"Unknown head: {head_type}")
    return backbone, head, head_type


# ============= Data =============
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


# ============= Worker =============
def train_and_save_predictions(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    backbone, head, head_type = build_model(variant, SEQ_LEN, horizon)
    backbone = backbone.to(DEVICE)
    head = head.to(DEVICE)

    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    if head_type == "point":
        crit = nn.HuberLoss(delta=1.0)
    elif head_type == "gauss":
        crit = lambda yp, yt: gaussian_nll(yt, yp[0], yp[1])
    else:  # gmm
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
                yp = out.unsqueeze(-1)
                loss = crit(yp, Y[idx])
            elif head_type == "gauss":
                mu, sigma = out
                loss = crit((mu, sigma), Y[idx])
            else:
                mu, sigma, pi = out
                loss = crit((mu, sigma, pi), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if head_type == "point":
            y_pred = out.cpu().numpy()
            h_train = backbone(X)
            train_pred = head(h_train).detach().cpu().numpy()
            train_resid = Y_train - train_pred
            sigma_per_step = np.std(train_resid, axis=0)
            N_SAMPLES = 500
            np.random.seed(seed)
            samples = np.random.normal(
                y_pred[None, :, :], sigma_per_step[None, None, :],
                size=(N_SAMPLES, y_pred.shape[0], y_pred.shape[1]),
            )
        elif head_type == "gauss":
            mu, sigma = out
            y_pred = mu.cpu().numpy().squeeze(-1)
            g = torch.Generator(device=DEVICE)
            g.manual_seed(seed)
            samples_t = sample_gaussian(mu, sigma, n_samples=500, generator=g)
            samples = samples_t.squeeze(-1).cpu().numpy()
        else:  # gmm
            mu, sigma, pi = out
            y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)
            N_SAMPLES = 500
            samples_t = sample_gmm(mu, sigma, pi, n_samples=N_SAMPLES)
            samples = samples_t.squeeze(-1).cpu().numpy()

    return {
        "variant": variant, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": y_pred.astype(np.float32),
        "samples": samples.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "head_type": head_type,
        "train_seconds": time.time() - t0,
    }


# ============= Metrics =============
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


# ============= Main =============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--column", required=True, help="Panel column to use as target")
    parser.add_argument("--horizons", default=None, help="Override horizons (e.g. '1,5,10,20')")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override n_jobs (default=min(cpu_count, 8))")
    parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS (default=80)")
    args = parser.parse_args()

    if args.horizons:
        global HORIZONS
        HORIZONS = [int(h) for h in args.horizons.split(",")]
    if args.epochs:
        global EPOCHS
        EPOCHS = args.epochs

    print("=" * 70)
    print(f" Experiment 36e: Daily 12-variant, column={args.column}, horizons={HORIZONS}")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    if args.column not in panel.columns:
        print(f"ERROR: column {args.column!r} not in panel. Available: {panel.columns.tolist()}")
        return
    series = panel[args.column].dropna().values.astype(np.float32)
    print(f"[data] column={args.column}, n={len(series)}")
    if len(series) < 2000:
        print(f"WARNING: only {len(series)} obs; walk-forward may have < 5 folds")

    folds_by_h = {h: build_folds(series, SEQ_LEN, h) for h in HORIZONS}
    for h in HORIZONS:
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
    HEADS = ["point", "gauss", "gmm"]
    VARIANTS = [f"{b}_{he}" for b in BACKBONES for he in HEADS]
    print(f"[variants] {VARIANTS}")
    n_tasks = len(VARIANTS) * sum(len(folds_by_h[h]) for h in HORIZONS) * len(SEEDS)
    print(f"[tasks] {n_tasks} total")

    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for variant in VARIANTS:
                for seed in SEEDS:
                    tasks.append((variant, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))

    t_start = time.time()
    n_jobs = args.n_jobs if args.n_jobs else min(os.cpu_count() or 1, 8)
    print(f"[parallel] n_jobs={n_jobs}, device={DEVICE}, epochs={EPOCHS}")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_save_predictions)(t) for t in tasks
    )
    print(f"[parallel] all done in {time.time() - t_start:.1f}s")

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

    metrics = {f"h{h}": {} for h in HORIZONS}
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
                coverage = {}
                for alpha in WINKLER_LEVELS:
                    q_lo = (1 - alpha) / 2
                    q_hi = (1 + alpha) / 2
                    lower = np.quantile(samples, q_lo, axis=0)
                    upper = np.quantile(samples, q_hi, axis=0)
                    coverage[f"coverage_{alpha:.2f}"] = coverage_at_level(truth, lower, upper)
                fold_metrics.append({
                    "fold": fi, "mae": mae, "crps": crps,
                    "crps_per_fold": crps, "mae_per_fold": mae,
                    "pinball": pinball, "coverage": coverage,
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
                "coverage": {},
                "n_folds": n_f,
            }
            for q in QUANTILES:
                k = f"pinball_{q:.2f}"
                agg["pinball"][k] = np.mean([m["pinball"][k] for m in fold_metrics])
            for lev in WINKLER_LEVELS:
                k = f"coverage_{lev:.2f}"
                agg["coverage"][k] = np.mean([m["coverage"][k] for m in fold_metrics])
            metrics[f"h{h}"][variant] = agg

    out = {
        "config": {
            "column": args.column,
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "epochs": EPOCHS, "lr": LR, "device": DEVICE,
            "n_total_tasks": n_tasks,
            "wall_clock_s": time.time() - t_start,
            "variants": VARIANTS,
        },
        "metrics": metrics,
    }
    out_path = RESULTS_DIR / f"36e_daily_{args.column}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 70)
    print(f" Summary ({args.column})")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n--- h={h} ---")
        print(f"  {'Variant':<20} {'MAE':>10} {'CRPS':>10} {'Cov@0.90':>10}")
        for variant in VARIANTS:
            v = metrics[f"h{h}"].get(variant, {})
            if not v:
                continue
            mae = v.get("mae", float("nan"))
            crps = v.get("crps", float("nan"))
            cov = v.get("coverage", {}).get("coverage_0.90", float("nan"))
            print(f"  {variant:<20} {mae:>10.5f} {crps:>10.5f} {cov:>10.4f}")


if __name__ == "__main__":
    main()
