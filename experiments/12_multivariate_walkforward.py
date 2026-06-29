"""
Experiment 12: Multivariate walk-forward + DM test (Phase 4).

5 channels: SP500_Return, Copper_Return, Oil_WTI_Return, Wheat_Return, M2_Growth.
Overlap: 1992-02 to 2023-09 (380 monthly obs, no NaN).

Setup:
- Anchored expanding-window walk-forward, 7 folds (~2.5-year test chunks each)
- Initial train: first 130 months (~10 years), test on next 30 months, slide by 30 months
- 3 seeds for deep models, averaged per fold
- Target = SP500_Return (channel 0); multivariate models predict all 5 channels
- PAB(1ch) is the control: same model, only SP500_Return input

Models: Linear, DLinear, iTransformer, TimesNet, PAB(5ch), PAB(1ch)

Parallelism:
- Same joblib loky n_jobs=8 scaffold as exp 11
- CPU workers (MPS serializes across processes)

Output:
- results/12_multivariate_walkforward.json: per-fold metrics + DM stats
- results/12_multivariate_summary.md: human-readable summary
- figures/12_multivariate_per_fold.png: per-fold MAE line plot
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as sp_stats

from src.models.ampd import AMPD
from src.models.period_attention import PeriodAttentionTimesNetLite

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]

CHANNELS = ["SP500_Return", "Copper_Return", "Oil_WTI_Return", "Wheat_Return", "M2_Growth"]
TARGET_COL = "SP500_Return"
TARGET_IDX = CHANNELS.index(TARGET_COL)
N_CHANNELS = len(CHANNELS)

# Walk-forward (hardcoded for clean fold boundaries given short series)
INITIAL_TRAIN = 130
TEST_WINDOW = 30
STEP = 30

# Training
EPOCHS = 60
LR = 1e-3
BS = 128
HIDDEN = 64

DEVICE = "cpu"  # workers run on CPU to avoid MPS contention


# ============================================================
# Multivariate baseline models
# ============================================================

class DLinearMV(nn.Module):
    """Multivariate DLinear: per-channel moving-average decomposition + linear."""

    def __init__(self, seq_len, horizon, n_channels, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # x: (B, T, C)
        x_t = x.transpose(1, 2)  # (B, C, T)
        pad = (self.kernel_size - 1) // 2
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, self.kernel_size, stride=1)  # (B, C, T)
        seasonal = x_t - trend
        y = self.linear_trend(trend) + self.linear_seasonal(seasonal)  # (B, C, horizon)
        return y.transpose(1, 2)  # (B, horizon, C)


class iTransformerMV(nn.Module):
    """iTransformer: variates as tokens, attention across variates."""

    def __init__(self, seq_len, horizon, n_channels, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.proj_in = nn.Linear(seq_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_channels, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                                batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.proj_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        h = x.transpose(1, 2)  # (B, C, T)
        h = self.proj_in(h)  # (B, C, d_model)
        h = h + self.pos_emb
        h = self.encoder(h)
        y = self.proj_out(h)  # (B, C, horizon)
        return y.transpose(1, 2)  # (B, horizon, C)


class TimesNetMV(nn.Module):
    """TimesNet: 2D-variation block (FFT top-k periods + 2D conv + flatten + linear)."""

    def __init__(self, seq_len, horizon, n_channels, top_k=2, hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.top_k = top_k
        self.hidden = hidden
        self.periods = []
        self.conv2d = nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.proj = nn.Linear(hidden, horizon * n_channels)

    def fit_periods(self, x_train_np):
        # x_train_np: (n_train, T) — use first channel for AMPD
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
        y = self.proj(agg)
        return y.reshape(B, self.horizon, C)


# Model registry: name -> (callable, n_channels)
# Linear uses sklearn (no torch model)
MODELS = {
    "Linear": ("linear", None),
    "DLinear": (DLinearMV, None),  # n_channels from data
    "iTransformer": (iTransformerMV, None),
    "TimesNet": (TimesNetMV, None),
    "PAB(5ch)": (PeriodAttentionTimesNetLite, 5),
    "PAB(1ch)": (PeriodAttentionTimesNetLite, 1),
}


# ============================================================
# Data: multivariate walk-forward folds
# ============================================================

def make_supervised_multivariate(panel_df: pd.DataFrame, channels: List[str],
                                 seq_len: int, horizon: int, target_idx: int):
    """Returns X (n_pairs, seq_len, n_channels), Y (n_pairs, horizon, n_channels)."""
    arr = panel_df[channels].values.astype(np.float32)  # (T, C)
    T, C = arr.shape
    X, Y = [], []
    for t in range(T - seq_len - horizon + 1):
        X.append(arr[t:t + seq_len])
        Y.append(arr[t + seq_len:t + seq_len + horizon])
    X = np.stack(X, axis=0)  # (n_pairs, seq_len, C)
    Y = np.stack(Y, axis=0)  # (n_pairs, horizon, C)
    return X, Y


def build_folds_multivariate(panel_df: pd.DataFrame, channels: List[str],
                              seq_len: int, horizon: int, target_idx: int):
    """Anchored expanding-window walk-forward for multivariate series."""
    T = len(panel_df)
    X_all, Y_all = make_supervised_multivariate(panel_df, channels, seq_len, horizon, target_idx)

    folds = []
    fold_idx = 0
    train_end_series = INITIAL_TRAIN
    while train_end_series + TEST_WINDOW + horizon <= T:
        test_end_series = train_end_series + TEST_WINDOW
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
            "train_end_series": train_end_series,
            "test_start_series": train_end_series,
            "test_end_series": test_end_series,
            "n_train_pairs": len(X_train),
            "n_test_pairs": len(X_test),
            "X_train": X_train, "Y_train": Y_train,
            "X_test": X_test, "Y_test": Y_test,
        })
        fold_idx += 1
        train_end_series += STEP
    return folds


# ============================================================
# Worker: train + predict one (model, horizon, fold, seed) tuple
# ============================================================

def train_and_predict(args):
    """Returns dict with model, horizon, fold, seed, preds, truth, train_seconds."""
    model_name, model_spec, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test, n_channels = args

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    target_slice = (slice(None), slice(None), TARGET_IDX)  # select channel 0

    if model_spec == "linear":
        # Multivariate linear regression per (horizon_step, channel) — sklearn
        n_train = X_train.reshape(len(X_train), -1)  # flatten channels
        n_test = X_test.reshape(len(X_test), -1)
        preds_per_step = []  # list of (n_test, C)
        for k in range(Y_train.shape[1]):
            lr = LinearRegression()
            Yk = Y_train[:, k, :]  # (n_train, C)
            lr.fit(n_train, Yk)
            preds_per_step.append(lr.predict(n_test))  # (n_test, C)
        yp = np.stack(preds_per_step, axis=1)  # (n_test, horizon, C)
    else:
        # Deep model
        model_cls = model_spec
        if model_cls is PeriodAttentionTimesNetLite:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels,
                          top_k=2, hidden=HIDDEN, num_heads=4, dropout=0.1)
        elif model_cls is DLinearMV:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels)
        elif model_cls is iTransformerMV:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels, d_model=HIDDEN)
        elif model_cls is TimesNetMV:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=n_channels, top_k=2, hidden=HIDDEN)
        else:
            kwargs = {}

        model = model_cls(**kwargs).to(DEVICE)

        # Periodic models need period fit on channel 0 (SP500_Return)
        if model_cls is PeriodAttentionTimesNetLite:
            model.fit_periods(X_train[:, :, 0], max_period=60, min_period=4)
            model = model.to(DEVICE)
        elif model_cls is TimesNetMV:
            model.fit_periods(X_train[:, :, 0])
            model.proj = model.proj.to(DEVICE)

        X = torch.tensor(X_train, dtype=torch.float32)
        Y = torch.tensor(Y_train, dtype=torch.float32)
        n = X.shape[0]
        bs = min(BS, n)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.HuberLoss(delta=1.0)
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                yp_batch = model(X[idx])
                loss = crit(yp_batch, Y[idx])
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            yp = model(Xt).cpu().numpy()

    elapsed = time.time() - t0
    # Slice target channel only for eval
    return {
        "model": model_name,
        "horizon": horizon,
        "fold": fold_idx,
        "seed": seed,
        "preds": yp[:, :, TARGET_IDX].astype(np.float32),  # (n_test, horizon)
        "truth": Y_test[:, :, TARGET_IDX].astype(np.float32),
        "train_seconds": elapsed,
    }


# ============================================================
# Diebold-Mariano with Newey-West HAC (same as exp 11)
# ============================================================

def dm_test(e1: np.ndarray, e2: np.ndarray, horizon: int, loss: str = "se") -> Dict[str, float]:
    assert e1.shape == e2.shape
    d = e1 ** 2 - e2 ** 2 if loss == "se" else np.abs(e1) - np.abs(e2)
    d_centered = d - d.mean()
    n = len(d_centered)
    h_lag = max(horizon - 1, 1)
    gamma0 = np.sum(d_centered * d_centered) / n
    var_d = gamma0
    for k in range(1, h_lag + 1):
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / n
        var_d += 2.0 * (1.0 - k / (h_lag + 1)) * gamma_k
    var_d = max(var_d, 1e-12)
    mean_d = float(np.mean(e1 ** 2) - np.mean(e2 ** 2)) if loss == "se" else float(np.mean(np.abs(e1)) - np.mean(np.abs(e2)))
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(dm_stat), df=n - 1))
    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": mean_d,
        "n": int(n),
        "h_lag": int(h_lag),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 12: Multivariate walk-forward (5 channels)")
    print("=" * 70)
    print(f"[device workers] {DEVICE}  [cpus] {os.cpu_count()}")

    panel_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    # 5-channel overlap: drop rows with any NaN in channels
    panel = panel_full[CHANNELS].dropna()
    print(f"[data] {len(panel)} months, channels={CHANNELS}")
    print(f"[data] date range: {panel.index[0].date()} -> {panel.index[-1].date()}")

    # Folds per horizon
    folds_by_h: Dict[int, List[Dict[str, Any]]] = {}
    for h in HORIZONS:
        folds = build_folds_multivariate(panel, CHANNELS, SEQ_LEN, h, TARGET_IDX)
        folds_by_h[h] = folds
        print(f"[folds h={h}] n_folds={len(folds)}, "
              f"first test {panel.index[folds[0]['test_start_series']].date()} -> "
              f"{panel.index[folds[0]['test_end_series']-1].date()}, "
              f"last test {panel.index[folds[-1]['test_start_series']].date()} -> "
              f"{panel.index[folds[-1]['test_end_series']-1].date()}")

    # Tasks
    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for name, (model_spec, fixed_ch) in MODELS.items():
                # 1-channel PAB: feed only channel 0
                if fixed_ch == 1:
                    X_train = fold["X_train"][:, :, TARGET_IDX:TARGET_IDX+1]
                    X_test = fold["X_test"][:, :, TARGET_IDX:TARGET_IDX+1]
                    # Y is single-channel
                    Y_train = fold["Y_train"][:, :, TARGET_IDX:TARGET_IDX+1]
                    Y_test = fold["Y_test"][:, :, TARGET_IDX:TARGET_IDX+1]
                    n_ch = 1
                else:
                    X_train = fold["X_train"]
                    X_test = fold["X_test"]
                    Y_train = fold["Y_train"]
                    Y_test = fold["Y_test"]
                    n_ch = N_CHANNELS

                for seed in (SEEDS if model_spec != "linear" else [0]):
                    tasks.append((
                        name, model_spec, h, fold["fold"], seed,
                        X_train, Y_train, X_test, Y_test, n_ch,
                    ))
    print(f"[tasks] {len(tasks)} total  ({sum(1 for t in tasks if t[1]=='linear')} Linear + "
          f"{sum(1 for t in tasks if t[1]!='linear')} deep)")

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    print(f"[parallel] n_jobs={n_jobs}, backend=loky")
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_predict)(t) for t in tasks
    )
    print(f"[parallel] all tasks done in {time.time() - t_start:.1f}s")

    # Aggregate: predictions per (model, horizon, fold) = mean over seeds
    by_mhf: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
    seed_count: Dict[Tuple[str, int, int], int] = {}
    train_time: Dict[Tuple[str, int, int], float] = {}
    for r in results_list:
        key = (r["model"], r["horizon"], r["fold"])
        if key not in by_mhf:
            by_mhf[key] = {"preds": np.zeros_like(r["preds"], dtype=np.float64),
                           "truth": r["truth"]}
            seed_count[key] = 0
            train_time[key] = 0.0
        by_mhf[key]["preds"] += r["preds"]
        seed_count[key] += 1
        train_time[key] += r["train_seconds"]
    for key in by_mhf:
        by_mhf[key]["preds"] /= seed_count[key]

    # Per-fold metrics
    per_fold_metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for (model, h, fold), data in by_mhf.items():
        per_fold_metrics.setdefault(f"h{h}", {}).setdefault(fold, {})[model] = {
            "mae": float(mean_absolute_error(data["truth"], data["preds"])),
            "rmse": float(np.sqrt(mean_squared_error(data["truth"], data["preds"]))),
            "n_test_pairs": int(data["truth"].shape[0]),
            "avg_train_seconds": float(train_time[(model, h, fold)] / seed_count[(model, h, fold)]),
        }

    # Aggregate per (model, horizon): mean MAE/RMSE across folds
    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for hkey, folds in per_fold_metrics.items():
        aggregate[hkey] = {}
        for fold_data in folds.values():
            for model, m in fold_data.items():
                aggregate[hkey].setdefault(model, {"mae_sum": 0.0, "rmse_sum": 0.0, "folds": 0})
                aggregate[hkey][model]["mae_sum"] += m["mae"]
                aggregate[hkey][model]["rmse_sum"] += m["rmse"]
                aggregate[hkey][model]["folds"] += 1
        for model, agg in aggregate[hkey].items():
            n_f = agg["folds"]
            agg["mae_mean"] = agg["mae_sum"] / n_f
            agg["rmse_mean"] = agg["rmse_sum"] / n_f
            agg["n_folds"] = n_f
            del agg["mae_sum"], agg["rmse_sum"], agg["folds"]

    # DM tests: PAB(5ch) vs each baseline + PAB(5ch) vs PAB(1ch) + PAB(1ch) vs Linear
    baselines_to_test = ["Linear", "DLinear", "iTransformer", "TimesNet", "PAB(1ch)"]
    dm_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for h in HORIZONS:
        dm_results[f"h{h}"] = {}
        for baseline in baselines_to_test:
            pab_preds_all, base_preds_all, truth_all = [], [], []
            for fold_idx in range(len(folds_by_h[h])):
                pab_key = ("PAB(5ch)", h, fold_idx)
                base_key = (baseline, h, fold_idx)
                if pab_key not in by_mhf or base_key not in by_mhf:
                    continue
                pab_preds_all.append(by_mhf[pab_key]["preds"])
                base_preds_all.append(by_mhf[base_key]["preds"])
                truth_all.append(by_mhf[pab_key]["truth"])
            if not pab_preds_all:
                continue
            pab_p = np.concatenate(pab_preds_all, axis=0)
            base_p = np.concatenate(base_preds_all, axis=0)
            truth_c = np.concatenate(truth_all, axis=0)
            e_pab = np.mean((pab_p - truth_c) ** 2, axis=1)
            e_base = np.mean((base_p - truth_c) ** 2, axis=1)
            dm = dm_test(e_pab, e_base, horizon=h, loss="se")
            dm["mean_se_pab"] = float(np.mean(e_pab))
            dm["mean_se_base"] = float(np.mean(e_base))
            dm["se_reduction_pct"] = float(100.0 * (1.0 - dm["mean_se_pab"] / max(dm["mean_se_base"], 1e-12)))
            dm_results[f"h{h}"][baseline] = dm

    # Save
    out = {
        "config": {
            "seq_len": SEQ_LEN,
            "horizons": HORIZONS,
            "seeds": SEEDS,
            "channels": CHANNELS,
            "target_col": TARGET_COL,
            "initial_train": INITIAL_TRAIN,
            "test_window": TEST_WINDOW,
            "step": STEP,
            "epochs": EPOCHS,
            "lr": LR,
            "hidden": HIDDEN,
            "device_workers": DEVICE,
            "n_jobs": n_jobs,
            "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start,
            "n_series": int(len(panel)),
            "n_folds_per_horizon": {h: len(folds_by_h[h]) for h in HORIZONS},
        },
        "aggregate": aggregate,
        "per_fold": per_fold_metrics,
        "dm_test": dm_results,
    }
    out_path = RESULTS_DIR / "12_multivariate_walkforward.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(" Multivariate walk-forward aggregate MAE (mean across folds)")
    print("=" * 70)
    model_order = ["Linear", "DLinear", "iTransformer", "TimesNet", "PAB(1ch)", "PAB(5ch)"]
    header = f"{'Horizon':<8}"
    for name in model_order:
        header += f" {name[:11]:>11}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for name in model_order:
            mae = aggregate[f"h{h}"][name]["mae_mean"]
            row += f" {mae:>11.5f}"
        print(row)

    print("\nBest per horizon (by MAE):")
    for h in HORIZONS:
        best = min(model_order, key=lambda k: aggregate[f"h{h}"][k]["mae_mean"])
        print(f"  h={h}: {best}  MAE={aggregate[f'h{h}'][best]['mae_mean']:.5f}")

    print("\n" + "=" * 70)
    print(" Diebold-Mariano: PAB(5ch) vs baseline (negative DM = PAB(5ch) better)")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n  h={h}:")
        print(f"    {'baseline':<14} {'DM stat':>10} {'p-value':>10} {'SE PAB':>10} {'SE base':>10} {'SE red%':>8}  sig")
        for baseline, dm in dm_results[f"h{h}"].items():
            sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else ("*" if dm["p_value"] < 0.10 else ""))
            print(f"    {baseline:<14} {dm['dm_stat']:>10.3f} {dm['p_value']:>10.4f} "
                  f"{dm['mean_se_pab']:>10.6f} {dm['mean_se_base']:>10.6f} "
                  f"{dm['se_reduction_pct']:>7.2f}%  {sig}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        colors = {'Linear':'#999', 'DLinear':'#888', 'iTransformer':'#555',
                  'TimesNet':'#1f77b4', 'PAB(1ch)':'#ff7f0e', 'PAB(5ch)':'#d62728'}
        markers = {'Linear':'o', 'DLinear':'s', 'iTransformer':'D',
                  'TimesNet':'X', 'PAB(1ch)':'^', 'PAB(5ch)':'*'}
        for col, h in enumerate(HORIZONS):
            ax = axes[col]
            folds_data = per_fold_metrics[f"h{h}"]
            fold_keys = sorted(folds_data.keys(), key=int)
            for m in model_order:
                maes = [folds_data[fk][m]["mae"] for fk in fold_keys]
                ax.plot(range(len(fold_keys)), maes, marker=markers[m], color=colors[m],
                        label=m, markersize=8, linewidth=1.5)
            ax.set_title(f"h={h}")
            ax.set_xlabel("Fold")
            ax.set_xticks(range(len(fold_keys)))
            ax.set_xticklabels([f"F{k}" for k in fold_keys])
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("MAE (SP500 test)")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        plt.suptitle("Phase 4 — Multivariate walk-forward MAE per fold", y=1.02)
        plt.tight_layout()
        fig_path = FIGURES_DIR / "12_multivariate_per_fold.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved {fig_path}")
    except Exception as e:
        print(f"[viz] failed: {e}")

    # Markdown summary
    md = ["# Phase 4 — Multivariate walk-forward + Diebold-Mariano", ""]
    cfg = out["config"]
    md.append(f"**Channels**: {', '.join(cfg['channels'])}")
    md.append(f"**Target**: {cfg['target_col']}  | **Overlap**: 1992-02 to 2023-09 ({cfg['n_series']} months)")
    md.append(f"**Setup**: anchored expanding-window walk-forward, {cfg['n_folds_per_horizon'][1]} folds, "
              f"initial_train={cfg['initial_train']}, test_window={cfg['test_window']}, step={cfg['step']}, "
              f"SEQ_LEN={cfg['seq_len']}, {len(cfg['seeds'])} seeds, device_workers={cfg['device_workers']}, n_jobs={cfg['n_jobs']}.")
    md.append("")
    md.append(f"**Wall clock**: {cfg['wall_clock_s']:.1f}s  ({cfg['wall_clock_s']/60:.1f} min)")
    md.append("")
    md.append("## Aggregate MAE (mean over folds)")
    md.append("")
    md.append("| Horizon | " + " | ".join(model_order) + " |")
    md.append("|---" * (len(model_order) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for name in model_order:
            cells.append(f"{aggregate[f'h{h}'][name]['mae_mean']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Diebold-Mariano: PAB(5ch) vs each baseline")
    md.append("")
    md.append("Negative DM stat = PAB(5ch) better. p-values from Student-t (df=n-1).")
    md.append("")
    for h in HORIZONS:
        md.append(f"### h={h}")
        md.append("")
        md.append("| Baseline | DM stat | p-value | SE PAB | SE base | SE reduction | sig |")
        md.append("|---|---|---|---|---|---|---|")
        for baseline, dm in dm_results[f"h{h}"].items():
            sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else ("*" if dm["p_value"] < 0.10 else ""))
            md.append(f"| {baseline} | {dm['dm_stat']:.3f} | {dm['p_value']:.4f} | "
                      f"{dm['mean_se_pab']:.6f} | {dm['mean_se_base']:.6f} | "
                      f"{dm['se_reduction_pct']:.2f}% | {sig} |")
        md.append("")
    md_path = RESULTS_DIR / "12_multivariate_summary.md"
    md_path.write_text("\n".join(md))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
