"""
Experiment 11: Walk-forward validation + Diebold-Mariano test.

Goal: replace the 70/30 holdout (which the reviewer called "fake") with a real
rolling-origin walk-forward. Compute DM test statistics for PAB vs each baseline.

Setup:
- Anchored expanding-window walk-forward, 5 folds (~11-year test chunks each)
- Initial train: first 50% of series (~77 years), test on the next 7%, slide by 10%
- 3 seeds for deep models, averaged per fold (Linear is deterministic)
- Loss: MAE and squared error; DM test uses squared error differential
- DM statistic with Newey-West HAC variance, lag = horizon - 1

Parallelism:
- joblib.Parallel(n_jobs=-1, backend='loky') over the (model, horizon, fold, seed) cross product
- Deep models run on CPU inside workers (MPS/CUDA serializes across processes)
  so 8-core parallel beats single-GPU serial for our small models on small data.
- Linear runs on sklearn (instant).

Output:
- results/11_walkforward_dm.json: per-fold metrics + DM stats
- results/11_walkforward_summary.md: human-readable summary
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# joblib/loky: cap each worker to 1 thread (we parallelize across processes)
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
SEEDS = [0, 1, 2]

# Walk-forward: anchored expanding window
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10

# Training
EPOCHS = 60
LR = 1e-3
BS = 128
HIDDEN = 64

DEVICE_FOR_SERIAL = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
# Workers run on CPU to avoid GPU contention across processes
DEVICE = "cpu"


# ============================================================
# Baseline models (mirrors experiments/10)
# ============================================================

class DLinear(nn.Module):
    def __init__(self, seq_len, horizon, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        pad = (self.kernel_size - 1) // 2
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, self.kernel_size, stride=1)
        seasonal = x_t - trend
        trend = trend.squeeze(1)
        seasonal = seasonal.squeeze(1)
        y = self.linear_trend(trend) + self.linear_seasonal(seasonal)
        return y.unsqueeze(-1)


class NBEATSBlock(nn.Module):
    def __init__(self, seq_len, horizon, hidden=64, theta_dim=8, basis="trend"):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.basis = basis
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.theta_b = nn.Linear(hidden, theta_dim)
        self.theta_f = nn.Linear(hidden, theta_dim)
        t_b = np.arange(seq_len, dtype=np.float32) / seq_len
        t_f = np.arange(horizon, dtype=np.float32) / horizon
        self.register_buffer("T_b", torch.tensor(t_b))
        self.register_buffer("T_f", torch.tensor(t_f))
        self.p = theta_dim // 2 if basis == "seasonality" else theta_dim

    def _basis(self, theta, T):
        return sum(theta[..., i:i+1] * (T ** i) for i in range(theta.shape[-1]))

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        b = self._basis(self.theta_b(h), self.T_b)
        f = self._basis(self.theta_f(h), self.T_f)
        return b, f


class NBEATS(nn.Module):
    def __init__(self, seq_len, horizon, hidden=64, theta_dim=8, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(seq_len, horizon, hidden, theta_dim,
                        basis=("trend" if i % 2 == 0 else "seasonality"))
            for i in range(n_blocks)
        ])

    def forward(self, x):
        x_in = x.squeeze(-1)
        residual = x_in
        forecast = torch.zeros(x.shape[0], self.blocks[0].horizon, device=x.device)
        for blk in self.blocks:
            b, f = blk(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast.unsqueeze(-1)


class PatchTST(nn.Module):
    def __init__(self, seq_len, horizon, patch_len=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                                batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model * self.n_patches, horizon)

    def forward(self, x):
        B, T, C = x.shape
        x = x.squeeze(-1)
        n_patches = T // self.patch_len
        patches = x[:, :n_patches * self.patch_len].reshape(B, n_patches, self.patch_len)
        h = self.patch_proj(patches)
        h = h + self.pos_emb[:, :n_patches]
        h = self.encoder(h)
        h = h.reshape(B, -1)
        return self.head(h).unsqueeze(-1)


class iTransformer(nn.Module):
    def __init__(self, seq_len, horizon, n_channels=1, d_model=64, n_heads=4, n_layers=2):
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
        h = x.transpose(1, 2)
        h = self.proj_in(h)
        h = h + self.pos_emb
        h = self.encoder(h)
        y = self.proj_out(h)
        return y.transpose(1, 2)


class TimesNet(nn.Module):
    def __init__(self, seq_len, horizon, n_channels=1, top_k=2, hidden=64, n_layers=2):
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


MODELS = {
    "Linear": None,
    "DLinear": DLinear,
    "NBEATS": NBEATS,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "TimesNet": TimesNet,
    "PAB(ours)": PeriodAttentionTimesNetLite,
}


# ============================================================
# Data: walk-forward folds
# ============================================================

def make_supervised(series, seq_len, horizon):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_folds(series, seq_len, horizon):
    """Anchored expanding-window walk-forward. Returns list of dicts per fold."""
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)

    X_all, Y_all = make_supervised(series, seq_len, horizon)

    # Train/test pair index range: pair t covers input ending at t+seq_len-1 and target
    # starting at t+seq_len, ending at t+seq_len+horizon-1.
    # So train_end_index (exclusive in pair index) corresponds to series position train_end.
    folds = []
    fold_idx = 0
    train_end_series = init_train
    while train_end_series + test_window + horizon <= n:
        test_end_series = train_end_series + test_window
        # pair index = series_index - seq_len
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
        train_end_series += step
    return folds


# ============================================================
# Worker: train + predict for one (model, horizon, fold, seed) tuple
# ============================================================

def train_and_predict(args):
    """Returns (model_name, horizon, fold, seed, predictions, truth, train_seconds)."""
    model_name, cls, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()

    if model_name == "Linear":
        # Per-horizon-step linear regression on sklearn (fast, deterministic)
        preds = []
        for k in range(Y_train.shape[1]):
            lr = LinearRegression()
            lr.fit(X_train, Y_train[:, k])
            preds.append(lr.predict(X_test))
        yp = np.array(preds).T  # (n_test, horizon)
    else:
        # Build model
        if cls is PeriodAttentionTimesNetLite:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=1,
                          top_k=2, hidden=HIDDEN, num_heads=4, dropout=0.1)
        elif cls is DLinear:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon)
        elif cls is NBEATS:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, hidden=HIDDEN)
        elif cls is PatchTST:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, d_model=HIDDEN, n_heads=4, n_layers=2)
        elif cls is iTransformer:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=1, d_model=HIDDEN)
        elif cls is TimesNet:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=1, top_k=2, hidden=HIDDEN)
        else:
            kwargs = {}

        model = cls(**kwargs).to(DEVICE)

        # Fit periods on training data for periodic models
        if cls is PeriodAttentionTimesNetLite:
            model.fit_periods(X_train[:, 0], max_period=360, min_period=4)
            model = model.to(DEVICE)
        elif cls is TimesNet:
            model.fit_periods(X_train[:, 0])
            model.proj = model.proj.to(DEVICE)

        X = torch.tensor(X_train, dtype=torch.float32)
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
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

        # Predict
        model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32)
        if Xt.ndim == 2:
            Xt = Xt.unsqueeze(-1)
        with torch.no_grad():
            yp = model(Xt).cpu().numpy().squeeze(-1)

    elapsed = time.time() - t0
    return {
        "model": model_name,
        "horizon": horizon,
        "fold": fold_idx,
        "seed": seed,
        "preds": yp.astype(np.float32),
        "truth": Y_test.astype(np.float32),
        "train_seconds": elapsed,
    }


# ============================================================
# Diebold-Mariano with Newey-West HAC
# ============================================================

def dm_test(e1: np.ndarray, e2: np.ndarray, horizon: int, loss: str = "se") -> Dict[str, float]:
    """
    Diebold-Mariano test.
    e1, e2: 1D arrays of per-step forecast errors (model 1, model 2).
    horizon: forecast horizon (used as Newey-West truncation lag).
    Returns dict with DM stat, p-value (two-sided), mean loss diff.
    """
    assert e1.shape == e2.shape
    n = len(e1)
    if loss == "se":
        d = e1 ** 2 - e2 ** 2
    elif loss == "ae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(loss)

    d = d - d.mean()
    n = len(d)
    # Newey-West HAC with lag = horizon - 1
    h_lag = max(horizon - 1, 1)
    gamma0 = np.sum(d * d) / n
    var_d = gamma0
    for k in range(1, h_lag + 1):
        gamma_k = np.sum(d[k:] * d[:-k]) / n
        var_d += 2.0 * (1.0 - k / (h_lag + 1)) * gamma_k
    var_d = max(var_d, 1e-12)

    mean_d = float(np.mean(e1 ** 2) - np.mean(e2 ** 2)) if loss == "se" else float(np.mean(np.abs(e1)) - np.mean(np.abs(e2)))
    dm_stat = mean_d / np.sqrt(var_d / n)
    # Two-sided p-value via t-distribution with n-1 df; for large n ~ N(0,1)
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(dm_stat), df=n - 1))
    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": mean_d,  # positive means model 2 (e2) is better (smaller error)
        "n": int(n),
        "h_lag": int(h_lag),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Experiment 11: Walk-forward + Diebold-Mariano")
    print("=" * 70)
    print(f"[device serial] {DEVICE_FOR_SERIAL}  [device workers] {DEVICE}")
    print(f"[cpus] {os.cpu_count()}  [torch threads] {torch.get_num_threads()}")

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    print(f"[data] n={n}, initial_train={init_train}, test_window={test_window}, step={step}")
    print(f"[data] date range: {panel.index[0].date()} -> {panel.index[-1].date()}")

    # Build folds per horizon
    folds_by_h = {}
    for h in HORIZONS:
        folds = build_folds(series, SEQ_LEN, h)
        folds_by_h[h] = folds
        print(f"[folds h={h}] n_folds={len(folds)}, "
              f"first test {panel.index[folds[0]['test_start_series']].date()} -> "
              f"{panel.index[folds[0]['test_end_series']-1].date()}, "
              f"last test {panel.index[folds[-1]['test_start_series']].date()} -> "
              f"{panel.index[folds[-1]['test_end_series']-1].date()}")

    # Build task list: (model, cls, horizon, fold, seed, ...)
    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for name, cls in MODELS.items():
                for seed in (SEEDS if cls is not None else [0]):
                    tasks.append((
                        name, cls, h, fold["fold"], seed,
                        fold["X_train"], fold["Y_train"],
                        fold["X_test"], fold["Y_test"],
                    ))
    print(f"[tasks] {len(tasks)} total  ({sum(1 for t in tasks if t[1] is None)} Linear + "
          f"{sum(1 for t in tasks if t[1] is not None)} deep)")

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

    # DM tests: PAB vs each baseline, per horizon, on concatenated test errors
    dm_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for h in HORIZONS:
        dm_results[f"h{h}"] = {}
        # Build concatenated truth and predictions per (model, fold)
        for baseline in MODELS.keys():
            if baseline == "PAB(ours)":
                continue
            # Concatenate predictions across folds (each prediction is (n_test, horizon))
            # For DM with multi-horizon: collapse horizon by taking MAE-equivalent per row:
            # use mean squared error across horizon steps → (n_test,) loss per row.
            pab_preds_all = []
            base_preds_all = []
            truth_all = []
            for fold_idx in range(len(folds_by_h[h])):
                pab_key = ("PAB(ours)", h, fold_idx)
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
            # Per-row squared error (mean across horizon)
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
            "initial_train_frac": INITIAL_TRAIN_FRAC,
            "test_frac": TEST_FRAC,
            "step_frac": STEP_FRAC,
            "epochs": EPOCHS,
            "lr": LR,
            "hidden": HIDDEN,
            "device_workers": DEVICE,
            "n_jobs": n_jobs,
            "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start,
            "n_series": int(n),
            "n_folds_per_horizon": {h: len(folds_by_h[h]) for h in HORIZONS},
        },
        "aggregate": aggregate,
        "per_fold": per_fold_metrics,
        "dm_test": dm_results,
    }
    out_path = RESULTS_DIR / "11_walkforward_dm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(" Walk-forward aggregate MAE (mean across folds)")
    print("=" * 70)
    header = f"{'Horizon':<8}"
    for name in MODELS.keys():
        header += f" {name[:11]:>11}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        row = f"  h={h:<4}"
        for name in MODELS.keys():
            mae = aggregate[f"h{h}"][name]["mae_mean"]
            row += f" {mae:>11.5f}"
        print(row)

    print("\nBest model per horizon (by MAE):")
    for h in HORIZONS:
        best = min(MODELS.keys(), key=lambda k: aggregate[f"h{h}"][k]["mae_mean"])
        print(f"  h={h}: {best}  MAE={aggregate[f'h{h}'][best]['mae_mean']:.5f}")

    print("\n" + "=" * 70)
    print(" Diebold-Mariano: PAB(ours) vs baseline  (negative DM stat = PAB better)")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n  h={h}:")
        print(f"    {'baseline':<14} {'DM stat':>10} {'p-value':>10} {'SE PAB':>10} {'SE base':>10} {'SE red%':>8}  sig")
        for baseline, dm in dm_results[f"h{h}"].items():
            sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else ("*" if dm["p_value"] < 0.10 else ""))
            print(f"    {baseline:<14} {dm['dm_stat']:>10.3f} {dm['p_value']:>10.4f} "
                  f"{dm['mean_se_pab']:>10.6f} {dm['mean_se_base']:>10.6f} "
                  f"{dm['se_reduction_pct']:>7.2f}%  {sig}")

    # Markdown summary
    md = ["# Phase 3 — Walk-forward + Diebold-Mariano", ""]
    cfg = out["config"]
    md.append(f"**Setup**: anchored expanding-window walk-forward, "
              f"{cfg['n_folds_per_horizon'][1]} folds, "
              f"SEQ_LEN={cfg['seq_len']}, "
              f"epochs={cfg['epochs']}, "
              f"{len(cfg['seeds'])} seeds, "
              f"device_workers={cfg['device_workers']} (n_jobs={cfg['n_jobs']}).")
    md.append("")
    md.append(f"**Wall clock**: {cfg['wall_clock_s']:.1f}s  ({cfg['wall_clock_s']/60:.1f} min)")
    md.append("")
    md.append("## Aggregate MAE (mean over folds)")
    md.append("")
    md.append("| Horizon | " + " | ".join(MODELS.keys()) + " |")
    md.append("|---" * (len(MODELS) + 1) + "|")
    for h in HORIZONS:
        cells = [f"h={h}"]
        for name in MODELS.keys():
            cells.append(f"{aggregate[f'h{h}'][name]['mae_mean']:.5f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Diebold-Mariano: PAB vs each baseline")
    md.append("")
    md.append("Negative DM stat = PAB better. p-values from Student-t (df=n-1).")
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
    md_path = RESULTS_DIR / "11_walkforward_summary.md"
    md_path.write_text("\n".join(md))
    print(f"\nSaved {md_path}")


if __name__ == "__main__":
    main()
