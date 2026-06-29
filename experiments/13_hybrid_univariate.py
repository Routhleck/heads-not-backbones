"""
Experiment 13: HybridPAB benchmark — Phase 3 univariate walk-forward (replay).

Same as exp 11 but ADDS HybridPAB (uni) to the model lineup. Re-runs all models
on the same walk-forward folds for clean MAE + DM comparison.

Models: Linear, DLinear, NBEATS, PatchTST, iTransformer, TimesNet, PAB(ours), HybridPAB(ours)
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
from src.models.hybrid_pab import HybridPAB

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
EPOCHS = 60
LR = 1e-3
BS = 128
HIDDEN = 64
# Auto-detect device: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# Models (same as exp 11)
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
    "Linear": ("linear", None),
    "DLinear": (DLinear, None),
    "NBEATS": (NBEATS, None),
    "PatchTST": (PatchTST, None),
    "iTransformer": (iTransformer, None),
    "TimesNet": (TimesNet, None),
    "PAB(ours)": (PeriodAttentionTimesNetLite, None),
    "HybridPAB(ours)": (HybridPAB, None),
}


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


def train_and_predict(args):
    model_name, model_spec, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    if model_spec == "linear":
        preds = []
        for k in range(Y_train.shape[1]):
            lr = LinearRegression()
            lr.fit(X_train, Y_train[:, k])
            preds.append(lr.predict(X_test))
        yp = np.array(preds).T
    else:
        cls = model_spec
        if cls is PeriodAttentionTimesNetLite:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=1,
                          top_k=2, hidden=HIDDEN, num_heads=4, dropout=0.1)
        elif cls is HybridPAB:
            kwargs = dict(seq_len=SEQ_LEN, horizon=horizon, n_channels=1,
                          hidden=HIDDEN, num_heads=4, dropout=0.1, output_channels=1)
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

        if cls is PeriodAttentionTimesNetLite:
            model.fit_periods(X_train[:, 0], max_period=360, min_period=4)
            model = model.to(DEVICE)
        elif cls is HybridPAB:
            model.fit_periods(X_train[:, 0], max_period=360, min_period=4)
            model = model.to(DEVICE)
        elif cls is TimesNet:
            model.fit_periods(X_train[:, 0])
            model.proj = model.proj.to(DEVICE)

        X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
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
        Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            yp = model(Xt).cpu().numpy().squeeze(-1)

    return {
        "model": model_name, "horizon": horizon, "fold": fold_idx, "seed": seed,
        "preds": yp.astype(np.float32), "truth": Y_test.astype(np.float32),
        "train_seconds": time.time() - t0,
    }


def dm_test(e1, e2, horizon):
    d = e1 ** 2 - e2 ** 2
    d_centered = d - d.mean()
    n = len(d_centered)
    h_lag = max(horizon - 1, 1)
    gamma0 = np.sum(d_centered * d_centered) / n
    var_d = gamma0
    for k in range(1, h_lag + 1):
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / n
        var_d += 2.0 * (1.0 - k / (h_lag + 1)) * gamma_k
    var_d = max(var_d, 1e-12)
    mean_d = float(np.mean(e1 ** 2) - np.mean(e2 ** 2))
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - sp_stats.t.cdf(abs(dm_stat), df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value),
            "mean_diff": mean_d, "n": int(n), "h_lag": int(h_lag)}


def main():
    print("=" * 70)
    print(" Experiment 13: HybridPAB univariate walk-forward (Phase 3 replay + HybridPAB)")
    print("=" * 70)
    print(f"[device workers] {DEVICE}  [cpus] {os.cpu_count()}")

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    n = len(series)
    print(f"[data] n={n}")

    folds_by_h = {}
    for h in HORIZONS:
        folds_by_h[h] = build_folds(series, SEQ_LEN, h)
        print(f"[folds h={h}] n_folds={len(folds_by_h[h])}")

    tasks = []
    for h in HORIZONS:
        for fold in folds_by_h[h]:
            for name, (cls, _) in MODELS.items():
                for seed in (SEEDS if cls != "linear" else [0]):
                    tasks.append((name, cls, h, fold["fold"], seed,
                                  fold["X_train"], fold["Y_train"],
                                  fold["X_test"], fold["Y_test"]))
    print(f"[tasks] {len(tasks)} total  ({sum(1 for t in tasks if t[1]=='linear')} Linear + "
          f"{sum(1 for t in tasks if t[1]!='linear')} deep)")

    t_start = time.time()
    n_jobs = min(os.cpu_count() or 1, 8)
    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(train_and_predict)(t) for t in tasks
    )
    print(f"[parallel] all tasks done in {time.time() - t_start:.1f}s")

    by_mhf: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
    seed_count: Dict[Tuple[str, int, int], int] = {}
    for r in results_list:
        key = (r["model"], r["horizon"], r["fold"])
        if key not in by_mhf:
            by_mhf[key] = {"preds": np.zeros_like(r["preds"], dtype=np.float64),
                           "truth": r["truth"]}
            seed_count[key] = 0
        by_mhf[key]["preds"] += r["preds"]
        seed_count[key] += 1
    for key in by_mhf:
        by_mhf[key]["preds"] /= seed_count[key]

    per_fold_metrics = {}
    for (model, h, fold), data in by_mhf.items():
        per_fold_metrics.setdefault(f"h{h}", {}).setdefault(fold, {})[model] = {
            "mae": float(mean_absolute_error(data["truth"], data["preds"])),
            "rmse": float(np.sqrt(mean_squared_error(data["truth"], data["preds"]))),
        }

    aggregate = {}
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

    # DM tests: HybridPAB vs each baseline
    baselines = ["Linear", "DLinear", "NBEATS", "PatchTST", "iTransformer", "TimesNet", "PAB(ours)"]
    dm_results = {}
    for h in HORIZONS:
        dm_results[f"h{h}"] = {}
        n_folds = max(f["fold"] for f in folds_by_h[h]) + 1
        for baseline in baselines:
            hybrid_preds_all, base_preds_all, truth_all = [], [], []
            for fi in range(n_folds):
                hybrid_key = ("HybridPAB(ours)", h, fi)
                base_key = (baseline, h, fi)
                if hybrid_key not in by_mhf or base_key not in by_mhf:
                    continue
                hybrid_preds_all.append(by_mhf[hybrid_key]["preds"])
                base_preds_all.append(by_mhf[base_key]["preds"])
                truth_all.append(by_mhf[hybrid_key]["truth"])
            if not hybrid_preds_all:
                continue
            hp = np.concatenate(hybrid_preds_all)
            bp = np.concatenate(base_preds_all)
            tc = np.concatenate(truth_all)
            e_h = np.mean((hp - tc) ** 2, axis=1)
            e_b = np.mean((bp - tc) ** 2, axis=1)
            dm = dm_test(e_h, e_b, horizon=h)
            dm["mean_se_hybrid"] = float(np.mean(e_h))
            dm["mean_se_base"] = float(np.mean(e_b))
            dm["se_reduction_pct"] = float(100.0 * (1.0 - dm["mean_se_hybrid"] / max(dm["mean_se_base"], 1e-12)))
            dm_results[f"h{h}"][baseline] = dm

    out = {
        "config": {
            "seq_len": SEQ_LEN, "horizons": HORIZONS, "seeds": SEEDS,
            "initial_train_frac": INITIAL_TRAIN_FRAC, "test_frac": TEST_FRAC,
            "step_frac": STEP_FRAC, "epochs": EPOCHS, "lr": LR, "hidden": HIDDEN,
            "device_workers": DEVICE, "n_jobs": n_jobs, "n_total_tasks": len(tasks),
            "wall_clock_s": time.time() - t_start, "n_series": int(n),
        },
        "aggregate": aggregate,
        "per_fold": per_fold_metrics,
        "dm_test": dm_results,
    }
    out_path = RESULTS_DIR / "13_hybrid_univariate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print(" Aggregate MAE (mean over folds)")
    print("=" * 70)
    model_order = ["Linear", "DLinear", "NBEATS", "PatchTST", "iTransformer", "TimesNet",
                   "PAB(ours)", "HybridPAB(ours)"]
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

    print("\nBest per horizon:")
    for h in HORIZONS:
        best = min(model_order, key=lambda k: aggregate[f"h{h}"][k]["mae_mean"])
        print(f"  h={h}: {best}  MAE={aggregate[f'h{h}'][best]['mae_mean']:.5f}")

    print("\n" + "=" * 70)
    print(" Diebold-Mariano: HybridPAB vs baseline (negative = HybridPAB better)")
    print("=" * 70)
    for h in HORIZONS:
        print(f"\n  h={h}:")
        print(f"    {'baseline':<14} {'DM stat':>10} {'p-value':>10} {'SE Hybrid':>10} {'SE base':>10} {'SE red%':>8}  sig")
        for baseline, dm in dm_results[f"h{h}"].items():
            sig = "***" if dm["p_value"] < 0.01 else ("**" if dm["p_value"] < 0.05 else ("*" if dm["p_value"] < 0.10 else ""))
            print(f"    {baseline:<14} {dm['dm_stat']:>10.3f} {dm['p_value']:>10.4f} "
                  f"{dm['mean_se_hybrid']:>10.6f} {dm['mean_se_base']:>10.6f} "
                  f"{dm['se_reduction_pct']:>7.2f}%  {sig}")


if __name__ == "__main__":
    main()
