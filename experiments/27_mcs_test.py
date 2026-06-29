"""
Experiment 27: Model Confidence Set (MCS) test (Hansen, Lunde, Nason 2011)
===========================================================================

Computes the MCS over 8 SOTA variants using squared error loss per test window.
The MCS is the largest set of models whose performance is statistically
indistinguishable from the best, controlling familywise error.

Protocol: same as exp 22 — 8 variants x 4 horizons x 3 seeds x 5 walk-forward
folds, trained for 80 epochs.  Output: per-cell (variant, horizon, seed, fold)
squared-error vectors, then MCS per horizon.

Backbones: TimesNet, DLinear, N-BEATS, iTransformer (PatchTST removed 2026-06-28
after the patching-sigma incompatibility was documented; see
DECISION_PATCHTST_DROPPED.md).

Usage (Windows GPU box):
    python experiments/27_mcs_test.py

Outputs:
    results/27_mcs.json        -- MCS inclusion table per horizon
    results/27_sqerrs.npz      -- raw squared errors per (variant, horizon)
"""

import sys, os, json, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ampd import AMPD
from src.models.nbeats import NBEATSBackbone
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

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
N_MIXTURES = 4
HIDDEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

VARIANTS = [
    "TimesNet_point", "TimesNet_gmm",
    "DLinear_point", "DLinear_gmm",
    "NBEATS_point", "NBEATS_gmm",
    "iTransformer_point", "iTransformer_gmm",
]


# ============================================================================
# Backbones (verbatim from exp 22 for reproducibility)
# ============================================================================
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
        return agg


class DLinearBackbone(nn.Module):
    def __init__(self, seq_len, hidden=HIDDEN, kernel=25):
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


class PointHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        return self.fc(x)                       # (B, H)


class GMMHead(nn.Module):
    """GMM head returning (mu, sigma, pi) — matches wpn.gmm_nll signature."""
    def __init__(self, hidden, horizon, n_mixtures=N_MIXTURES):
        super().__init__()
        self.horizon = horizon
        self.n_mixtures = n_mixtures
        self.fc_mu = nn.Linear(hidden, horizon * n_mixtures)
        self.fc_sigma = nn.Linear(hidden, horizon * n_mixtures)
        self.fc_pi = nn.Linear(hidden, horizon * n_mixtures)

    def forward(self, x):
        B = x.shape[0]
        mu = self.fc_mu(x).view(B, self.horizon, 1, self.n_mixtures)
        sigma = F.softplus(self.fc_sigma(x)).view(B, self.horizon, 1, self.n_mixtures) + 1e-3
        pi = F.softmax(self.fc_pi(x), dim=-1).view(B, self.horizon, 1, self.n_mixtures)
        return mu, sigma, pi


def build_model(variant, horizon):
    if variant.startswith("TimesNet"):
        backbone = TimesNetBackbone(SEQ_LEN)
    elif variant.startswith("DLinear"):
        backbone = DLinearBackbone(SEQ_LEN)
    elif variant.startswith("NBEATS"):
        backbone = NBEATSBackbone(SEQ_LEN)
    elif variant.startswith("iTransformer"):
        backbone = ITransformerBackboneAdapter(SEQ_LEN)
    else:
        raise ValueError(variant)
    if variant.endswith("_point"):
        head = PointHead(HIDDEN, horizon)
        head_type = "point"
    else:
        head = GMMHead(HIDDEN, horizon)
        head_type = "gmm"
    return backbone, head, head_type


# ============================================================================
# Data
# ============================================================================
def load_univariate_returns():
    import pandas as pd
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    return series


def make_supervised(series, seq_len=SEQ_LEN, horizon=1):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def build_folds(series, horizon):
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    X_all, Y_all = make_supervised(series, SEQ_LEN, horizon)
    folds = []
    fold_idx = 0
    train_end_series = init_train
    while train_end_series + test_window + horizon <= n:
        test_end_series = train_end_series + test_window
        train_end_pair = train_end_series - SEQ_LEN
        test_end_pair = test_end_series - SEQ_LEN
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


# ============================================================================
# Worker — train one cell, return per-test-window squared errors
# ============================================================================
def train_and_collect_sqerr(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone, head, head_type = build_model(variant, horizon)
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
    else:
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
            else:
                mu, sigma, pi = out
                loss = crit((mu, sigma, pi), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    # Predict
    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if head_type == "point":
            y_pred = out.cpu().numpy()                            # (n_test, H)
        else:
            mu, sigma, pi = out
            y_pred = gmm_point_predict(mu, pi).cpu().numpy().squeeze(-1)  # (n_test, H)

    # Per-test-window mean squared error
    sqerr = ((y_pred - Y_test) ** 2).mean(axis=1)                # (n_test_windows,)
    return sqerr.astype(np.float32)


# ============================================================================
# MCS test (Hansen-Lunde-Nason 2011) on squared errors
# ============================================================================
def mcs_test(losses: np.ndarray, alpha: float = 0.1, n_boot: int = 5000, seed: int = 0):
    """
    losses: (T, M) array — T observations, M models.
    Returns: dict with 'mcs_indices', 'pvals', 'drop_history'.

    Step-down: keep dropping the worst model until all survivors pass the
    bootstrap-based MCS inclusion test.
    """
    rng = np.random.default_rng(seed)
    T, M = losses.shape

    candidates = list(range(M))
    drop_history = []

    while len(candidates) > 1:
        idx = np.array(candidates)
        L = losses[:, idx]
        d = L.mean(axis=0)
        Om = np.cov(L.T, ddof=1)
        if Om.ndim == 0:
            Om = np.array([[float(Om)]])
        Om_inv = np.linalg.pinv(Om)

        diff = d[:, None] - d[None, :]
        var_d = np.diag(Om_inv)[:, None] + np.diag(Om_inv)[None, :] - 2 * Om_inv
        var_d = var_d / T
        with np.errstate(divide="ignore", invalid="ignore"):
            t_mat = diff / np.sqrt(np.maximum(var_d, 1e-12))
        np.fill_diagonal(t_mat, -np.inf)
        TR_stat = float(t_mat.max())

        # Bootstrap TR distribution
        L_centered = L - L.mean(axis=0, keepdims=True)
        boot_TR = np.empty(n_boot)
        for b in range(n_boot):
            ix = rng.integers(0, T, size=T)
            L_b = L_centered[ix]
            Om_b = (L_b.T @ L_b) / (T - 1)
            Om_b_inv = np.linalg.pinv(Om_b)
            d_b = L[ix].mean(axis=0)
            diff_b = d_b[:, None] - d_b[None, :]
            var_b = np.diag(Om_b_inv)[:, None] + np.diag(Om_b_inv)[None, :] - 2 * Om_b_inv
            var_b = var_b / T
            with np.errstate(divide="ignore", invalid="ignore"):
                t_mat_b = diff_b / np.sqrt(np.maximum(var_b, 1e-12))
            np.fill_diagonal(t_mat_b, -np.inf)
            boot_TR[b] = t_mat_b.max()

        p_val = float((boot_TR >= TR_stat).mean())
        if p_val > alpha:
            break

        worst_local = int(np.argmax(d))
        worst_global = candidates[worst_local]
        drop_history.append({"dropped_model_idx": worst_global,
                             "TR_stat": TR_stat, "p_val": p_val})
        candidates.pop(worst_local)

    pvals = np.full(M, np.nan)
    for entry in drop_history:
        pvals[entry["dropped_model_idx"]] = entry["p_val"]

    return {
        "mcs_indices": candidates,
        "pvals": pvals.tolist(),
        "drop_history": drop_history,
        "alpha": alpha,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print(" Experiment 27: MCS test (Hansen-Lunde-Nason 2011) over 8 SOTA variants")
    print("=" * 72)
    print(f"[setup] {len(VARIANTS)} variants x {len(HORIZONS)} horizons x {len(SEEDS)} seeds x 5 folds")

    series = load_univariate_returns()
    print(f"[data] {len(series)} monthly log-returns, range {series.min():.3f} to {series.max():.3f}")

    # Storage: sqerrs_by_variant[variant][horizon] = np.array (all windows concatenated)
    sqerrs_by_variant = {v: {} for v in VARIANTS}

    for horizon in HORIZONS:
        print(f"\n=== Horizon h={horizon} ===")
        folds = build_folds(series, horizon)
        # Build task list
        tasks = []
        for seed in SEEDS:
            for fi, f in enumerate(folds):
                for v in VARIANTS:
                    tasks.append((v, horizon, fi, seed,
                                  f["X_train"], f["Y_train"], f["X_test"], f["Y_test"]))
        print(f"[tasks] {len(tasks)} total for h={horizon}")

        t0 = time.time()
        results = Parallel(n_jobs=1, verbose=0)(
            delayed(train_and_collect_sqerr)(t) for t in tasks
        )
        elapsed = time.time() - t0
        print(f"[trained] {len(results)} cells in {elapsed:.1f}s")

        # Aggregate per variant
        for v in VARIANTS:
            v_idx = [i for i, t in enumerate(tasks) if t[0] == v]
            sqerrs_by_variant[v][horizon] = np.concatenate([results[i] for i in v_idx])
            print(f"  {v:<20}  n_obs={len(sqerrs_by_variant[v][horizon]):4d}  "
                  f"mean_sqerr={sqerrs_by_variant[v][horizon].mean():.6f}")

    # Save raw sqerrs
    npz_path = RESULTS_DIR / "27_sqerrs.npz"
    np.savez_compressed(
        npz_path,
        **{f"{v}__h{h}": sqerrs_by_variant[v][h] for v in VARIANTS for h in HORIZONS},
    )
    print(f"\n[saved] {npz_path}")

    # MCS test per horizon
    mcs_results = {}
    for horizon in HORIZONS:
        T = min(len(sqerrs_by_variant[v][horizon]) for v in VARIANTS)
        L = np.stack([sqerrs_by_variant[v][horizon][:T] for v in VARIANTS], axis=1)

        result = mcs_test(L, alpha=0.10, n_boot=2000, seed=42)
        mcs_set = result["mcs_indices"]
        mcs_names = [VARIANTS[i] for i in mcs_set]

        mcs_results[f"h{horizon}"] = {
            "T_observations": int(T),
            "alpha": 0.10,
            "MCS_models": mcs_names,
            "MCS_size": len(mcs_names),
            "per_model_pvals": dict(zip(VARIANTS,
                                        [None if np.isnan(p) else float(p)
                                         for p in result["pvals"]])),
            "drop_order": [
                {"model": VARIANTS[entry["dropped_model_idx"]],
                 "TR_stat": entry["TR_stat"], "p_val": entry["p_val"]}
                for entry in result["drop_history"]
            ],
        }
        print(f"\n[h={horizon}] 90% MCS: {mcs_names}  (size {len(mcs_names)})")
        for entry in result["drop_history"]:
            print(f"   dropped {VARIANTS[entry['dropped_model_idx']]:<20}  "
                  f"TR={entry['TR_stat']:.4f}  p={entry['p_val']:.4f}")

    out = {
        "config": {
            "VARIANTS": VARIANTS,
            "HORIZONS": HORIZONS,
            "SEEDS": SEEDS,
            "EPOCHS": EPOCHS,
            "n_folds": 5,
            "alpha": 0.10,
            "n_boot": 2000,
        },
        "mcs_per_horizon": mcs_results,
    }
    out_path = RESULTS_DIR / "27_mcs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] saved {out_path}")


if __name__ == "__main__":
    main()