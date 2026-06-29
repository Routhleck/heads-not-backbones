"""
Experiment 33: Trading-strategy backtest
=========================================

Converts quantile forecasts to positions and measures risk-adjusted
returns (Sharpe ratio) of a simple mean-reversion strategy.

Strategy:
  - At each test window t, compute the forecast mean (mu_t) and the
    10%/90% quantile spread (q90 - q10) of the GMM forecast.
  - Position size = -sign(mu_t) / (q90 - q10)
    (Bet against the predicted direction, scaled by uncertainty; if the
    forecast is highly uncertain (wide quantile spread), take a smaller
    position.)
  - Return at t = position_{t-1} * actual_return_t
    (one-period delay: use yesterday's forecast to size today's position).

Compare:
  - TimesNet_gmm strategy
  - TimesNet_point strategy (no uncertainty info → fixed position size)
  - Buy-and-hold benchmark
  - Naive "always-flat" benchmark

Metric: annualized Sharpe ratio (12 monthly returns), max drawdown,
annualized return.

Usage:
    python experiments/33_trading_backtest.py   (Windows GPU box)
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
from src.models.wpn import gmm_nll, gmm_point_predict, sample_gmm

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"

SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
SEEDS = [0]
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
EPOCHS = 80
LR = 1e-3
BS = 128
N_MIXTURES = 4
HIDDEN = 64
N_SAMPLES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================================
# Backbones + heads (verbatim from exp 22)
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


class PointHead(nn.Module):
    def __init__(self, hidden, horizon):
        super().__init__()
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):
        return self.fc(x)


class GMMHead(nn.Module):
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


# ============================================================================
# Data
# ============================================================================
def load_series():
    import pandas as pd
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return panel["SP500_Return"].dropna().values.astype(np.float32)


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
# Worker — train and collect mu, q10, q90 per test window
# ============================================================================
def train_and_get_forecasts(args):
    variant, horizon, fold_idx, seed, X_train, Y_train, X_test, Y_test = args
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = TimesNetBackbone(SEQ_LEN).to(DEVICE)
    if variant.endswith("_point"):
        head = PointHead(HIDDEN, horizon).to(DEVICE)
    else:
        head = GMMHead(HIDDEN, horizon).to(DEVICE)

    X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    Y = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    n = X.shape[0]
    bs = min(BS, n)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                     lr=LR, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    if variant.endswith("_point"):
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
            if variant.endswith("_point"):
                yp = out.unsqueeze(-1)
                loss = crit(yp, Y[idx])
            else:
                mu_, sigma_, pi_ = out
                loss = crit((mu_, sigma_, pi_), Y[idx])
            loss.backward()
            opt.step()
        sched.step()

    backbone.eval()
    head.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        h_test = backbone(Xt)
        out = head(h_test)
        if variant.endswith("_point"):
            y_pred = out.cpu().numpy()
            h_train = backbone(X)
            train_pred = head(h_train).detach().cpu().numpy()
            train_resid = Y_train - train_pred
            sigma_per_step = np.std(train_resid, axis=0)
            np.random.seed(seed)
            samples = np.random.normal(
                y_pred[None, :, :], sigma_per_step[None, None, :],
                size=(N_SAMPLES, y_pred.shape[0], y_pred.shape[1]),
            )
        else:
            mu_, sigma_, pi_ = out
            samples = sample_gmm(mu_, sigma_, pi_, n_samples=N_SAMPLES).squeeze(-1).cpu().numpy()

    # Aggregate: for each test window, mu = mean, q10, q90 across H horizons
    # Use the FIRST horizon (h-step ahead forecast)
    if variant.endswith("_point"):
        # Use point forecast as mu, and Gaussian q10/q90
        mu = y_pred[:, 0]                                # (n_test,)
        q10 = mu + sp_stats.norm.ppf(0.10) * sigma_per_step[0] if False else None
        # Simpler: just use mu and quantiles from samples
        samples_h = samples[:, :, 0]                      # (N, n_test)
        q10 = np.quantile(samples_h, 0.10, axis=0)
        q90 = np.quantile(samples_h, 0.90, axis=0)
        mu = np.mean(samples_h, axis=0)
    else:
        samples_h = samples[:, :, 0]                      # (N, n_test)
        q10 = np.quantile(samples_h, 0.10, axis=0)
        q90 = np.quantile(samples_h, 0.90, axis=0)
        mu = np.mean(samples_h, axis=0)

    return {
        "mu": mu.astype(np.float32),
        "q10": q10.astype(np.float32),
        "q90": q90.astype(np.float32),
        "truth": Y_test[:, 0].astype(np.float32),
        "fold": fold_idx,
    }


# ============================================================================
# Trading metrics
# ============================================================================
def sharpe(returns, rf=0.0):
    """Annualized Sharpe (monthly returns)."""
    excess = returns - rf
    if excess.std() < 1e-12:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(12))


def max_drawdown(returns):
    """Max drawdown of cumulative log-returns."""
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def annualized_return(returns):
    return float(np.expm1(returns.sum() / len(returns) * 12)) if len(returns) > 0 else 0.0


def compute_strategy_metrics(positions, returns):
    """Apply position[t-1] to return[t] to avoid look-ahead."""
    pnl = positions[:-1] * returns[1:]
    return {
        "n_periods": int(len(pnl)),
        "sharpe": sharpe(pnl),
        "max_drawdown": max_drawdown(pnl),
        "annualized_return": annualized_return(pnl),
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
    }


def main():
    print("=" * 72)
    print(" Experiment 33: Trading-strategy backtest")
    print("=" * 72)
    print(f"[setup] TIMESNET_POINT vs TIMESNET_GMM at all 4 horizons")

    series = load_series()
    print(f"[data] {len(series)} monthly log-returns")

    # Collect forecasts across all folds for each (variant, horizon)
    forecasts_by_variant = {v: {} for v in ["TimesNet_point", "TimesNet_gmm"]}

    for horizon in HORIZONS:
        folds = build_folds(series, horizon)
        print(f"\n=== Horizon h={horizon} ===")
        for variant in ["TimesNet_point", "TimesNet_gmm"]:
            tasks = [(variant, horizon, fi, s,
                      f["X_train"], f["Y_train"], f["X_test"], f["Y_test"])
                     for s in SEEDS for fi, f in enumerate(folds)]
            t0 = time.time()
            results = Parallel(n_jobs=1, verbose=0)(
                delayed(train_and_get_forecasts)(t) for t in tasks
            )
            elapsed = time.time() - t0

            # Concatenate across folds
            mu = np.concatenate([r["mu"] for r in results])
            q10 = np.concatenate([r["q10"] for r in results])
            q90 = np.concatenate([r["q90"] for r in results])
            truth = np.concatenate([r["truth"] for r in results])
            forecasts_by_variant[variant][horizon] = {
                "mu": mu, "q10": q10, "q90": q90, "truth": truth,
            }
            print(f"  {variant:<20}  trained {len(tasks)} cells in {elapsed:.1f}s, "
                  f"got {len(truth)} test points")

    # Compute strategies
    print("\n" + "=" * 72)
    print(" Trading-strategy results")
    print("=" * 72)

    results_table = []

    # Benchmark: buy-and-hold
    # We use the truth from h=1 fold set as the SP500 return series
    sp_ret_truth = forecasts_by_variant["TimesNet_gmm"][1]["truth"]

    bh_metrics = compute_strategy_metrics(np.ones(len(sp_ret_truth) - 1), sp_ret_truth[1:])
    print(f"\n[Benchmark: buy-and-hold]  Sharpe={bh_metrics['sharpe']:.3f}  "
          f"AnnReturn={bh_metrics['annualized_return']:.3%}  "
          f"MaxDD={bh_metrics['max_drawdown']:.3f}")
    results_table.append({"strategy": "buy_and_hold", "horizon": 1, **bh_metrics})

    # Benchmark: always flat
    flat_metrics = compute_strategy_metrics(np.zeros(len(sp_ret_truth) - 1), sp_ret_truth[1:])
    print(f"[Benchmark: always flat]  Sharpe={flat_metrics['sharpe']:.3f}  "
          f"AnnReturn={flat_metrics['annualized_return']:.3%}  "
          f"MaxDD={flat_metrics['max_drawdown']:.3f}")
    results_table.append({"strategy": "always_flat", "horizon": 1, **flat_metrics})

    for variant in ["TimesNet_point", "TimesNet_gmm"]:
        for horizon in HORIZONS:
            d = forecasts_by_variant[variant][horizon]
            mu, q10, q90, truth = d["mu"], d["q10"], d["q90"], d["truth"]
            # Uncertainty-aware position: -sign(mu) / (q90 - q10)
            spread = q90 - q10
            spread = np.maximum(spread, 1e-3)  # avoid div-by-zero
            positions = -np.sign(mu) / spread
            # Clip extreme positions
            positions = np.clip(positions, -10, 10)
            metrics = compute_strategy_metrics(positions, truth)
            print(f"\n[{variant:<20} h={horizon}]  Sharpe={metrics['sharpe']:.3f}  "
                  f"AnnReturn={metrics['annualized_return']:.3%}  "
                  f"MaxDD={metrics['max_drawdown']:.3f}  "
                  f"n_periods={metrics['n_periods']}")
            results_table.append({
                "strategy": f"{variant}_uncertainty_sized",
                "horizon": horizon,
                **metrics,
            })

    # Save
    out = {
        "config": {
            "EPOCHS": EPOCHS,
            "HORIZONS": HORIZONS,
            "SEEDS": SEEDS,
            "N_SAMPLES": N_SAMPLES,
            "strategy": "negative sign(mu) / (q90 - q10), clipped to [-10, 10]",
            "execution": "position[t-1] * return[t] (1-period delay)",
        },
        "results": results_table,
    }
    out_path = RESULTS_DIR / "33_trading_backtest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()