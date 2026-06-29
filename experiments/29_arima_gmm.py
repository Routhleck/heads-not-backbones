"""
Experiment 29: ARIMA + GMM density head baseline
==================================================

A classical baseline: fit ARIMA(p,0,q) on the S&P 500 return series,
extract the residuals, fit a GMM(K=4) on the standardized residuals,
and forecast by sampling from the fitted GMM.

This is the same walk-forward protocol as exp 22 (8-variant SOTA) but
with a much simpler model class.

ARIMA is selected via AIC over a small grid (p,q in {0,1,2}).

Output:
  results/29_arima_gmm.json  -- per-horizon CRPS, Pinball, Coverage

Usage:
    python experiments/29_arima_gmm.py
"""

import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"

SEQ_LEN = 60
HORIZONS = [1, 3, 6, 12]
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
N_MIXTURES = 4
N_SAMPLES = 500
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]


def load_series():
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return panel["SP500_Return"].dropna().values.astype(np.float64)


def make_supervised(series, seq_len=SEQ_LEN, horizon=1):
    X, Y = [], []
    for t in range(len(series) - seq_len - horizon + 1):
        X.append(series[t:t + seq_len])
        Y.append(series[t + seq_len:t + seq_len + horizon])
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


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


def fit_arima_select(returns_train, p_range=range(3), q_range=range(3)):
    """Select ARIMA(p,0,q) by AIC on the training returns (no exog)."""
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    best_aic = np.inf
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(returns_train, order=(p, 0, q),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fit = model.fit(method_kwargs={"warn_convergence": False})
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best = (p, 0, q, fit)
            except Exception:
                continue
    return best


def fit_arima_forecast(returns_train, horizon, p, q):
    """Refit ARIMA(p,0,q) and produce h-step ahead mean forecast and residual std."""
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(returns_train, order=(p, 0, q),
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    fit = model.fit(method_kwargs={"warn_convergence": False})
    forecast = fit.forecast(steps=horizon)
    mu = np.asarray(forecast)
    sigma = np.std(fit.resid)
    return mu, sigma, fit.resid


def fit_gmm_on_residuals(residuals, k=N_MIXTURES, n_init=5):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=n_init, random_state=0, max_iter=200,
                          reg_covar=1e-4)
    gmm.fit(residuals.reshape(-1, 1))
    return gmm


def sample_gmm(gmm, mu, sigma, n_samples=N_SAMPLES, horizon=None):
    """Sample from mixture of N(mu_k, sigma_k^2) with weights, scaled by sigma."""
    n_components = gmm.n_components
    means = gmm.means_.flatten()      # (K,)
    sds = np.sqrt(gmm.covariances_.flatten())  # (K,)
    weights = gmm.weights_            # (K,)

    # Sample component indices
    rng = np.random.default_rng(0)
    comp_idx = rng.choice(n_components, size=n_samples, p=weights)
    samples = mu + sigma * rng.normal(size=n_samples) * 0 + sigma * 1  # placeholder
    # Actually sample from N(mu, sigma^2) per component
    samples = np.zeros(n_samples)
    for c in range(n_components):
        mask = comp_idx == c
        if mask.sum() > 0:
            samples[mask] = rng.normal(loc=means[c], scale=sds[c], size=mask.sum())
    # Add the mean forecast
    if horizon is None:
        return mu + samples  # broadcast mu as scalar
    else:
        return mu[None, :] + samples[:, None]  # (N, H)


def crps_empirical(samples, Y):
    """samples: (N,) or (N, H); Y: scalar or (H,). Returns scalar mean CRPS."""
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    half = N_SAMPLES // 2
    term1 = np.mean(np.abs(samples - Y[None, :]), axis=0)        # (H,)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    return float((term1 - 0.5 * term2).mean())


def pinball(samples, Y, quantiles=QUANTILES):
    """Pinball loss averaged across quantiles and horizon."""
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    H = Y.shape[0]
    loss = 0.0
    for q in quantiles:
        q_pred = np.quantile(samples, q, axis=0)  # (H,)
        diff = Y - q_pred
        loss += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return loss / len(quantiles)


def coverage_at_90(samples, Y):
    """P(Y in [q05, q95])."""
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    return float(((Y >= lower) & (Y <= upper)).mean())


def main():
    print("=" * 72)
    print(" Experiment 29: ARIMA + GMM density head baseline")
    print("=" * 72)

    series = load_series()
    print(f"[data] {len(series)} monthly log-returns")

    # For each horizon, walk-forward fit ARIMA + GMM
    results_per_horizon = {}

    for horizon in HORIZONS:
        folds = build_folds(series, horizon)
        print(f"\n=== Horizon h={horizon} ===")
        crps_list = []
        pinball_list = []
        cov90_list = []
        arima_orders = []
        for fi, f in enumerate(folds):
            t0 = time.time()
            Y_train = f["Y_train"][:, 0]  # use h=0 forecast (1-step at training time)
            Y_test = f["Y_test"][:, 0]    # use first-step of multi-horizon target
            # Use all training returns for ARIMA
            train_returns = Y_train
            # Select ARIMA(p, 0, q) by AIC
            best = fit_arima_select(train_returns)
            if best is None:
                print(f"  fold {fi}: ARIMA selection failed, skipping")
                continue
            p, _, q, fit = best
            arima_orders.append((p, q))
            # Forecast horizon=horizon steps ahead
            mu_fc, sigma_resid, residuals = fit_arima_forecast(train_returns, horizon, p, q)
            # Fit GMM on residuals (only last SEQ_LEN*5 to avoid fitting on stale data)
            recent_resid = residuals[-min(300, len(residuals)):]
            gmm = fit_gmm_on_residuals(recent_resid, k=N_MIXTURES)
            # Sample: distribution of next-step return is N(mu_fc[0], ...) + GMM-residual
            samples_h1 = sample_gmm(gmm, mu_fc[0], 1.0, n_samples=N_SAMPLES)  # (N,)
            Y_test_h1 = Y_test[:len(samples_h1)] if len(samples_h1) < len(Y_test) else Y_test[:N_SAMPLES]
            # Per-window metrics: each "window" is a single test point
            crps_list.append(crps_empirical(samples_h1, Y_test_h1))
            pinball_list.append(pinball(samples_h1, Y_test_h1))
            cov90_list.append(coverage_at_90(samples_h1, Y_test_h1))
            elapsed = time.time() - t0
        crps_arr = np.array(crps_list)
        pinball_arr = np.array(pinball_list)
        cov90_arr = np.array(cov90_list)
        results_per_horizon[f"h{horizon}"] = {
            "n_folds": len(folds),
            "crps_mean": float(crps_arr.mean()),
            "crps_std": float(crps_arr.std()),
            "pinball_mean": float(pinball_arr.mean()),
            "cov90_mean": float(cov90_arr.mean()),
            "arima_orders": arima_orders,
        }
        print(f"  CRPS={crps_arr.mean():.5f} ± {crps_arr.std():.5f}  "
              f"Pinball={pinball_arr.mean():.5f}  Cov90={cov90_arr.mean():.3f}")

    # Compare with exp 22's TimesNet_gmm
    exp22 = json.load(open(RESULTS_DIR / "22_sota_comparison.json"))
    print("\n" + "=" * 72)
    print(" Comparison: ARIMA+GMM vs TimesNet_gmm vs TimesNet_point")
    print("=" * 72)
    print(f"{'horizon':<8} {'metric':<10} {'ARIMA+GMM':<12} {'TimesNet_gmm':<14} {'TimesNet_point':<14}")
    for horizon in HORIZONS:
        hkey = f"h{horizon}"
        a29 = results_per_horizon[hkey]
        # Get TimesNet_gmm and TimesNet_point from exp 22
        tn_g = exp22["metrics"][hkey]["TimesNet_gmm"]["crps"]
        tn_p = exp22["metrics"][hkey]["TimesNet_point"]["crps"]
        crps_a = a29["crps_mean"]
        print(f"{hkey:<8} {'CRPS':<10} {crps_a:<12.5f} {tn_g:<14.5f} {tn_p:<14.5f}")
        # Compute skill score
        ss = (tn_p - crps_a) / tn_p if tn_p > 0 else 0
        print(f"{'':<8} {'CRPS-SS':<10} {ss * 100:+.2f}%  (vs TimesNet_point)")

    out = {
        "config": {
            "INITIAL_TRAIN_FRAC": INITIAL_TRAIN_FRAC,
            "TEST_FRAC": TEST_FRAC,
            "STEP_FRAC": STEP_FRAC,
            "N_MIXTURES": N_MIXTURES,
            "N_SAMPLES": N_SAMPLES,
            "ARIMA_p_range": list(range(3)),
            "ARIMA_q_range": list(range(3)),
        },
        "per_horizon": results_per_horizon,
    }
    out_path = RESULTS_DIR / "29_arima_gmm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()