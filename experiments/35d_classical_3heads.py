"""
Experiment 35d: Classical baselines × 3 heads (ARIMA & GARCH with point, Gaussian, GMM).

Mirrors the 12-variant deep-backbone experiment (exp 22+34) on classical
baselines. For each (classical model, head) we run the same walk-forward
protocol used in exp 22/29/23:

  ARIMA(p, 0, q) with p,q AIC-selected:
    - point:    forecast = conditional mean
    - Gaussian: forecast = N(mean, sigma_resid^2)
    - GMM:      forecast = mean + GMM-fit on residuals  (from exp 29)

  GARCH(1, 1) with skewed-t innovations:
    - point:    forecast = conditional mean
    - Gaussian: forecast = N(mean, sigma_t^2) using conditional variance
    - GMM:      forecast = mean + GMM-fit on standardized residuals  (from exp 23)

Output: results/35d_classical_3heads.json

Tasks: 2 classical models x 3 heads x 4 horizons x 5 folds = 120 cells.
Wall clock on Mac CPU: ~5-10 min.
"""
import sys
import os
import json
import time
import warnings
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
        folds.append({
            "fold": fold_idx,
            "X_train": X_all[:train_end_pair],
            "Y_train": Y_all[:train_end_pair],
            "X_test": X_all[train_end_pair:test_end_pair],
            "Y_test": Y_all[train_end_pair:test_end_pair],
        })
        fold_idx += 1
        train_end_series += step
    return folds


# ====================================================================
# ARIMA
# ====================================================================
def fit_arima_select(returns_train, p_range=range(3), q_range=range(3)):
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
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(returns_train, order=(p, 0, q),
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    fit = model.fit(method_kwargs={"warn_convergence": False})
    # Use get_forecast for proper h-step mean and std
    fc = fit.get_forecast(steps=horizon)
    mu = np.asarray(fc.predicted_mean)
    var = np.asarray(fc.var_pred_mean)
    # var may be a 2D array (h x 1) in newer statsmodels — flatten
    sigma_h = np.sqrt(np.asarray(var).flatten())
    sigma_resid = np.std(fit.resid)
    return mu, sigma_resid, fit.resid, sigma_h


def fit_gauss_on_residuals(residuals):
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    return mu, sigma


def fit_gmm_on_residuals(residuals, k=N_MIXTURES, n_init=5):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=n_init, random_state=0, max_iter=200,
                          reg_covar=1e-4)
    gmm.fit(residuals.reshape(-1, 1))
    return gmm


def sample_gmm(gmm, n_samples=N_SAMPLES):
    """Sample from a fitted 1D GMM."""
    n_components = gmm.n_components
    means = gmm.means_.flatten()
    sds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    rng = np.random.default_rng(0)
    comp_idx = rng.choice(n_components, size=n_samples, p=weights)
    samples = np.zeros(n_samples)
    for c in range(n_components):
        mask = comp_idx == c
        if mask.sum() > 0:
            samples[mask] = rng.normal(loc=means[c], scale=sds[c], size=mask.sum())
    return samples


def sample_gauss(mu, sigma, n_samples=N_SAMPLES):
    return np.random.default_rng(0).normal(loc=mu, scale=sigma, size=n_samples)


# ====================================================================
# GARCH(1, 1)
# ====================================================================
def fit_garch11(returns_train):
    from arch import arch_model
    am = arch_model(returns_train * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    return res


def garch11_forecast_dist(res, horizon):
    """Return conditional mean and per-step conditional std (in original scale)."""
    fc = res.forecast(horizon=horizon, reindex=False)
    # In arch 8.x, fc.mean and fc.variance may be DataFrames
    mean_raw = fc.mean
    var_raw = fc.variance
    if hasattr(mean_raw, "values"):
        mean_fc = np.asarray(mean_raw.values[-1, :]) / 100.0
    else:
        mean_fc = np.asarray(mean_raw)[-1, :] / 100.0
    if hasattr(var_raw, "values"):
        var_fc = np.asarray(var_raw.values[-1, :]) / (100.0 ** 2)
    else:
        var_fc = np.asarray(var_raw)[-1, :] / (100.0 ** 2)
    sigma_fc = np.sqrt(np.maximum(var_fc, 1e-12))
    # std_resid may be Series or ndarray depending on arch version
    sr = res.std_resid
    if hasattr(sr, "dropna"):
        sr = sr.dropna().values
    else:
        sr = np.asarray(sr)
    return mean_fc, sigma_fc, sr


# ====================================================================
# Metrics
# ====================================================================
def crps_empirical(samples, Y):
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    half = N_SAMPLES // 2
    term1 = np.mean(np.abs(samples - Y[None, :]), axis=0)
    term2 = np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0)
    return float((term1 - 0.5 * term2).mean())


def pinball(samples, Y, quantiles=QUANTILES):
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    H = Y.shape[0]
    loss = 0.0
    for q in quantiles:
        q_pred = np.quantile(samples, q, axis=0)
        diff = Y - q_pred
        loss += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return loss / len(quantiles)


def coverage_at_90(samples, Y):
    if Y.ndim == 0:
        Y = np.array([Y])
    if samples.ndim == 1:
        samples = samples[:, None]
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    return float(((Y >= lower) & (Y <= upper)).mean())


# ====================================================================
# Workers
# ====================================================================
def arima_run_one(args):
    """ARIMA(p,0,q) for one (horizon, fold). Returns dict of (head -> samples).

    For each head, we forecast the h-step distribution:
      point: N(mu_h, sigma_h^2)  using model-derived h-step std
      gauss: N(mu_h + mu_resid, sigma_h^2)  where mu_resid, sigma_resid are from training residuals
      gmm:   mu_h + GMM-on-residuals (scaled by sigma_h / sigma_resid for h-step)
    and compare to Y_test[h-1].
    """
    horizon, fold, Y_train, Y_test = args
    best = fit_arima_select(Y_train)
    if best is None:
        return None
    p, _, q, fit = best
    mu_fc, sigma_resid, residuals, sigma_h_arr = fit_arima_forecast(Y_train, horizon, p, q)
    # Use recent residuals for innovation distribution
    recent = residuals[-min(300, len(residuals)):]
    recent = np.asarray(recent).flatten()

    # h-step forecast: use the (h-1)-th step mean and std
    h_idx = min(horizon - 1, len(mu_fc) - 1, len(sigma_h_arr) - 1)
    mu_h = float(mu_fc[h_idx])
    sigma_h = float(sigma_h_arr[h_idx])

    if len(Y_test) <= h_idx:
        return None
    Y_h = float(Y_test[h_idx])

    out = {"horizon": horizon, "fold": fold, "arima_order": (p, q), "Y_test": Y_h}

    # Point: N(mu_h, sigma_h^2)
    out["point_samples"] = np.random.default_rng(0).normal(
        loc=mu_h, scale=sigma_h, size=N_SAMPLES
    )
    # Gauss: fit Gaussian on residuals
    g_mu, g_sigma = fit_gauss_on_residuals(recent)
    # The residual Gaussian has std = g_sigma, but the h-step forecast has
    # std = sigma_h. Scale the residual innovation by sigma_h / g_sigma.
    scale = sigma_h / max(g_sigma, 1e-6)
    out["gauss_samples"] = sample_gauss(mu_h + g_mu * scale, sigma_h)
    # GMM: similar — scale residual innovation to h-step
    gmm = fit_gmm_on_residuals(recent)
    gmm_resid_samples = sample_gmm(gmm) * scale
    out["gmm_samples"] = mu_h + gmm_resid_samples
    return out


def garch_run_one(args):
    horizon, fold, Y_train, Y_test = args
    try:
        res = fit_garch11(Y_train)
    except Exception:
        return None
    mean_fc, sigma_fc, std_resid = garch11_forecast_dist(res, horizon)
    recent = std_resid[-min(300, len(std_resid)):]

    # h-step forecast: use the (h-1)-th step
    h_idx = min(horizon - 1, len(mean_fc) - 1)
    mu_h = mean_fc[h_idx]
    sigma_h = sigma_fc[h_idx]
    if len(Y_test) <= h_idx:
        return None
    Y_h = float(Y_test[h_idx])

    out = {"horizon": horizon, "fold": fold, "Y_test": Y_h}

    # Point: N(mu_h, sigma_h^2)
    out["point_samples"] = np.random.default_rng(0).normal(
        loc=mu_h, scale=sigma_h, size=N_SAMPLES
    )
    # Gauss: fit Gaussian on standardized residuals
    g_mu, g_sigma = fit_gauss_on_residuals(recent)
    out["gauss_samples"] = mu_h + sigma_h * sample_gauss(g_mu, g_sigma)
    # GMM: fit GMM on standardized residuals
    gmm = fit_gmm_on_residuals(recent)
    out["gmm_samples"] = mu_h + sigma_h * sample_gmm(gmm)
    return out


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 72)
    print(" Experiment 35d: Classical baselines × 3 heads (ARIMA & GARCH)")
    print("=" * 72)

    series = load_series()
    print(f"[data] {len(series)} monthly log-returns")

    results = {
        "ARIMA": {f"h{h}": {} for h in HORIZONS},
        "GARCH": {f"h{h}": {} for h in HORIZONS},
    }

    for model_name, run_fn, num in [
        ("ARIMA", arima_run_one, 1),
        ("GARCH", garch_run_one, 2),
    ]:
        print(f"\n=== {model_name} ===")
        for horizon in HORIZONS:
            folds = build_folds(series, horizon)
            print(f"\n--- h={horizon} ---")
            tasks = [(horizon, fi, f["Y_train"][:, 0], f["Y_test"][:, 0]) for fi, f in enumerate(folds)]
            t0 = time.time()
            # Sequential (CPU-bound statsmodels, no benefit from parallel)
            cells = []
            for t in tasks:
                r = run_fn(t)
                if r is not None:
                    cells.append(r)
            print(f"  {len(cells)} cells in {time.time()-t0:.1f}s")

            for head in ["point", "gauss", "gmm"]:
                key = f"{head}_samples"
                crps_l, pin_l, cov_l = [], [], []
                for c in cells:
                    if key in c and "Y_test" in c:
                        s = c[key]
                        Y = np.array([c["Y_test"]])
                        crps_l.append(crps_empirical(s, Y))
                        pin_l.append(pinball(s, Y))
                        cov_l.append(coverage_at_90(s, Y))
                if not crps_l:
                    continue
                results[model_name][f"h{horizon}"][head] = {
                    "n_folds": len(crps_l),
                    "crps_mean": float(np.mean(crps_l)),
                    "crps_std": float(np.std(crps_l)),
                    "pinball_mean": float(np.mean(pin_l)),
                    "cov90_mean": float(np.mean(cov_l)),
                }
                print(f"  {head:<6} CRPS={np.mean(crps_l):.5f}  Pinball={np.mean(pin_l):.5f}  Cov90={np.mean(cov_l):.3f}")

    # Add baseline TimesNet_point CRPS for CRPS-Skill-Score computation
    exp22 = json.load(open(RESULTS_DIR / "22_sota_comparison.json"))
    baseline_crps = {f"h{h}": exp22["metrics"][f"h{h}"]["TimesNet_point"]["crps"] for h in HORIZONS}

    out = {
        "config": {
            "horizons": HORIZONS,
            "n_mixtures_gmm": N_MIXTURES,
            "n_samples": N_SAMPLES,
            "init_train_frac": INITIAL_TRAIN_FRAC,
            "test_frac": TEST_FRAC,
            "step_frac": STEP_FRAC,
        },
        "results": results,
        "baseline_crps_TimesNet_point": baseline_crps,
    }
    # Compute CRPS-SS
    for model in ["ARIMA", "GARCH"]:
        for h in HORIZONS:
            for head in ["point", "gauss", "gmm"]:
                if head in results[model][f"h{h}"]:
                    v = results[model][f"h{h}"][head]
                    v["crps_skill_score_vs_TimesNet_point"] = (
                        1.0 - v["crps_mean"] / baseline_crps[f"h{h}"]
                    )

    out_path = RESULTS_DIR / "35d_classical_3heads.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {out_path}")

    # Summary
    print("\n" + "=" * 72)
    print(" Summary — CRPS-Skill-Score vs TimesNet_point")
    print("=" * 72)
    for h in HORIZONS:
        print(f"\n--- h={h} ---")
        print(f"  {'Variant':<20} {'CRPS':>10} {'CRPS-SS':>10} {'Cov@90':>10}")
        for model in ["ARIMA", "GARCH"]:
            for head in ["point", "gauss", "gmm"]:
                v = results[model][f"h{h}"].get(head)
                if v is None:
                    continue
                crps = v["crps_mean"]
                ss = v["crps_skill_score_vs_TimesNet_point"] * 100
                cov = v["cov90_mean"]
                print(f"  {model}_{head:<10} {crps:>10.5f} {ss:>+10.2f}% {cov:>10.3f}")


if __name__ == "__main__":
    main()
