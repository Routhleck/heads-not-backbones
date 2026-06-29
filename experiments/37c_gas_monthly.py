"""Phase 2.3: GAS (Generalized AutoRegressive Score) baseline.

GAS (Creal et al. 2013, Harvey 2013) is a state-space model for time-varying
parameters updated by the score of the conditional density. For VaR
forecasting, the simplest GAS specification is:

    sigma_t^2 = omega + A * sigma_{t-1}^2 + B * s_{t-1}
    s_t = score of log-likelihood at t wrt sigma_t^2

For Gaussian innovations:
    y_t = sigma_t * eps_t,  eps_t ~ N(0, 1)
    log L = -0.5 * log(sigma_t^2) - 0.5 * (y_t / sigma_t)^2
    d log L / d sigma_t^2 = -0.5 / sigma_t^2 + 0.5 * y_t^2 / sigma_t^4

For skew-t innovations (GAS with skew-t density), the score is more complex.

We implement a simple Gaussian-GAS for the volatility, then build
h-step forecasts by iterating sigma forward and computing predictive
quantiles at each step.

Reference:
  Creal, D., Koopman, S.J., Lucas, A., 2013. Generalized Autoregressive
    Score Models with Applications. Journal of Applied Econometrics 28(5), 777-795.
"""
import sys
import os
import json
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RES.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
HORIZONS = [1, 3, 6, 12]
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
N_SAMPLES = 500


def pinball_loss_array(truth, q_pred, q_level):
    diff = truth - q_pred
    return float(np.maximum(q_level * diff, (q_level - 1) * diff).mean())


def fit_gas_gaussian(y):
    """Fit Gaussian-GAS(1,1) on y (centered to mean 0)."""
    y_centered = y - y.mean()
    n = len(y_centered)

    def neg_log_lik(params):
        omega, A, B = params
        if A < 0 or A >= 0.999 or omega <= 0:
            return 1e10
        sigma2 = np.zeros(n)
        # Initialize sigma^2 with unconditional
        sigma2[0] = np.var(y_centered)
        s = np.zeros(n)  # score
        for t in range(1, n):
            # Score of log L wrt sigma^2:
            # log L_t = -0.5*log(sigma^2_t) - 0.5*(y_t/sigma_t)^2
            # d/d sigma^2 = -0.5/sigma^2 + 0.5*y^2/sigma^4
            # For sigma2=1 and y=eps: s = -0.5 + 0.5*eps^2
            eps = y_centered[t-1] / np.sqrt(sigma2[t-1])
            s_t = -0.5 / sigma2[t-1] + 0.5 * y_centered[t-1]**2 / sigma2[t-1]**2
            # Multiply by scaling s_t -> s_t/scale (we use raw, parameters absorb it)
            sigma2[t] = omega + A * sigma2[t-1] + B * s_t
            if sigma2[t] <= 0:
                return 1e10
        # Likelihood
        ll = -0.5 * np.sum(np.log(sigma2) + y_centered**2 / sigma2)
        return -ll

    best = None
    for x0 in [[0.001, 0.9, 0.05], [0.01, 0.85, 0.1], [0.0001, 0.95, 0.02]]:
        try:
            r = minimize(neg_log_lik, x0, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    if best is None:
        return None
    omega, A, B = best.x
    return omega, A, B, best.fun


def gas_forecast_quantiles(y_train, q_levels, horizon, n_test):
    """Fit GAS and produce h-step-ahead quantile predictions on test set."""
    omega, A, B, _ = fit_gas_gaussian(y_train)
    if omega is None:
        return None
    y = np.asarray(y_train, dtype=np.float64)
    y_mean = y.mean()
    y_centered = y - y_mean
    n = len(y)

    # Reconstruct sigma^2 path on training data
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(y_centered)
    for t in range(1, n):
        eps = y_centered[t-1] / np.sqrt(sigma2[t-1])
        s_t = -0.5 / sigma2[t-1] + 0.5 * y_centered[t-1]**2 / sigma2[t-1]**2
        sigma2[t] = omega + A * sigma2[t-1] + B * s_t

    # For h-step-ahead, use last sigma^2 (assumes score=0 in future)
    # The unconditional expectation of score for Gaussian is 0
    # so E[sigma^2_{t+h}] = omega / (1-A) + A^h * (sigma^2_t - omega/(1-A))
    sigma2_last = sigma2[-1]
    uncond = omega / (1 - A) if A < 1 else sigma2_last
    sigma2_forecast = uncond + (A**horizon) * (sigma2_last - uncond)
    sigma_forecast = np.sqrt(sigma2_forecast)

    # Quantile predictions: y_mean + sigma * z_q
    from scipy.stats import norm
    quantile_preds = {q: np.full(n_test, y_mean + sigma_forecast * norm.ppf(q)) for q in q_levels}
    return quantile_preds


def main():
    print("=" * 70)
    print(" Phase 2.3: GAS (Generalized AutoRegressive Score) baseline")
    print("=" * 70)
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values
    print(f"[data] monthly SP500_Return n={len(series)} (fractional)")

    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    folds = []
    train_end = init_train
    fi = 0
    while train_end + test_window <= n:
        test_end = train_end + test_window
        folds.append({
            "fold": fi,
            "train": series[:train_end],
            "test": series[train_end:test_end],
        })
        train_end += step
        fi += 1
    print(f"[folds] n_folds={len(folds)}")

    with open(RES / "22_sota_comparison.json") as f:
        sota = json.load(f)
    baseline_crps = {f"h{h}": sota["metrics"][f"h{h}"]["TimesNet_point"]["crps"] for h in HORIZONS}

    results = {}
    for h in HORIZONS:
        hkey = f"h{h}"
        results[hkey] = {"crps_list": [], "pinball_list": {q: [] for q in QUANTILES}}
        for fold in folds:
            t0 = time.time()
            y_train = fold["train"]
            y_test = fold["test"]
            n_test = len(y_test)

            KEY_QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]
            try:
                all_quantiles_pred = gas_forecast_quantiles(y_train, KEY_QUANTILES, h, n_test)
            except Exception as e:
                print(f"  h={h} fold={fold['fold']} fit failed: {e}")
                continue
            if all_quantiles_pred is None:
                print(f"  h={h} fold={fold['fold']} fit returned None")
                continue
            # Check for NaN
            if any(np.any(np.isnan(all_quantiles_pred[kq])) for kq in KEY_QUANTILES):
                print(f"  h={h} fold={fold['fold']} fit produced NaN")
                continue

            # Interpolate remaining quantiles
            sorted_kq = sorted(KEY_QUANTILES)
            for q in QUANTILES:
                if q not in all_quantiles_pred:
                    pred_q = np.zeros(n_test)
                    for ti in range(n_test):
                        vals = np.array([all_quantiles_pred[kq][ti] for kq in sorted_kq])
                        pred_q[ti] = np.interp(q, sorted_kq, vals)
                    all_quantiles_pred[q] = pred_q

            for q in QUANTILES:
                pinball = pinball_loss_array(y_test, all_quantiles_pred[q], q)
                results[hkey]["pinball_list"][q].append(pinball)

            # CRPS via interpolation
            sorted_q = sorted(QUANTILES)
            samples_pred = np.zeros((N_SAMPLES, n_test))
            rng = np.random.default_rng(0)
            u_samples = rng.uniform(0, 1, N_SAMPLES)
            for j, u in enumerate(u_samples):
                for ti in range(n_test):
                    quantiles_arr = np.array([all_quantiles_pred[q][ti] for q in sorted_q])
                    samples_pred[j, ti] = np.interp(u, sorted_q, quantiles_arr)

            half = N_SAMPLES // 2
            crps = float(np.mean(np.abs(samples_pred - y_test[None]), axis=0).mean() -
                        0.5 * np.mean(np.abs(samples_pred[:half] - samples_pred[half:half*2]), axis=0).mean())
            results[hkey]["crps_list"].append(crps)
            print(f"  h={h} fold={fold['fold']} CRPS={crps:.5f} pinball@0.05={pinball:.5f} ({time.time()-t0:.1f}s)")

        # Aggregate over only successful folds
        if results[hkey]["crps_list"]:
            results[hkey]["crps_mean"] = float(np.mean(results[hkey]["crps_list"]))
            results[hkey]["crps_std"] = float(np.std(results[hkey]["crps_list"]))
            results[hkey]["n_successful_folds"] = len(results[hkey]["crps_list"])
            results[hkey]["crps_skill_score_vs_TimesNet_point_pct"] = \
                (baseline_crps[hkey] - results[hkey]["crps_mean"]) / baseline_crps[hkey] * 100
            results[hkey]["pinball_mean"] = {q: float(np.mean(results[hkey]["pinball_list"][q])) for q in QUANTILES if results[hkey]["pinball_list"][q]}

    out = {
        "config": {
            "data": "monthly S&P 500 (Shiller)",
            "n_total": n,
            "n_folds": len(folds),
            "quantiles": QUANTILES,
            "n_samples": N_SAMPLES,
            "method": "GAS(1,1) Gaussian (Creal-Koopman-Lucas 2013), h-step = unconditional mean-reverting forecast",
        },
        "baseline_crps_TimesNet_point": baseline_crps,
        "results": results,
    }
    with open(RES / "37c_gas_monthly.json", "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 70)
    print(" GAS summary (CRPS-SS vs TimesNet_point, %)")
    print("=" * 70)
    print(f"  h=1     h=3     h=6     h=12")
    cells = [f"{results[f'h{h}'].get('crps_skill_score_vs_TimesNet_point_pct', 0):+.2f}%" for h in HORIZONS]
    print("  " + "  ".join(f"{c:>7s}" for c in cells))
    print()
    print("Pinball at q=0.05 (5% VaR proper score):")
    for h in HORIZONS:
        v = results[f"h{h}"].get("pinball_mean", {}).get(0.05, None)
        if v is not None:
            print(f"  h={h}: {v:.5f}")
    print()
    print("Pinball at q=0.01 (1% VaR proper score):")
    for h in HORIZONS:
        v = results[f"h{h}"].get("pinball_mean", {}).get(0.01, None)
        if v is not None:
            print(f"  h={h}: {v:.5f}")
    print(f"\nSaved {RES / '37c_gas_monthly.json'}")


if __name__ == "__main__":
    main()
