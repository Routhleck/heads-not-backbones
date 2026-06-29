"""
Phase 2.1 (extension): GARCH + 3 heads on DAILY S&P 500.

Following the pattern of 35d_classical_3heads.py (monthly) and 37a_garch_monthly.py,
this runs GARCH(1,1) with skewed-t innovations under three heads (point / Gaussian
/ GMM K=4) on the daily panel. Horizons {1, 5, 10, 20} days.

Outputs results/37d_garch_daily.json in the same shape as 35d_classical_3heads.json
so it can be merged into the cross-asset table.
"""
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
np.random.seed(0)

DAILY_HORIZONS = [1, 5, 10, 20]
N_SAMPLES = 500
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
N_MIXTURES = 4
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_folds(series, horizon):
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    folds = []
    fold_idx = 0
    train_end = init_train
    while train_end + test_window + horizon <= n:
        test_end = train_end + test_window
        train = series[:train_end]
        test = series[train_end:test_end]
        folds.append({"fold": fold_idx, "train": train, "test": test})
        fold_idx += 1
        train_end += step
    return folds


def fit_garch_point(tr, te, horizon):
    """GARCH(1,1) skewed-t, point prediction = median of simulated paths.
    Rescale to percent internally (matches 37a pattern).
    """
    tr_pct = tr * 100
    te_pct = te * 100
    try:
        am = arch_model(tr_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        return None, None
    fc = res.forecast(horizon=horizon, reindex=False, simulations=2000, method="simulation")
    paths = fc.simulations.values[0]  # (2000, horizon) in percent
    cond_vol = np.sqrt(fc.variance.values[0, :])
    cond_mean = res.params.get("mu", 0.0)
    N = len(te)
    samples_pct = np.zeros((N_SAMPLES, N, horizon))
    std_resid = np.asarray(res.std_resid)
    std_resid = std_resid[~np.isnan(std_resid)]
    for h in range(horizon):
        innov_sample = np.random.choice(std_resid, size=(N_SAMPLES, N), replace=True)
        samples_pct[:, :, h] = cond_mean + cond_vol[h] * innov_sample
    samples = samples_pct / 100  # back to decimal
    point_pred = np.tile(np.median(paths, axis=0) / 100, (N, 1))
    return samples, point_pred


def fit_garch_gauss(tr, te, horizon):
    """GARCH(1,1) normal innovation, point prediction = conditional mean."""
    tr_pct = tr * 100
    try:
        am = arch_model(tr_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        return None, None
    fc = res.forecast(horizon=horizon, reindex=False, simulations=2000, method="simulation")
    paths = fc.simulations.values[0]
    cond_vol = np.sqrt(fc.variance.values[0, :])
    cond_mean = res.params.get("mu", 0.0)
    N = len(te)
    samples_pct = np.zeros((N_SAMPLES, N, horizon))
    for h in range(horizon):
        samples_pct[:, :, h] = cond_mean + cond_vol[h] * np.random.normal(size=(N_SAMPLES, N))
    samples = samples_pct / 100
    point_pred = np.tile(np.median(paths, axis=0) / 100, (N, 1))
    return samples, point_pred


def fit_garch_gmm(tr, te, horizon, n_mixtures=N_MIXTURES, seed=0):
    """GARCH(1,1) skewed-t + GMM K=4 on standardized residuals."""
    tr_pct = tr * 100
    try:
        am = arch_model(tr_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        return None, None
    std_resid = np.asarray(res.std_resid)
    std_resid = std_resid[~np.isnan(std_resid)]
    if len(std_resid) < 50:
        return None, None
    gmm = GaussianMixture(n_components=n_mixtures, random_state=seed, max_iter=200)
    gmm.fit(std_resid.reshape(-1, 1))
    fc = res.forecast(horizon=horizon, reindex=False, simulations=2000, method="simulation")
    cond_vol = np.sqrt(fc.variance.values[0, :])
    cond_mean = res.params.get("mu", 0.0)
    N = len(te)
    samples_pct = np.zeros((N_SAMPLES, N, horizon))
    for h in range(horizon):
        gmm_samples, _ = gmm.sample(N_SAMPLES * N)
        gmm_samples = gmm_samples[:N_SAMPLES * N, 0].reshape(N_SAMPLES, N)
        samples_pct[:, :, h] = cond_mean + cond_vol[h] * gmm_samples
    samples = samples_pct / 100
    point_pred = np.tile(np.zeros(horizon), (N, 1))
    return samples, point_pred


# Metrics
def crps_from_samples(samples, truth):
    half = samples.shape[0] // 2
    crps = float(
        np.mean(np.abs(samples - truth[None]), axis=0).mean() -
        0.5 * np.mean(np.abs(samples[:half] - samples[half:half * 2]), axis=0).mean()
    )
    return crps


def pinball_loss_array(truth, q_pred, q_level):
    diff = truth - q_pred
    return float(np.maximum(q_level * diff, (q_level - 1) * diff).mean())


def main():
    panel = pd.read_csv(ROOT / "data" / "raw" / "panel_daily.csv",
                        index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float64)
    print(f"[data] daily SP500_Return n={len(series)}")

    methods = {
        "point": fit_garch_point,
        "gauss": fit_garch_gauss,
        "gmm": fit_garch_gmm,
    }

    # Baseline CRPS = TimesNet_point from 36e daily
    with open(RESULTS_DIR / "36e_daily_SP500_Return.json") as f:
        daily = json.load(f)
    baseline_crps = {
        f"h{h}": daily["metrics"][f"h{h}"]["TimesNet_point"]["crps"]
        for h in DAILY_HORIZONS
    }
    print(f"[baseline] TimesNet_point CRPS = {baseline_crps}")

    results_by_method = {name: {f"h{h}": {"crps_per_fold": []} for h in DAILY_HORIZONS}
                         for name in methods}

    t0 = time.time()
    for method_name, fit_fn in methods.items():
        print(f"\n[{method_name}]")
        for h in DAILY_HORIZONS:
            folds = build_folds(series, h)
            for fold in folds:
                samples, point_pred = fit_fn(fold["train"], fold["test"], h)
                if samples is None:
                    continue
                truth = fold["test"]
                if truth.ndim == 1:
                    truth = truth[:, None]
                crps = crps_from_samples(samples, truth)
                # Pinball @ 0.05 (1-step ahead, used for daily VaR)
                q05_h1 = np.quantile(samples[:, :, 0], 0.05, axis=0)  # (N,)
                truth_h1 = truth[:, 0]  # (N,)
                pinball_05 = pinball_loss_array(truth_h1, q05_h1, 0.05)
                # Coverage @ 90% (1-step)
                q_lo = np.quantile(samples[:, :, 0], 0.05, axis=0)
                q_hi = np.quantile(samples[:, :, 0], 0.95, axis=0)
                cov = float(np.mean((truth_h1 >= q_lo) & (truth_h1 <= q_hi)))
                results_by_method[method_name][f"h{h}"]["crps_per_fold"].append({
                    "fold": fold["fold"],
                    "crps": crps,
                    "pinball_05": pinball_05,
                    "cov90": cov,
                })
        agg = {}
        for h in DAILY_HORIZONS:
            folds_data = results_by_method[method_name][f"h{h}"]["crps_per_fold"]
            if not folds_data:
                continue
            mean_crps = float(np.mean([f["crps"] for f in folds_data]))
            ss = (1 - mean_crps / baseline_crps[f"h{h}"]) * 100
            agg[f"h{h}"] = {
                "n_folds": len(folds_data),
                "crps_mean": mean_crps,
                "crps_std": float(np.std([f["crps"] for f in folds_data])),
                "pinball_05_mean": float(np.mean([f["pinball_05"] for f in folds_data])),
                "cov90_mean": float(np.mean([f["cov90"] for f in folds_data])),
                "crps_per_fold": [f["crps"] for f in folds_data],
                "crps_skill_score_vs_TimesNet_point": ss / 100,
            }
            print(f"  h={h}: CRPS={mean_crps:.5f}, CRPS-SS={ss:+.2f}%, "
                  f"pin05={agg[f'h{h}']['pinball_05_mean']:.5f}, cov90={agg[f'h{h}']['cov90_mean']:.2%}")
        results_by_method[method_name] = agg

    out = {
        "config": {
            "data": "daily SP500_Return (2014-2024)",
            "horizons": DAILY_HORIZONS,
            "n_simulation_paths": 2000,
            "n_samples_per_cell": N_SAMPLES,
            "methods": list(methods.keys()),
        },
        "baseline_crps_TimesNet_point": baseline_crps,
        "results": {"GARCH": results_by_method},
    }
    out_path = RESULTS_DIR / "37d_garch_daily.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")
    print(f"[elapsed] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()