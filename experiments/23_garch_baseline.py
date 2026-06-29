"""
Experiment 23: Classical baseline — GARCH(1,1) + GMM density head.

Why this experiment: The user-feedback review for ICAIF 2026 flagged
"why not compare to GARCH" as the most likely reviewer question. We
fit a GARCH(1,1,1) model on each training fold, then a K=4 Gaussian
mixture on the standardized residuals. The forecast density at each
test point is then a mixture of N(mu_t, sigma_t^2 / alpha_k^2) — the
same GMM-head architecture as TimesNet+GMM but with a classical
GARCH backbone for the time-varying mean/volatility.

Walk-forward protocol is **identical** to exp 19 / 22:
- 5 anchored expanding-window folds
- h ∈ {1, 3, 6, 12} months
- INITIAL_TRAIN_FRAC = 0.50, TEST_FRAC = 0.07, STEP_FRAC = 0.10
- Series: S&P 500 monthly log-returns from data/raw/panel_monthly.csv

We also evaluate Historical Simulation as a non-parametric baseline
(rolling-window empirical CDF).

Output: results/23_garch_baseline.json with same schema as 22.
"""
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Mirror exp 19 / 22 constants exactly so the comparison is apples-to-apples
INITIAL_TRAIN_FRAC = 0.50
TEST_FRAC = 0.07
STEP_FRAC = 0.10
HORIZONS = [1, 3, 6, 12]
N_MIXTURES = 4
HS_WINDOW = 120  # historical simulation: rolling 10y (120 months) window

# Suppress arch library warnings (GARCH convergence on small panels)
warnings.filterwarnings("ignore")
from arch import arch_model  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"


# ============================================================
# Walk-forward folds (identical to exp 19)
# ============================================================
def build_fold_indices(n: int, init_train: int, test_window: int, step: int):
    """Return list of (train_idx_array, test_idx_array, fold_idx)."""
    folds = []
    fold_idx = 0
    train_end = init_train
    while train_end + test_window <= n:
        test_end = train_end + test_window
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        folds.append((train_idx, test_idx, fold_idx))
        fold_idx += 1
        train_end += step
    return folds


# ============================================================
# Density representation
# ============================================================
class GMMDensity:
    """Mixture of Gaussians over a 1-D return value."""
    def __init__(self, pi, mu, sigma):
        # pi, mu, sigma: shape (K,)
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def sample(self, n, rng):
        comp = rng.choice(len(self.pi), size=n, p=self.pi)
        return self.mu[comp] + self.sigma[comp] * rng.standard_normal(n)

    def mean(self):
        return float(np.sum(self.pi * self.mu))

    def quantile(self, q):
        # Monte-Carlo quantile (since mixture is not analytically invertible)
        rng = np.random.default_rng(2024)
        s = self.sample(20000, rng)
        return float(np.quantile(s, q))


# ============================================================
# GARCH(1,1) + GMM
# ============================================================
def fit_garch_gmm(train_returns: np.ndarray):
    """Fit GARCH(1,1) and K=4 GMM on standardized residuals.

    Returns (arch_model, fit_result, gmm_density_on_z) where
    gmm_density_on_z models the standardized residuals
    z_t = (r_t - mu_t) / sigma_t.
    """
    # 1. GARCH(1,1) with constant mean
    am = arch_model(train_returns * 100,  # arch works better with % returns
                    mean="Constant", vol="GARCH", p=1, q=1,
                    dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    # 2. Standardized residuals
    z = np.asarray(res.std_resid).astype(np.float64)
    z = z[~np.isnan(z)]
    z = z[np.isfinite(z)]
    # 3. K=4 GMM on standardized residuals
    if len(z) < 50:
        return am, res, GMMDensity(np.array([1.0]), np.array([0.0]), np.array([1.0]))
    gmm = GaussianMixture(n_components=N_MIXTURES, covariance_type="full",
                          random_state=0, n_init=3, max_iter=200)
    gmm.fit(z.reshape(-1, 1))
    pi = gmm.weights_.astype(np.float64)
    mu = gmm.means_.reshape(-1).astype(np.float64)
    sigma = np.sqrt(gmm.covariances_.reshape(-1)).astype(np.float64)
    return am, res, GMMDensity(pi, mu, sigma)


def forecast_garch_density(am, model_fit, gmm_z, train_returns, test_returns, horizon: int):
    """Filter GARCH through the test set with training parameters frozen,
    then for each test point t produce the 1-step-ahead density.

    Returns:
      mu_per_step: shape (n_test, horizon) — point forecast at each (t, h)
      sigma_per_step: shape (n_test, horizon) — filtered sigma at each (t, h)
    """
    # 1) Get the GARCH parameters from the training fit
    params = model_fit.params
    # params: [mu, omega, alpha, beta] for Constant-mean GARCH(1,1)
    mu_const = float(params["mu"])
    omega = float(params["omega"])
    alpha_g = float(params["alpha[1]"])
    beta_g = float(params["beta[1]"])

    # 2) Get the LAST filtered variance from training (σ²_{T|T-1})
    # Use the fitted model's conditional_volatility
    train_vol = np.asarray(model_fit.conditional_volatility)  # in % scale
    sigma2_last = float(train_vol[-1] ** 2)
    eps_last = float(np.asarray(model_fit.resid)[-1])

    # 3) Run GARCH recursion through the test set
    full_returns_pct = np.concatenate([train_returns, test_returns]) * 100  # % scale
    n_test = len(test_returns)
    test_vol_pct = np.zeros(n_test)  # σ_{t+1|t} in % scale
    test_mean_pct = np.zeros(n_test)
    # h-step forecast (iterated from current state)
    sigma2 = sigma2_last
    eps = eps_last
    mu_val = mu_const  # constant mean in % scale
    for i in range(n_test):
        # store 1-step-ahead forecast at t = i+1
        test_mean_pct[i] = mu_val
        test_vol_pct[i] = np.sqrt(sigma2)
        # update state with observed test return
        r_obs = full_returns_pct[len(train_returns) + i]
        eps = r_obs - mu_val
        sigma2 = omega + alpha_g * eps ** 2 + beta_g * sigma2

    # 4) Build per-(t, h) forecasts. For h > 1 we use the h-step iterated forecast
    # starting at test point t. The simplest approach: at test point t (with index i),
    # the h-step ahead forecast uses sigma²_t|t-1 from the recursion and the standard
    # GARCH h-step variance formula.
    # mu_per_step[i, h] = mu (constant)
    # var_per_step[i, h] = sigma²_{t+h-1|t-1} (for direct h-step forecast at t)
    # For simplicity we use the "1-step-ahead density applied with the sigma
    # at t+h-1" — this is a reasonable iterated approximation.
    mu_per_step = np.full((n_test, horizon), mu_const / 100.0)  # convert to decimal
    sigma_per_step = np.zeros((n_test, horizon))
    for i in range(n_test):
        for h in range(horizon):
            # Use the sigma at index i+h-1 (clamped to n_test-1 for last horizon)
            idx = min(i + h - 1, n_test - 1) if h > 1 else i
            sigma_per_step[i, h] = test_vol_pct[idx] / 100.0
    return mu_per_step, sigma_per_step


def build_density_lists(mu_per_step, sigma_per_step, gmm_z, horizon):
    """Build densities_by_step: list of H lists, each of n_test GMMDensity objects."""
    n_test = mu_per_step.shape[0]
    out = []
    for h in range(horizon):
        d_list = []
        for i in range(n_test):
            m = mu_per_step[i, h]
            s = sigma_per_step[i, h]
            sigma_k = s * gmm_z.sigma
            d_list.append(GMMDensity(gmm_z.pi.copy(), np.full_like(gmm_z.sigma, m), sigma_k))
        out.append(d_list)
    return out


# ============================================================
# Historical Simulation (rolling-window empirical CDF)
# ============================================================
def hs_density(train_returns: np.ndarray, window: int = HS_WINDOW):
    """Use last `window` training returns as empirical distribution.
    Returns a frozen empirical density."""
    sample = train_returns[-window:]
    class Empirical:
        def __init__(self, x):
            self.x = np.sort(x)
        def sample(self, n, rng):
            idx = rng.integers(0, len(self.x), size=n)
            return self.x[idx]
        def mean(self):
            return float(self.x.mean())
        def quantile(self, q):
            return float(np.quantile(self.x, q))
    return Empirical(sample)


# ============================================================
# Metrics (mirror exp 19 / 21 evaluation)
# ============================================================
def crps_emp(truth, samples):
    """Empirical CRPS: E|S-y| - 0.5 E|S-S'|"""
    n = len(samples)
    term1 = np.mean(np.abs(samples - truth))
    # pairwise |S - S'|
    if n > 200:
        # sub-sample for efficiency
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        s_sub = samples[idx]
        term2 = np.mean(np.abs(s_sub[:, None] - s_sub[None, :]))
    else:
        term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
    return term1 - 0.5 * term2


def evaluate_horizon(densities_by_step, truth_steps, rng):
    """Evaluate MAE, CRPS, Pinball@0.05/0.5/0.95, Coverage@0.90.

    densities_by_step: list of length H; each is a list of length n_test of GMMDensity
    truth_steps:        shape (n_test, H)
    """
    n_test, H = truth_steps.shape
    assert len(densities_by_step) == H, f"got {len(densities_by_step)} densities, truth has H={H}"

    mae_sum = 0.0
    crps_sum = 0.0
    pinball_sums = {0.05: 0.0, 0.5: 0.0, 0.95: 0.0}
    coverage_hits = 0
    total = 0

    for i in range(n_test):
        for h in range(H):
            d = densities_by_step[h][i]
            y = truth_steps[i, h]
            mae_sum += abs(d.mean() - y)
            samples = d.sample(500, rng)
            crps_sum += crps_emp(y, samples)
            for q in (0.05, 0.5, 0.95):
                qh = d.quantile(q)
                diff = y - qh
                pinball_sums[q] += max(q * diff, (q - 1) * diff)
            lo, hi = d.quantile(0.05), d.quantile(0.95)
            if lo <= y <= hi:
                coverage_hits += 1
            total += 1

    return {
        "mae": mae_sum / total,
        "crps": crps_sum / total,
        "pinball_0.05": pinball_sums[0.05] / total,
        "pinball_0.50": pinball_sums[0.5] / total,
        "pinball_0.95": pinball_sums[0.95] / total,
        "coverage_0.90": coverage_hits / total,
    }


def evaluate_single_horizon(density_list, truth_vec, rng):
    """Evaluate one horizon. density_list: list of n GMMDensity. truth_vec: (n,)."""
    assert len(density_list) == len(truth_vec), \
        f"density_list len {len(density_list)} != truth_vec len {len(truth_vec)}"
    n = len(density_list)
    mae_sum = 0.0
    crps_sum = 0.0
    pinball_sums = {0.05: 0.0, 0.5: 0.0, 0.95: 0.0}
    coverage_hits = 0
    for i, (d, y) in enumerate(zip(density_list, truth_vec)):
        mae_sum += abs(d.mean() - y)
        samples = d.sample(500, rng)
        crps_sum += crps_emp(y, samples)
        for q in (0.05, 0.5, 0.95):
            qh = d.quantile(q)
            diff = y - qh
            pinball_sums[q] += max(q * diff, (q - 1) * diff)
        lo, hi = d.quantile(0.05), d.quantile(0.95)
        if lo <= y <= hi:
            coverage_hits += 1
    return {
        "mae": mae_sum / n,
        "crps": crps_sum / n,
        "pinball_0.05": pinball_sums[0.05] / n,
        "pinball_0.50": pinball_sums[0.5] / n,
        "pinball_0.95": pinball_sums[0.95] / n,
        "coverage_0.90": coverage_hits / n,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print(" Experiment 23: GARCH(1,1) + GMM and Historical Simulation baselines")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH)
    series = panel["SP500_Return"].dropna().values.astype(np.float64)
    print(f"[data] n={len(series)}")
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)
    folds = build_fold_indices(n, init_train, test_window, step)
    print(f"[folds] {len(folds)} folds, init_train={init_train}, test_window={test_window}, step={step}")

    rng = np.random.default_rng(42)

    results = {"garch_gmm": {}, "historical_sim": {}}
    for model_name in ("garch_gmm", "historical_sim"):
        for h in HORIZONS:
            results[model_name][f"h{h}"] = {}

    for fold_idx, (train_idx, test_idx, fid) in enumerate(folds):
        train_r = series[train_idx]
        test_r = series[test_idx]
        print(f"\n[fold {fid}] train={len(train_r)}, test={len(test_r)}")

        # ---- GARCH(1,1) + GMM ----
        print(f"  [garch_gmm] fitting on {len(train_r)} pts...")
        try:
            am, res, gmm_z = fit_garch_gmm(train_r)
            mu_per_step, sigma_per_step = forecast_garch_density(
                am, res, gmm_z, train_r, test_r, max(HORIZONS)
            )
            n_test = len(test_r)
            truth_steps = np.zeros((n_test, max(HORIZONS)))
            for i in range(n_test):
                for h in range(max(HORIZONS)):
                    if i + h < n_test:
                        truth_steps[i, h] = test_r[i + h]
            n_valid = n_test - (max(HORIZONS) - 1)
            if n_valid <= 0:
                print(f"  [garch_gmm] not enough test points, skip")
                continue
            truth_steps_v = truth_steps[:n_valid]
            mu_per_step_v = mu_per_step[:n_valid]
            sigma_per_step_v = sigma_per_step[:n_valid]
            dens_by_h = build_density_lists(mu_per_step_v, sigma_per_step_v, gmm_z, max(HORIZONS))
            m = evaluate_horizon(dens_by_h, truth_steps_v, rng)
            print(f"  [garch_gmm] MAE={m['mae']:.5f} CRPS={m['crps']:.5f} "
                  f"Pinball@0.05={m['pinball_0.05']:.5f} "
                  f"Pinball@0.95={m['pinball_0.95']:.5f} "
                  f"Coverage@0.90={m['coverage_0.90']:.3f}")
            for h in HORIZONS:
                m_h = evaluate_single_horizon(
                    dens_by_h[h-1],
                    truth_steps_v[:, h-1],
                    rng,
                )
                for key, val in m_h.items():
                    results["garch_gmm"][f"h{h}"].setdefault(f"per_fold_{key}", []).append(val)
        except Exception as e:
            print(f"  [garch_gmm] FAILED on fold {fid}: {e}")

        # ---- Historical Simulation ----
        print(f"  [historical_sim] fitting on last {HS_WINDOW} train pts...")
        hs = hs_density(train_r, HS_WINDOW)
        n_test = len(test_r)
        truth_steps = np.zeros((n_test, max(HORIZONS)))
        for i in range(n_test):
            for h in range(max(HORIZONS)):
                if i + h < n_test:
                    truth_steps[i, h] = test_r[i + h]
        n_valid = n_test - (max(HORIZONS) - 1)
        if n_valid > 0:
            truth_steps_v = truth_steps[:n_valid]
            dens_by_h = []
            for h in range(max(HORIZONS)):
                dens_by_h.append([hs] * n_valid)
            m = evaluate_horizon(dens_by_h, truth_steps_v, rng)
            print(f"  [historical_sim] MAE={m['mae']:.5f} CRPS={m['crps']:.5f} "
                  f"Coverage@0.90={m['coverage_0.90']:.3f}")
            for h in HORIZONS:
                m_h = evaluate_single_horizon(
                    dens_by_h[h-1],
                    truth_steps_v[:, h-1],
                    rng,
                )
                for key, val in m_h.items():
                    results["historical_sim"][f"h{h}"].setdefault(f"per_fold_{key}", []).append(val)

    # ---- Aggregate (mean over folds) ----
    METRIC_KEYS = ("mae", "crps", "pinball_0.05", "pinball_0.50",
                   "pinball_0.95", "coverage_0.90")
    out = {"config": {
        "INITIAL_TRAIN_FRAC": INITIAL_TRAIN_FRAC,
        "TEST_FRAC": TEST_FRAC,
        "STEP_FRAC": STEP_FRAC,
        "HORIZONS": HORIZONS,
        "N_MIXTURES": N_MIXTURES,
        "HS_WINDOW": HS_WINDOW,
        "n_series": int(n),
    }, "metrics": {}}
    for model_name in ("garch_gmm", "historical_sim"):
        out["metrics"][model_name] = {}
        for h in HORIZONS:
            entry = {}
            for key in METRIC_KEYS:
                folds_vals = results[model_name][f"h{h}"].get(f"per_fold_{key}", [])
                if folds_vals:
                    entry[f"mean_{key}"] = float(np.mean(folds_vals))
                    entry[f"std_{key}"] = float(np.std(folds_vals))
            entry["n_folds"] = len(results[model_name][f"h{h}"].get("per_fold_crps", []))
            if entry["n_folds"] > 0:
                out["metrics"][model_name][f"h{h}"] = entry
    out_path = RESULTS_DIR / "23_garch_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()