"""
Experiment 42: VaR backtest + FRTB capital quantification on monthly S&P 500.

Uses 22_sota_comparison's train_and_save_predictions() to get per-fold
samples for TimesNet_point, TimesNet_gmm, NBEATS_gmm, plus an inline GARCH+GMM
function. Then computes:
  - VaR(99%), VaR(95%), ES(97.5%) per fold
  - Kupiec unconditional-coverage test
  - Christoffersen independence test
  - FRTB-style capital ($100 notional × ES)

Output: results/42_var_backtest.json
"""
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
np.random.seed(0)

# Re-use 22's training infrastructure
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp_22", ROOT / "experiments" / "22_sota_comparison.py"
)
exp_22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_22)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1]  # 1-step VaR backtest
N_SAMPLES = 500
INITIAL_TRAIN_FRAC = exp_22.INITIAL_TRAIN_FRAC
TEST_FRAC = exp_22.TEST_FRAC
STEP_FRAC = exp_22.STEP_FRAC
SEQ_LEN = exp_22.SEQ_LEN
SEED = 0


# ============= VaR + ES from samples =============
def var_es_from_samples(samples, truth, levels=(0.01, 0.05, 0.025)):
    """samples (N_SAMPLES, N, H), truth (N, H). Uses 1-step (h=0).

    VaR at confidence (1-q): the loss L = -r such that P(L > VaR) = q.
      => VaR_q = -quantile(returns, q)  [positive loss number]
      => violation: realized r < -VaR_q = quantile(returns, q)
    ES at confidence (1-q): mean of outcomes worse than VaR.
    """
    out = {}
    s_h1 = samples[:, :, 0]
    t_h1 = truth[:, 0]
    for q in levels:
        var_thresh = np.quantile(s_h1, q, axis=0)  # the q-quantile of returns (NEGATIVE)
        var_loss = -var_thresh                     # positive loss number (the VaR value)
        es_pred = np.zeros_like(var_thresh)
        for i in range(s_h1.shape[1]):
            tail = s_h1[:, i][s_h1[:, i] <= var_thresh[i]]
            es_pred[i] = -tail.mean() if len(tail) > 0 else 0.0  # ES = mean of losses in tail
        violations = (t_h1 < var_thresh).astype(int)  # FIXED: was -var_thresh
        out[q] = {
            "var_thresh_mean": float(var_thresh.mean()),
            "var_loss_mean": float(var_loss.mean()),
            "es_mean": float(es_pred.mean()),
            "violations": violations.tolist(),
            "violation_rate": float(violations.mean()),
            "n_violations": int(violations.sum()),
            "n_obs": int(len(violations)),
        }
    return out


# ============= Kupiec + Christoffersen tests =============
def kupiec_test(n_viol, n_obs, alpha):
    if n_obs == 0:
        return float("nan"), float("nan")
    pi_hat = n_viol / n_obs
    if pi_hat == 0 or pi_hat == 1:
        return float("inf"), 0.0
    ll_null = n_viol * np.log(alpha) + (n_obs - n_viol) * np.log(1 - alpha)
    ll_alt = n_viol * np.log(pi_hat) + (n_obs - n_viol) * np.log(1 - pi_hat)
    lr = -2 * (ll_null - ll_alt)
    p = 1 - sp_stats.chi2.cdf(lr, df=1)
    return float(lr), float(p)


def christoffersen_test(violations):
    n00 = n01 = n10 = n11 = 0
    for i in range(len(violations) - 1):
        a, b = int(violations[i]), int(violations[i + 1])
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        elif a == 1 and b == 1: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_alt = 0
    if n00 > 0 and pi > 0: ll_alt += n00 * np.log(1 - pi)
    if n01 > 0 and pi01 > 0: ll_alt += n01 * np.log(pi01)
    if n10 > 0 and (1 - pi11) > 0: ll_alt += n10 * np.log(1 - pi11)
    if n11 > 0 and pi11 > 0: ll_alt += n11 * np.log(pi11)
    ll_null = 0
    if (n00 + n10) > 0 and (1 - pi) > 0:
        ll_null += (n00 + n10) * np.log(1 - pi)
    if (n01 + n11) > 0 and pi > 0:
        ll_null += (n01 + n11) * np.log(pi)
    lr = -2 * (ll_null - ll_alt)
    p = 1 - sp_stats.chi2.cdf(lr, df=1)
    return float(lr), float(p)


# ============= Forecasting =============
def get_deep_samples(panel_csv, backbone_name, head_type, h=1):
    """Re-train backbone+head and return per-fold (samples, truth) pairs."""
    panel = pd.read_csv(panel_csv, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float32)
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)

    # Build supervised pairs (same as 22)
    X_all, Y_all = [], []
    for t in range(n - SEQ_LEN - h + 1):
        X_all.append(series[t:t + SEQ_LEN])
        Y_all.append(series[t + SEQ_LEN:t + SEQ_LEN + h])
    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    fold_results = []
    train_end_series = init_train
    fold_idx = 0
    while train_end_series + test_window + h <= n:
        test_end_series = train_end_series + test_window
        train_end_pair = train_end_series - SEQ_LEN
        test_end_pair = test_end_series - SEQ_LEN
        if train_end_pair <= 0 or test_end_pair <= train_end_pair:
            train_end_series += step
            continue
        X_train = X_all[:train_end_pair]
        Y_train = Y_all[:train_end_pair]
        X_test = X_all[train_end_pair:test_end_pair]
        Y_test = Y_all[train_end_pair:test_end_pair]
        # Use 22's train_and_save_predictions
        args = (f"{backbone_name}_{head_type}", h, fold_idx, SEED,
                X_train, Y_train, X_test, Y_test)
        out = exp_22.train_and_save_predictions(args)
        fold_results.append({
            "fold": fold_idx,
            "samples": out["samples"],
            "truth": out["truth"],
            "preds": out["preds"],
        })
        fold_idx += 1
        train_end_series += step
    return fold_results


def get_garch_gmm_samples(panel_csv, h=1, K=4):
    """GARCH(1,1) skewed-t + GMM K on standardized residuals."""
    from arch import arch_model
    from sklearn.mixture import GaussianMixture

    panel = pd.read_csv(panel_csv, index_col=0, parse_dates=True)
    series = panel["SP500_Return"].dropna().values.astype(np.float64)
    n = len(series)
    init_train = int(n * INITIAL_TRAIN_FRAC)
    test_window = int(n * TEST_FRAC)
    step = int(n * STEP_FRAC)

    fold_results = []
    train_end_series = init_train
    fold_idx = 0
    while train_end_series + test_window + h <= n:
        test_end_series = train_end_series + test_window
        train = series[:train_end_series]
        test = series[train_end_series:test_end_series]
        tr_pct = train * 100
        try:
            am = arch_model(tr_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
            res = am.fit(disp="off", show_warning=False)
        except Exception:
            train_end_series += step; fold_idx += 1; continue
        std_resid = np.asarray(res.std_resid)
        std_resid = std_resid[~np.isnan(std_resid)]
        if len(std_resid) < 50:
            train_end_series += step; fold_idx += 1; continue
        gmm = GaussianMixture(n_components=K, random_state=0, max_iter=200)
        gmm.fit(std_resid.reshape(-1, 1))
        fc = res.forecast(horizon=h, reindex=False, simulations=2000, method="simulation")
        cond_vol = np.sqrt(fc.variance.values[0, :])
        cond_mean = res.params.get("mu", 0.0)
        N = len(test)
        samples_pct = np.zeros((N_SAMPLES, N, h))
        for hh in range(h):
            gmm_samples, _ = gmm.sample(N_SAMPLES * N)
            gmm_samples = gmm_samples[:N_SAMPLES * N, 0].reshape(N_SAMPLES, N)
            samples_pct[:, :, hh] = cond_mean + cond_vol[hh] * gmm_samples
        samples = samples_pct / 100
        # Build truth
        Y = []
        for t in range(n - SEQ_LEN - h + 1):
            Y.append(series[t + SEQ_LEN:t + SEQ_LEN + h])
        Y = np.array(Y, dtype=np.float32)
        train_end_pair = train_end_series - SEQ_LEN
        test_end_pair = test_end_series - SEQ_LEN
        truth = Y[train_end_pair:test_end_pair]
        fold_results.append({
            "fold": fold_idx,
            "samples": samples,
            "truth": truth,
        })
        fold_idx += 1
        train_end_series += step
    return fold_results


# ============= Main =============
def main():
    panel_csv = ROOT / "data" / "raw" / "panel_monthly.csv"

    print("=" * 70)
    print(" Experiment 42: VaR backtest + FRTB capital on monthly S&P 500")
    print("=" * 70)

    variants = {
        "TimesNet_point": ("deep", "TimesNet", "point"),
        "TimesNet_gmm": ("deep", "TimesNet", "gmm"),
        "NBEATS_gmm": ("deep", "NBEATS", "gmm"),
        "GARCH_gmm": ("garch", None, None),
    }

    backtest = {}
    for vname, vinfo in variants.items():
        print(f"\n[{vname}]")
        t0 = time.time()
        if vinfo[0] == "deep":
            _, bb_name, head_type = vinfo
            folds = get_deep_samples(panel_csv, bb_name, head_type, h=1)
        else:
            folds = get_garch_gmm_samples(panel_csv, h=1)
        print(f"  trained in {time.time() - t0:.1f}s ({len(folds)} folds)")

        per_fold = []
        for fi, fold in enumerate(folds):
            stats = var_es_from_samples(fold["samples"], fold["truth"],
                                       levels=(0.01, 0.05, 0.025))
            entry = {"fold": fi, "n_obs": stats[0.05]["n_obs"]}
            for q, st in stats.items():
                lr_uc, p_uc = kupiec_test(st["n_violations"], st["n_obs"], q)
                lr_cc, p_cc = christoffersen_test(st["violations"])
                entry[f"q{q}"] = {
                    "violation_rate": st["violation_rate"],
                    "n_violations": st["n_violations"],
                    "kupiec_p": p_uc,
                    "christoffersen_p": p_cc,
                    "es_mean": st["es_mean"],
                    "var_thresh_mean": st["var_thresh_mean"],
                    "var_loss_mean": st["var_loss_mean"],
                }
            per_fold.append(entry)

        agg = {"per_fold": per_fold}
        for q in [0.01, 0.05, 0.025]:
            vrates = [pf[f"q{q}"]["violation_rate"] for pf in per_fold]
            pvals = [pf[f"q{q}"]["kupiec_p"] for pf in per_fold]
            cc_pvals = [pf[f"q{q}"]["christoffersen_p"] for pf in per_fold]
            es_means = [pf[f"q{q}"]["es_mean"] for pf in per_fold]
            agg[f"q{q}"] = {
                "violation_rate_mean": float(np.mean(vrates)),
                "kupiec_p_mean": float(np.mean(pvals)),
                "christoffersen_p_mean": float(np.mean(cc_pvals)),
                "kupiec_pass_rate_5pct": float(np.mean([p > 0.05 for p in pvals])),
                "es_mean_across_folds": float(np.mean(es_means)),
            }
        backtest[vname] = agg

    # Summary tables
    print("\n" + "=" * 70)
    print(" VaR backtest summary (1-step ahead, monthly SP500, 5 folds)")
    print("=" * 70)
    print(f"{'Variant':22s} {'q':>6s} {'ViolRate':>9s} {'KupPass':>8s} {'ChristP':>7s} {'ES(%)':>7s}")
    for vname, agg in backtest.items():
        for q in [0.01, 0.05, 0.025]:
            d = agg[f"q{q}"]
            print(f"{vname:22s} {q:>6.3f} {d['violation_rate_mean']*100:>8.2f}% "
                  f"{d['kupiec_pass_rate_5pct']*100:>7.0f}% "
                  f"{d['christoffersen_p_mean']:>7.3f} "
                  f"{d['es_mean_across_folds']*100:>6.2f}")

    print("\n" + "=" * 70)
    print(" FRTB-style capital: ES(97.5%) × $100 notional (monthly SP500, h=1)")
    print("=" * 70)
    print(f"{'Variant':22s} {'ES(97.5%)':>10s} {'Capital/$100':>15s} {'Δ vs TimesNet_point':>22s}")
    baseline_es = backtest["TimesNet_point"]["q0.025"]["es_mean_across_folds"]
    for vname, agg in backtest.items():
        d = agg["q0.025"]
        es = d["es_mean_across_folds"]
        cap = es * 100
        delta = (es - baseline_es) / baseline_es * 100
        print(f"{vname:22s} {es*100:>9.2f}% {cap:>14.2f}$ {delta:>+21.1f}%")

    out = {"backtest": backtest}
    out_path = RESULTS_DIR / "42_var_backtest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()