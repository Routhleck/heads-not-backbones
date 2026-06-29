"""
Per-fold CRPS Differential test (proper DM test approximation).

For each (backbone, horizon) cell, we have 5 fold CRPS means from
exp 22 (point, gmm) and exp 34 (gauss). We compute the per-fold
CRPS differential d_f = CRPS_A_f - CRPS_B_f for each of the 5
folds, and run a paired t-test (df=4) to test H0: mean(d) = 0.

We also bootstrap the 5 fold CRPS means 10,000 times to get a
non-parametric 95% CI on the CRPS-SS difference.

Note: with 5 fold observations, the t-test is underpowered for
small effects. The test is therefore reported as "suggestive"
rather than "definitive". For a fully-powered DM test using
per-test-point CRPS, the experiment would need to save per-point
arrays (~128 per fold × 5 folds = 640 obs per cell).

Output: results/35g_dm_per_fold.json with 48 cells of
(paired t-test, bootstrap CI) for:
  - point vs gauss (per backbone × horizon)
  - point vs gmm   (per backbone × horizon)
  - gauss vs gmm   (per backbone × horizon)
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))

# Load both experiments
with open(ROOT / "results/22_sota_comparison.json") as f:
    exp22 = json.load(f)
with open(ROOT / "results/34_gaussian_head.json") as f:
    exp34 = json.load(f)

HORIZONS = [1, 3, 6, 12]
BACKBONES = ["TimesNet", "DLinear", "NBEATS", "PatchTST"]
HEADS = ["point", "gauss", "gmm"]
N_BOOT = 10_000


def paired_t_test(d):
    """Paired t-test on the fold-level CRPS differentials.

    Args:
        d: array of n_fold CRPS differentials (A_f - B_f for f in folds)
    Returns: dict with mean, std, t_stat, df, p_value
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean_d = float(d.mean())
    std_d = float(d.std(ddof=1)) if n > 1 else 0.0
    if std_d < 1e-12:
        return {"mean": mean_d, "std": std_d, "t_stat": 0.0, "df": n - 1,
                "p_value": 1.0, "n": n}
    t_stat = mean_d / (std_d / np.sqrt(n))
    p_val = 2.0 * (1.0 - sp_stats.t.cdf(abs(t_stat), df=n - 1))
    return {
        "mean": mean_d, "std": std_d, "t_stat": float(t_stat),
        "df": n - 1, "p_value": float(p_val), "n": n,
    }


def bootstrap_ci(a, b, n_boot=N_BOOT, alpha=0.05, seed=42):
    """Bootstrap CI on CRPS-SS percentage difference.

    For each resample, draw n_fold with replacement from (a, b) and
    compute 1 - mean(a)/mean(b).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    assert len(b) == n
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ai = a[idx].mean()
        bi = b[idx].mean()
        if bi > 0:
            stats[i] = (1.0 - ai / bi) * 100
        else:
            stats[i] = 0.0
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1.0 - alpha / 2))
    return {"mean_pct": float((1.0 - a.mean() / b.mean()) * 100),
            "ci_lo_pct": lo, "ci_hi_pct": hi}


# Build the per-fold CRPS arrays
# exp 22 has crps_per_fold for [point, gmm] × 4 backbones × 4 horizons
# exp 34 has crps_per_fold for [gauss] × 4 backbones × 4 horizons
per_fold = {}
for h in HORIZONS:
    for bb in BACKBONES:
        for head in HEADS:
            key = f"{bb}_{head}"
            if head in ("point", "gmm"):
                per_fold[(h, key)] = exp22["metrics"][f"h{h}"][f"{bb}_{head}"]["crps_per_fold"]
            else:  # gauss
                per_fold[(h, key)] = exp34["metrics"][f"h{h}"][bb]["crps_per_fold"]

# 48 cells: 3 comparisons × 4 backbones × 4 horizons
results = {f"h{h}": {bb: {} for bb in BACKBONES} for h in HORIZONS}
for h in HORIZONS:
    for bb in BACKBONES:
        for (a_head, b_head, label) in [
            ("point", "gauss", "point_vs_gauss"),
            ("point", "gmm", "point_vs_gmm"),
            ("gauss", "gmm", "gauss_vs_gmm"),
        ]:
            a_crps = per_fold[(h, f"{bb}_{a_head}")]
            b_crps = per_fold[(h, f"{bb}_{b_head}")]
            d = [a - b for a, b in zip(a_crps, b_crps)]
            tt = paired_t_test(d)
            ci = bootstrap_ci(b_crps, a_crps)  # CRPS-SS of B over A
            results[f"h{h}"][bb][label] = {
                "n_folds": tt["n"],
                "diff_a_minus_b": tt["mean"],  # mean of (CRPS_A - CRPS_B) per fold
                "std_diff": tt["std"],
                "t_stat": tt["t_stat"],
                "df": tt["df"],
                "p_value": tt["p_value"],
                "sig_5pct": "**" if tt["p_value"] < 0.05 else ("*" if tt["p_value"] < 0.10 else ""),
                "crps_ss_b_vs_a_pct": ci["mean_pct"],
                "ci_lo_pct": ci["ci_lo_pct"],
                "ci_hi_pct": ci["ci_hi_pct"],
                "ci_excludes_zero": ci["ci_lo_pct"] > 0 or ci["ci_hi_pct"] < 0,
            }

# Print summary
print("=" * 78)
print(" Per-fold CRPS Differential Test (5-fold paired t-test + bootstrap CI)")
print("=" * 78)

for label, key, a_head, b_head in [
    ("POINT vs GAUSS", "point_vs_gauss", "point", "gauss"),
    ("POINT vs GMM", "point_vs_gmm", "point", "gmm"),
    ("GAUSS vs GMM", "gauss_vs_gmm", "gauss", "gmm"),
]:
    print(f"\n{'='*78}")
    print(f"  {label}  (positive d_f = {a_head} better; positive CRPS-SS = {b_head} better)")
    print(f"{'='*78}")
    print(f"  {'Backbone':<10} {'Horizon':<8} {'mean(d)':>10} {'p-value':>10} {'sig':>5} {'CRPS-SS(B)':>12} {'95% CI':>20} {'excl 0':>7}")
    for h in HORIZONS:
        for bb in BACKBONES:
            r = results[f"h{h}"][bb][key]
            print(f"  {bb:<10} h={h:<6} {r['diff_a_minus_b']*100:>+9.4f}% {r['p_value']:>10.4f} {r['sig_5pct']:>5} "
                  f"{r['crps_ss_b_vs_a_pct']:>+11.2f}% [{r['ci_lo_pct']:>+6.2f}, {r['ci_hi_pct']:>+6.2f}] {'YES' if r['ci_excludes_zero'] else 'no':>7}")

# Save
out = {
    "config": {
        "n_folds": 5,
        "n_bootstrap": N_BOOT,
        "test": "paired t-test on per-fold CRPS differentials + bootstrap CI on CRPS-SS",
        "interpretation": (
            "With 5 fold means per cell, the paired t-test has df=4 and "
            "is underpowered for small effects. The test is reported as "
            "suggestive, not definitive. A fully-powered DM test would "
            "require per-test-point CRPS arrays."
        ),
    },
    "results": results,
}
out_path = ROOT / "results/35g_dm_per_fold.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {out_path}")
