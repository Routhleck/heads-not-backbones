"""
Experiment 01: Synthetic Data Sanity Check.

Goal: verify that AMPD can recover known periods from synthetic data.

This is the GO/NO-GO gate for v2 direction. If AMPD cannot recover 4 periods
in a clean synthetic signal, the entire direction is suspect.

Reference: proposal-v2.md §6.5
"""
import sys
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import json
from src.data.synthetic_gen import generate_synthetic, save_synthetic_csv
from src.models.ampd import AMPD


def match_periods(discovered: np.ndarray, true_periods: np.ndarray) -> list:
    """Find optimal 1-1 matching between discovered and true periods via rel error."""
    from scipy.optimize import linear_sum_assignment
    cost = np.abs(discovered[:, None] / true_periods[None, :] - 1.0)
    row_idx, col_idx = linear_sum_assignment(cost)
    return [
        {
            "discovered": float(discovered[r]),
            "discovered_years": float(discovered[r] / 12),
            "true": float(true_periods[c]),
            "true_years": float(true_periods[c] / 12),
            "rel_err": float(cost[r, c]),
        }
        for r, c in zip(row_idx, col_idx)
    ]


def run_sanity_check(
    true_periods_months: list,
    T: int = 720,
    n_assets: int = 4,
    noise_std: float = 0.1,
    seed: int = 42,
    label: str = "default",
    out_dir: Path = None,
):
    """Run one synthetic sanity check and return metrics."""
    print(f"\n{'=' * 70}")
    print(f"  Synthetic Sanity Check: {label}")
    print(f"{'=' * 70}")
    print(f"  True periods (months): {true_periods_months}")
    print(f"  True periods (years):  {[p/12 for p in true_periods_months]}")
    print(f"  T={T}, n_assets={n_assets}, noise_std={noise_std}, seed={seed}")

    # Generate data
    X, meta = generate_synthetic(
        true_periods_months=true_periods_months,
        n_assets=n_assets,
        T=T,
        noise_std=noise_std,
        seed=seed,
    )
    print(f"  X shape: {X.shape}, mean={X.mean():.3f}, std={X.std():.3f}")

    # Save
    if out_dir is not None:
        save_synthetic_csv(X, meta, out_dir, name=f"synthetic_{label}")

    # Run AMPD on the multi-asset average (most basic case)
    ampd = AMPD(top_k=4)
    discovered = ampd.fit_discover(X)
    print(f"\n  Discovered periods (months): {discovered.astype(int).tolist()}")
    print(f"  Discovered periods (years):  {(discovered/12).round(2).tolist()}")

    # Match
    true_arr = np.array(true_periods_months)
    matches = match_periods(discovered, true_arr)
    matches.sort(key=lambda m: m["true"])  # sort by true period
    print(f"\n  Optimal matching (sorted by true period):")
    for m in matches:
        status = "OK" if m["rel_err"] < 0.05 else ("WARN" if m["rel_err"] < 0.15 else "MISS")
        print(f"    {status}: discovered {m['discovered_years']:5.2f}y vs true "
              f"{m['true_years']:5.2f}y  rel_err={m['rel_err']*100:5.1f}%")

    rel_errs = [m["rel_err"] for m in matches]
    metrics = {
        "label": label,
        "true_periods_months": true_periods_months,
        "T": T,
        "n_assets": n_assets,
        "noise_std": noise_std,
        "seed": seed,
        "discovered_periods_months": discovered.tolist(),
        "matches": matches,
        "max_rel_err": float(max(rel_errs)),
        "mean_rel_err": float(np.mean(rel_errs)),
        "median_rel_err": float(np.median(rel_errs)),
        "all_within_5pct": bool(max(rel_errs) < 0.05),
        "all_within_15pct": bool(max(rel_errs) < 0.15),
    }
    print(f"\n  Summary:")
    print(f"    max rel_err:   {metrics['max_rel_err']*100:.1f}%")
    print(f"    mean rel_err:  {metrics['mean_rel_err']*100:.1f}%")
    print(f"    all within 5%: {metrics['all_within_5pct']}")
    print(f"    all within 15%: {metrics['all_within_15pct']}")

    return metrics


def main():
    out_dir = ROOT / "data" / "synthetic"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ============================================================
    # Test 1: Standard 4-period (周金涛's 60y/25y/18y/7y in months)
    # ============================================================
    r1 = run_sanity_check(
        true_periods_months=[720, 300, 216, 84],  # 60y, 25y, 18y, 7y
        T=720,
        n_assets=4,
        noise_std=0.1,
        seed=42,
        label="test1_zhoujintao_low_noise",
        out_dir=out_dir,
    )
    all_results.append(r1)

    # ============================================================
    # Test 2: Higher noise
    # ============================================================
    r2 = run_sanity_check(
        true_periods_months=[720, 300, 216, 84],
        T=720,
        n_assets=4,
        noise_std=0.3,
        seed=42,
        label="test2_zhoujintao_high_noise",
        out_dir=out_dir,
    )
    all_results.append(r2)

    # ============================================================
    # Test 3: Kondratiev classical 50y + Juglar 10y + Kitchin 4y
    # ============================================================
    r3 = run_sanity_check(
        true_periods_months=[600, 120, 48],  # 50y, 10y, 4y
        T=720,
        n_assets=4,
        noise_std=0.1,
        seed=42,
        label="test3_kondratiev_classical",
        out_dir=out_dir,
    )
    all_results.append(r3)

    # ============================================================
    # Test 4: Negative control — deliberately wrong periods
    # Used to test if anchor regularization helps (later experiments)
    # ============================================================
    r4 = run_sanity_check(
        true_periods_months=[156, 204, 264, 396],  # 13y, 17y, 22y, 33y
        T=720,
        n_assets=4,
        noise_std=0.1,
        seed=42,
        label="test4_negative_control",
        out_dir=out_dir,
    )
    all_results.append(r4)

    # ============================================================
    # Save all results
    # ============================================================
    out_path = results_dir / "01_synthetic_sanity_check.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved full results to: {out_path}")

    # ============================================================
    # Final verdict
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  GO / NO-GO VERDICT")
    print(f"{'=' * 70}")
    test1_pass = r1["all_within_5pct"]
    test2_pass = r2["all_within_15pct"]
    test3_pass = r3["all_within_15pct"]
    test4_pass = r4["all_within_15pct"]
    print(f"  Test 1 (Zhou 4-period, low noise, 5% tol):  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (Zhou 4-period, high noise, 15% tol): {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (Kondratiev 3-period, 15% tol):       {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Test 4 (Negative control, 15% tol):          {'PASS' if test4_pass else 'FAIL'}")

    overall_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"\n  OVERALL: {'GO' if overall_pass else 'NO-GO'}")
    if not overall_pass:
        print("  ⚠️  AMPD failed to recover periods → re-examine method before real data experiments.")
    else:
        print("  ✅ AMPD passes sanity check → proceed to real data (W1-2 timeline continues).")


if __name__ == "__main__":
    main()