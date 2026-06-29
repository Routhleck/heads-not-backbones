"""
Experiment 03: Apply AMPD to real financial time series.

Goals:
  1. Discover periods in S&P 500 monthly returns 1871-2023
  2. Discover periods in multi-asset panel (SP500 + DXY + FedFunds + M2)
  3. Compare discovered periods vs 4 theory prior sets:
     - None (pure data-driven)
     - Zhou Jintao: 30y, 20y, 7y
     - Kondratiev-Juglar: 25y, 10y, 4y
     - Kuznets: 20y, 10y, 5y

Reference: proposal-v2.md §4.2 + §5 (v2.2 simplified scope)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd

from src.models.ampd import AMPD


DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


PRIOR_SETS = {
    "None": [],
    "Zhou_Jintao": [30 * 12, 20 * 12, 7 * 12],          # months
    "Kondratiev_Juglar": [25 * 12, 10 * 12, 4 * 12],
    "Kuznets": [20 * 12, 10 * 12, 5 * 12],
}


def discover_periods_on_series(
    X: pd.Series,
    name: str,
    top_k: int = 6,
    max_period: int = 360,
):
    """Apply AMPD to a single series and return discovered periods + amplitudes."""
    ampd = AMPD(top_k=top_k, max_period=max_period, min_period=6)
    periods = ampd.fit_discover(X.dropna().values)
    print(f"\n[{name}] Discovered periods (months): {[round(p,1) for p in periods]}")
    print(f"[{name}] Discovered periods (years):  {[round(p/12,2) for p in periods]}")
    print(f"[{name}] Amplitudes: {[round(a,2) for a in ampd.last_amplitudes_]}")
    return periods, ampd.last_amplitudes_


def discover_periods_multi_asset(
    X: pd.DataFrame,
    name: str,
    top_k: int = 6,
    max_period: int = 360,
    merge_tol: float = 0.05,
):
    """Apply AMPD to multi-asset panel and merge discovered periods."""
    ampd = AMPD(top_k=top_k, max_period=max_period, min_period=6)
    # Need to call fit_discover_multi directly
    periods = ampd.fit_discover_multi(X.values, merge_tol=merge_tol)
    print(f"\n[{name}] Multi-asset discovered periods (months): {[round(p,1) for p in periods]}")
    print(f"[{name}] Multi-asset discovered periods (years):  {[round(p/12,2) for p in periods]}")
    return periods


def evaluate_against_priors(discovered: np.ndarray, prior_periods_months: list, tol: float = 0.10):
    """For each prior period, find the closest discovered period and report rel error."""
    if not prior_periods_months or len(discovered) == 0:
        return []
    results = []
    for p in prior_periods_months:
        if len(discovered) == 0:
            results.append({"prior_years": p / 12, "matched": None, "rel_err": None})
            continue
        idx = np.argmin(np.abs(discovered - p))
        matched = discovered[idx]
        rel_err = abs(matched - p) / p
        results.append({
            "prior_months": p,
            "prior_years": p / 12,
            "matched_months": float(matched),
            "matched_years": float(matched / 12),
            "rel_err": float(rel_err),
        })
    return results


def main():
    print("=" * 70)
    print(" Experiment 03: AMPD on Real Financial Time Series")
    print("=" * 70)

    # Load panel
    print(f"\n[Load] Reading {DATA_PATH}")
    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"[Load] Panel shape: {panel.shape}")
    print(f"[Load] Date range: {panel.index.min()} → {panel.index.max()}")

    all_results = {}

    # ------------------------------------------------------------
    # Test 1: S&P 500 monthly returns — single asset
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Test 1: SP500 monthly log-returns")
    print("=" * 70)
    sp_ret = panel["SP500_Return"].dropna()
    print(f"  Date range: {sp_ret.index.min()} → {sp_ret.index.max()}, n={len(sp_ret)}")
    discovered, amplitudes = discover_periods_on_series(sp_ret, "SP500_Return", top_k=6)

    test1 = {
        "series": "SP500_Return",
        "n_obs": len(sp_ret),
        "date_range": [str(sp_ret.index.min()), str(sp_ret.index.max())],
        "discovered_periods_months": [float(p) for p in discovered],
        "discovered_periods_years": [float(p/12) for p in discovered],
        "amplitudes": [float(a) for a in amplitudes],
    }
    # Compare against each prior set
    test1["prior_comparisons"] = {}
    for name, priors in PRIOR_SETS.items():
        test1["prior_comparisons"][name] = evaluate_against_priors(discovered, priors)
    all_results["test1_sp500_returns"] = test1

    # ------------------------------------------------------------
    # Test 2: SP500 real price — trend + cycle (single asset)
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Test 2: SP500 real price (deflated)")
    print("=" * 70)
    sp_real = panel["SP500_Real"].dropna()
    print(f"  Date range: {sp_real.index.min()} → {sp_real.index.max()}, n={len(sp_real)}")
    discovered, amplitudes = discover_periods_on_series(sp_real, "SP500_Real", top_k=6)
    test2 = {
        "series": "SP500_Real",
        "n_obs": len(sp_real),
        "date_range": [str(sp_real.index.min()), str(sp_real.index.max())],
        "discovered_periods_months": [float(p) for p in discovered],
        "discovered_periods_years": [float(p/12) for p in discovered],
        "amplitudes": [float(a) for a in amplitudes],
        "prior_comparisons": {n: evaluate_against_priors(discovered, p) for n, p in PRIOR_SETS.items()},
    }
    all_results["test2_sp500_real"] = test2

    # ------------------------------------------------------------
    # Test 3: Multi-asset on returns (SP500 + DXY + M2 + FedFunds)
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Test 3: Multi-asset on returns (SP500, DXY, M2_Growth, FedFunds)")
    print("=" * 70)
    multi_assets = panel[["SP500_Return", "DXY_Return", "M2_Growth", "FedFunds"]].dropna()
    print(f"  Date range: {multi_assets.index.min()} → {multi_assets.index.max()}, n={len(multi_assets)}")
    discovered = discover_periods_multi_asset(multi_assets, "MultiAsset_4series", top_k=6)
    test3 = {
        "series": ["SP500_Return", "DXY_Return", "M2_Growth", "FedFunds"],
        "n_obs": len(multi_assets),
        "date_range": [str(multi_assets.index.min()), str(multi_assets.index.max())],
        "discovered_periods_months": [float(p) for p in discovered],
        "discovered_periods_years": [float(p/12) for p in discovered],
        "prior_comparisons": {n: evaluate_against_priors(discovered, p) for n, p in PRIOR_SETS.items()},
    }
    all_results["test3_multi_asset"] = test3

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" SUMMARY: Discovered periods vs Theory Priors")
    print("=" * 70)
    for tname, tdata in all_results.items():
        print(f"\n[{tname}]")
        print(f"  Discovered (years): {[round(p,2) for p in tdata['discovered_periods_years']]}")
        for prior_name, comps in tdata["prior_comparisons"].items():
            if not comps:
                continue
            print(f"  vs {prior_name}:")
            for c in comps:
                if c.get("matched_years"):
                    status = "✓" if c["rel_err"] < 0.10 else "✗"
                    print(f"    {status} prior {c['prior_years']:.1f}y → "
                          f"matched {c['matched_years']:.2f}y, rel_err {c['rel_err']*100:.1f}%")
                else:
                    print(f"    - no match for prior {c['prior_years']:.1f}y")

    # Save full results
    out = RESULTS_DIR / "03_real_data_periods.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()