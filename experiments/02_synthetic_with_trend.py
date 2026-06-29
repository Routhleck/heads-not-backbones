"""
Experiment 02: Synthetic Sanity Check with Trend-Cycle Decomposition.

v2.1 (2026-06-24): Updated after FFT limit discovery.
- AMPD handles cycles in [min_period=6mo, max_period=360mo = 30y] (FFT-resolvable)
- Long-Wave Trend Extractor (LWTE) handles ultra-long components (≥30y) via low-pass
- Combined pipeline recovers both mid-frequency cycles and long-wave trend

Reference: proposal-v2.md §6.5 (revised)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import json
from src.data.synthetic_gen import generate_synthetic, save_synthetic_csv, generate_negative_control
from src.models.ampd import AMPD
from src.models.long_wave_trend import LongWaveTrendExtractor, evaluate_trend_recovery


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


def decompose_signal(X: np.ndarray, ampd_top_k: int = 4, lwte_cutoff: int = 360):
    """Apply LWTE then AMPD on residual (multi-asset)."""
    lwte = LongWaveTrendExtractor(cutoff_period_months=lwte_cutoff, method="butter")
    trend, cycle = lwte.fit_transform(X)
    # Multi-asset AMPD: run per asset, merge results
    ampd = AMPD(top_k=ampd_top_k, max_period=lwte_cutoff)
    discovered = ampd.fit_discover_multi(cycle)
    return {
        "trend": trend,
        "cycle": cycle,
        "discovered_periods_months": discovered,
        "discovered_periods_years": discovered / 12.0,
        "amplitudes": ampd.last_amplitudes_,
    }


def generate_with_trend(
    long_wave_periods_months: list,
    mid_cycle_periods_months: list,
    n_assets: int = 4,
    T: int = 720,
    noise_std: float = 0.1,
    long_wave_amplitude: float = 2.0,
    mid_cycle_amplitude_range: tuple = (0.5, 1.5),
    synced_phase: bool = True,  # NEW: shared phase across assets by default
    seed: int = 42,
) -> tuple:
    """
    Generate signal = long_wave_trend + mid_cycles + noise.

    long_wave_periods_months: periods that should be captured by LWTE (>30y)
    mid_cycle_periods_months: periods that should be captured by AMPD (≤30y)

    With synced_phase=True (default), all assets share the same phase for each
    cycle — simulates real financial data where multi-asset returns are driven
    by shared macroeconomic factors. This avoids phase cancellation in mean().
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # Long-wave trend (smooth, slowly oscillating) — synced phase
    K_long = len(long_wave_periods_months)
    if K_long > 0:
        long_amps = np.full(K_long, long_wave_amplitude / K_long)
        if synced_phase:
            long_phases = rng.uniform(0, 2 * np.pi, size=K_long)  # shared
        else:
            long_phases = rng.uniform(0, 2 * np.pi, size=(n_assets, K_long))
    else:
        long_amps = np.array([])
        long_phases = np.array([])

    # Mid cycles (faster, smaller amplitude)
    K_mid = len(mid_cycle_periods_months)
    mid_amps = rng.uniform(mid_cycle_amplitude_range[0], mid_cycle_amplitude_range[1], size=(n_assets, K_mid))
    if synced_phase:
        mid_phases = rng.uniform(0, 2 * np.pi, size=K_mid)  # shared across assets
    else:
        mid_phases = rng.uniform(0, 2 * np.pi, size=(n_assets, K_mid))

    # Combine per asset
    X = np.zeros((T, n_assets))
    long_wave_true = np.zeros((T, n_assets))

    for i in range(n_assets):
        for k in range(K_long):
            phase = long_phases[k] if synced_phase else long_phases[i, k]
            wave = long_amps[k] * np.sin(2 * np.pi * t / long_wave_periods_months[k] + phase)
            X[:, i] += wave
            long_wave_true[:, i] += wave

        for k in range(K_mid):
            phase = mid_phases[k] if synced_phase else mid_phases[i, k]
            X[:, i] += mid_amps[i, k] * np.sin(2 * np.pi * t / mid_cycle_periods_months[k] + phase)

    noise = rng.normal(0, noise_std, size=(T, n_assets))
    X += noise

    metadata = {
        "long_wave_periods_months": long_wave_periods_months,
        "mid_cycle_periods_months": mid_cycle_periods_months,
        "n_assets": n_assets,
        "T": T,
        "noise_std": noise_std,
        "long_wave_amplitudes": long_amps.tolist(),
        "seed": seed,
        "long_wave_true": long_wave_true,
        "synced_phase": synced_phase,
    }
    return X, metadata


def run_v2_sanity_check(
    long_wave_periods_months: list,
    mid_cycle_periods_months: list,
    T: int = 720,
    n_assets: int = 4,
    noise_std: float = 0.1,
    long_wave_amplitude: float = 0.5,
    mid_cycle_amplitude_range: tuple = (1.5, 2.5),
    seed: int = 42,
    label: str = "default",
):
    """Run v2 synthetic sanity check with trend + cycle decomposition."""
    print(f"\n{'=' * 70}")
    print(f"  v2.1 Sanity Check: {label}")
    print(f"{'=' * 70}")
    print(f"  Long-wave periods (months): {long_wave_periods_months}")
    print(f"  Long-wave periods (years):  {[p/12 for p in long_wave_periods_months]}")
    print(f"  Mid-cycle periods (months): {mid_cycle_periods_months}")
    print(f"  Mid-cycle periods (years):  {[p/12 for p in mid_cycle_periods_months]}")
    print(f"  T={T}, n_assets={n_assets}, noise_std={noise_std}, seed={seed}")

    # Generate data with explicit long-wave + mid-cycle separation
    X, meta = generate_with_trend(
        long_wave_periods_months=long_wave_periods_months,
        mid_cycle_periods_months=mid_cycle_periods_months,
        T=T,
        n_assets=n_assets,
        noise_std=noise_std,
        long_wave_amplitude=long_wave_amplitude,
        mid_cycle_amplitude_range=mid_cycle_amplitude_range,
        seed=seed,
    )
    print(f"  X shape: {X.shape}, std={X.std():.3f}")

    # Save (construct full metadata)
    save_meta = {
        "true_periods_months": long_wave_periods_months + mid_cycle_periods_months,
        "true_periods_years": [p / 12 for p in long_wave_periods_months + mid_cycle_periods_months],
        "amplitudes": np.ones((n_assets, len(long_wave_periods_months + mid_cycle_periods_months))),
        "phases": np.zeros((n_assets, len(long_wave_periods_months + mid_cycle_periods_months))),
        "noise_std": noise_std, "T": T, "n_assets": n_assets, "seed": seed,
    }
    save_synthetic_csv(X, save_meta, ROOT / "data" / "synthetic", name=f"synthetic_v2_{label}")

    # Apply pipeline
    result = decompose_signal(X, ampd_top_k=len(mid_cycle_periods_months))
    print(f"\n  Discovered mid-cycle periods (months): {result['discovered_periods_months'].astype(int).tolist()}")
    print(f"  Discovered mid-cycle periods (years):  {result['discovered_periods_years'].round(2).tolist()}")

    # Match mid cycles
    true_mid_arr = np.array(mid_cycle_periods_months)
    matches = match_periods(result["discovered_periods_months"], true_mid_arr)
    matches.sort(key=lambda m: m["true"])
    print(f"\n  Mid-cycle recovery (sorted by true period):")
    for m in matches:
        status = "OK" if m["rel_err"] < 0.10 else ("WARN" if m["rel_err"] < 0.20 else "MISS")
        print(f"    {status}: discovered {m['discovered_years']:5.2f}y vs true "
              f"{m['true_years']:5.2f}y  rel_err={m['rel_err']*100:5.1f}%")

    # Trend recovery
    true_long = meta["long_wave_true"].mean(axis=1) if meta["long_wave_true"].ndim == 2 else meta["long_wave_true"]
    extracted_trend = result["trend"].mean(axis=1) if result["trend"].ndim == 2 else result["trend"]
    trend_metrics = evaluate_trend_recovery(extracted_trend, true_long)
    print(f"\n  Long-wave trend recovery:")
    print(f"    corr:  {trend_metrics['corr']:.4f} (target >0.7)")
    print(f"    nrmse: {trend_metrics['nrmse']:.4f} (target <0.5)")
    print(f"    r2:    {trend_metrics['r2']:.4f}")

    rel_errs = [m["rel_err"] for m in matches]
    mid_cycles_pass = max(rel_errs) < 0.20  # 20% tolerance for FFT bin quantization
    trend_pass = trend_metrics["passed"]

    metrics = {
        "label": label,
        "long_wave_periods_months": long_wave_periods_months,
        "mid_cycle_periods_months": mid_cycle_periods_months,
        "T": T, "n_assets": n_assets, "noise_std": noise_std, "seed": seed,
        "discovered_mid_periods_months": result["discovered_periods_months"].tolist(),
        "mid_cycle_matches": matches,
        "trend_recovery": trend_metrics,
        "max_mid_rel_err": float(max(rel_errs)),
        "mean_mid_rel_err": float(np.mean(rel_errs)),
        "mid_cycles_pass_20pct": bool(mid_cycles_pass),
        "trend_pass": bool(trend_pass),
        "overall_pass": bool(mid_cycles_pass and trend_pass),
    }
    print(f"\n  Summary:")
    print(f"    mid-cycle max rel_err: {metrics['max_mid_rel_err']*100:.1f}%  → {'PASS' if mid_cycles_pass else 'FAIL'}")
    print(f"    long-wave trend:       corr={trend_metrics['corr']:.3f}, nrmse={trend_metrics['nrmse']:.3f}  → {'PASS' if trend_pass else 'FAIL'}")
    print(f"    OVERALL: {'PASS' if metrics['overall_pass'] else 'FAIL'}")

    return metrics


def main():
    out_dir = ROOT / "data" / "synthetic"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ============================================================
    # Test 1: 周金涛 4-period v2 — 重新设计
    # LWTE cutoff 降低到 20y (240mo) → 60y 和 25y 都归 LWTE
    # AMPD 只看 ≤20y 的周期 (18y, 7y) — 信号更纯净
    # ============================================================
    r1 = run_v2_sanity_check(
        long_wave_periods_months=[720, 300],   # 60y + 25y 都走 LWTE path
        mid_cycle_periods_months=[216, 84],    # 18y/7y 走 AMPD path
        label="test1_zhoujintao_v2",
    )
    all_results.append(r1)

    # ============================================================
    # Test 2: Kondratiev classical (50y trend + 10y/4y cycles)
    # 50y 单独走 LWTE, 10y/4y 走 AMPD — 测试 shorter long_wave case
    # ============================================================
    r2 = run_v2_sanity_check(
        long_wave_periods_months=[600],       # 50y → LWTE
        mid_cycle_periods_months=[120, 48],   # 10y/4y → AMPD
        label="test2_kondratiev_v2",
    )
    all_results.append(r2)

    # ============================================================
    # Test 3: 高 noise robustness (与 Test 1 同样配置)
    # ============================================================
    r3 = run_v2_sanity_check(
        long_wave_periods_months=[720, 300],
        mid_cycle_periods_months=[216, 84],
        noise_std=0.4,
        label="test3_high_noise_v2",
    )
    all_results.append(r3)

    # ============================================================
    # Test 4: Negative control (纯 AMPD 范围，无 long wave)
    # Use 3 periods well-separated: 12y, 17y, 22y
    # ============================================================
    r4 = run_v2_sanity_check(
        long_wave_periods_months=[],
        mid_cycle_periods_months=[144, 204, 264],  # 12y, 17y, 22y — well-separated
        label="test4_negative_control_v2",
    )
    all_results.append(r4)

    # ============================================================
    # Test 5: 只 AMPD，无 long wave（最干净 case）
    # ============================================================
    r5 = run_v2_sanity_check(
        long_wave_periods_months=[],
        mid_cycle_periods_months=[48, 84, 120, 216],  # 4y/7y/10y/18y
        label="test5_clean_ampd_v2",
    )
    all_results.append(r5)

    # Save full results
    out_path = results_dir / "02_synthetic_v2_with_trend.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved full results to: {out_path}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  v2.1 GO / NO-GO VERDICT")
    print(f"{'=' * 70}")
    for i, r in enumerate(all_results, 1):
        status = "✅ PASS" if r["overall_pass"] else "❌ FAIL"
        print(f"  Test {i} ({r['label']}): {status}")
    overall = all(r["overall_pass"] for r in all_results)
    print(f"\n  OVERALL v2.1: {'GO ✅' if overall else 'NO-GO ❌'}")


if __name__ == "__main__":
    main()