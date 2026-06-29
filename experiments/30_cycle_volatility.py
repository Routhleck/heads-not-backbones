"""
Experiment 30: AMPD cycles vs S&P 500 realized volatility
============================================================

Tests whether the cycle structure detected by AMPD in S&P 500 returns
themselves is associated with realized volatility regimes.

Hypothesis: When AMPD detects a SHORT cycle (< 60 months) in SP500 returns,
realized volatility is HIGHER than when AMPD detects a LONG cycle (> 120
months). Short cycles correspond to mean-reverting, high-frequency regimes
(rapid bull/bear cycles during stressed periods); long cycles correspond
to low-frequency drift regimes (secular bull markets).

Analysis:
  - 25 rolling windows (exp 20), each 360 months.
  - Bin windows by SP500_Return AMPD top-1 period:
      < 60 months  (short cycle: high vol regime)
      60-120 months (medium)
      > 120 months (long: secular drift regime)
  - For each window, compute mean 12-month rolling std of SP500_Return.
  - Kruskal-Wallis test across bins.

Usage:
    python experiments/30_cycle_volatility.py
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
AMPD_JSON = ROOT / "results" / "20_ampd_cycle_analysis.json"
RESULTS_DIR = ROOT / "results"


def main():
    print("=" * 72)
    print(" Experiment 30: AMPD cycles vs SP500 realized volatility")
    print("=" * 72)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    sp_ret = panel["SP500_Return"].dropna()

    ampd = json.load(open(AMPD_JSON))
    te = ampd["time_evolution"]
    sp_windows = te["SP500_Return"]
    print(f"[windows] {len(sp_windows)} rolling windows of "
          f"{ampd['config']['rolling_window_months']} months")

    # Build per-window dataset
    rows = []
    for win in sp_windows:
        start = pd.Timestamp(win["start_date"])
        end = pd.Timestamp(win["end_date"])

        win_ret = sp_ret[(sp_ret.index >= start) & (sp_ret.index <= end)]
        if len(win_ret) < 24:
            continue

        # Realized volatility = mean of 12-month rolling std
        vol_12m = win_ret.rolling(12, min_periods=6).std().mean()
        if np.isnan(vol_12m):
            continue

        # SP500 AMPD top-1 period (months)
        periods = win.get("periods_months", [])
        sp_top = periods[0] if periods else np.nan

        rows.append({
            "start": start, "end": end,
            "sp_top_period_months": sp_top,
            "vol_12m": vol_12m,
            "abs_return_mean": float(win_ret.abs().mean()),
        })
    df = pd.DataFrame(rows)
    print(f"[matched] {len(df)} windows with valid SP500 + AMPD data")
    print(f"[period range] {df['sp_top_period_months'].min():.1f} to {df['sp_top_period_months'].max():.1f} months")

    # ----- Bin by period -----
    df["period_bin"] = pd.cut(
        df["sp_top_period_months"],
        bins=[0, 60, 120, 1e6],
        labels=["<60m\n(short)", "60-120m\n(medium)", ">120m\n(long)"],
    )

    summary = df.groupby("period_bin", observed=False).agg(
        count=("vol_12m", "count"),
        vol_mean=("vol_12m", "mean"),
        vol_std=("vol_12m", "std"),
        vol_median=("vol_12m", "median"),
        absret_mean=("abs_return_mean", "mean"),
    ).round(4)
    print("\n[AMPD top-1 period bin] -> SP500 12m realized vol:")
    print(summary)

    # Kruskal-Wallis test
    groups = [g["vol_12m"].values for _, g in df.groupby("period_bin", observed=False) if len(g) >= 2]
    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
        print(f"\n[Kruskal-Wallis across 3 bins] H={h_stat:.4f}, p={p_kw:.4f}")
    else:
        h_stat, p_kw = np.nan, np.nan

    # Mann-Whitney U: short vs long
    short_v = df[df["period_bin"] == "<60m\n(short)"]["vol_12m"].values
    long_v = df[df["period_bin"] == ">120m\n(long)"]["vol_12m"].values
    if len(short_v) >= 2 and len(long_v) >= 2:
        u, p_u = stats.mannwhitneyu(short_v, long_v, alternative="greater")
        print(f"[Mann-Whitney U short>long vol] U={u:.4f}, p={p_u:.4f}  (n_short={len(short_v)}, n_long={len(long_v)})")
    else:
        u, p_u = np.nan, np.nan
        print(f"[Mann-Whitney] insufficient data: n_short={len(short_v)}, n_long={len(long_v)}")

    # Spearman correlation
    df_valid = df.dropna(subset=["sp_top_period_months"])
    df_valid["log_period"] = np.log(df_valid["sp_top_period_months"])
    rho, p_rho = stats.spearmanr(df_valid["log_period"], df_valid["vol_12m"])
    print(f"\n[Spearman corr log(SP500 top period) vs SP500 vol]  rho={rho:.4f}, p={p_rho:.4f}, n={len(df_valid)}")

    # ----- Save -----
    out = {
        "config": {
            "rolling_window_months": ampd["config"]["rolling_window_months"],
            "rolling_step_months": ampd["config"]["rolling_step_months"],
            "vol_window_months": 12,
            "period_bins_months": [60, 120],
            "series_used": "SP500_Return",
        },
        "summary_by_bin": summary.reset_index().to_dict(orient="records"),
        "tests": {
            "kruskal_wallis_3bins": {
                "H": float(h_stat) if not np.isnan(h_stat) else None,
                "p_value": float(p_kw) if not np.isnan(p_kw) else None,
                "n_windows": int(len(df)),
            },
            "mann_whitney_short_vs_long": {
                "U": float(u) if not np.isnan(u) else None,
                "p_value": float(p_u) if not np.isnan(p_u) else None,
                "alternative": "short > long",
                "n_short": int(len(short_v)),
                "n_long": int(len(long_v)),
            },
            "spearman_log_period_vs_vol": {
                "rho": float(rho),
                "p_value": float(p_rho),
                "n": int(len(df_valid)),
            },
        },
        "per_window": [
            {
                "start": str(r["start"].date()),
                "end": str(r["end"].date()),
                "sp_top_period_months": float(r["sp_top_period_months"]),
                "vol_12m": float(r["vol_12m"]),
                "abs_return_mean": float(r["abs_return_mean"]),
                "period_bin": str(r["period_bin"]).replace("\n", " "),
            }
            for _, r in df.iterrows()
        ],
    }
    out_path = RESULTS_DIR / "30_cycle_volatility.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()