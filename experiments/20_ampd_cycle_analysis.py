"""
Experiment 20: AMPD cycle analysis on 8 financial/economic series.

Uses AMPD as a stand-alone analysis tool (not just as TimesNet's input) to find cycles
in each of 8 financial/economic series. Outputs:

1. Spectrum per series — periodogram (frequency vs amplitude) with AMPD top-K periods marked
2. Cross-series comparison — heatmap of which periods appear in which series
3. Time evolution — rolling AMPD on a 30-year window shows how dominant periods shift over time
4. Theoretical cycle matching — does AMPD find Kitchin/Juglar/Kuznets/Kondratiev in monthly data?

Series analyzed (all from panel_monthly.csv):
- SP500_Return (S&P 500 monthly log-returns, 1871-2023)
- CPI (CPIAUCSL, 1947-2025)
- FedFunds (Effective Fed Funds Rate, 1954-2025)
- M2_Growth (M2 YoY growth, 1960-2025)
- Copper_Return (1992-2025)
- Oil_WTI_Return (1986-2025)
- Wheat_Return (1992-2025)
- DXY_Return (2006-2025)
"""
import sys
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Use TrueType fonts in PDF (avoids LaTeX font substitution)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ampd import AMPD

warnings.filterwarnings("ignore")

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ============================================================
# Series to analyze
# ============================================================

SERIES_CONFIG = {
    "SP500_Return": {"name": "S&P 500 returns", "color": "#1f77b4"},
    "CPI": {"name": "CPI (level)", "color": "#ff7f0e"},
    "FedFunds": {"name": "Fed Funds Rate", "color": "#2ca02c"},
    "M2_Growth": {"name": "M2 growth (YoY)", "color": "#d62728"},
    "Copper_Return": {"name": "Copper returns", "color": "#9467bd"},
    "Oil_WTI_Return": {"name": "Oil WTI returns", "color": "#8c564b"},
    "Wheat_Return": {"name": "Wheat returns", "color": "#e377c2"},
    "DXY_Return": {"name": "DXY (USD) returns", "color": "#7f7f7f"},
}

# Rolling window config for time evolution
ROLLING_WINDOW = 360  # months = 30 years
ROLLING_STEP = 60     # months = 5 years

# Theoretical cycle periods (months)
THEORETICAL_CYCLES = {
    "Kitchin (3-5y)": (36, 60),
    "Juglar (7-11y)": (84, 132),
    "Kuznets (15-25y)": (180, 300),
    "Kondratiev (40-60y)": (480, 720),
}


def ampd_top_k(series, top_k=5, max_period=None, min_period=6):
    """Run AMPD and return top-K periods with amplitudes."""
    if max_period is None:
        max_period = min(360, len(series) // 2)
    amp = AMPD(top_k=top_k, max_period=max_period, min_period=min_period)
    periods = amp.fit_discover(series)
    return periods


def rolling_ampd(series, window, step, top_k=3, min_period=12):
    """Rolling AMPD: for each window starting at i, find top-K periods."""
    n = len(series)
    results = []
    for start in range(0, n - window + 1, step):
        end = start + window
        win_series = series[start:end]
        if np.std(win_series) < 1e-8:
            continue
        # Detrend the window
        win_detrend = win_series - win_series.mean()
        try:
            periods = ampd_top_k(win_detrend, top_k=top_k, max_period=window // 2,
                                 min_period=min_period)
            results.append({
                "start_idx": start,
                "end_idx": end,
                "periods_months": periods.tolist() if hasattr(periods, "tolist") else list(periods),
                "periods_years": [p / 12 for p in periods],
            })
        except Exception as e:
            print(f"  [skip window {start}:{end}] {e}")
    return results


def main():
    print("=" * 70)
    print(" Experiment 20: AMPD cycle analysis on financial/economic series")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"[panel] shape={panel.shape}, range={panel.index[0].date()} to {panel.index[-1].date()}")

    # ============================================================
    # 1. Spectrum per series (top-K AMPD periods)
    # ============================================================
    print("\n[1] AMPD top-K periods per series")
    series_periods = {}
    for col, cfg in SERIES_CONFIG.items():
        if col not in panel.columns:
            print(f"  [{col}] NOT IN PANEL")
            continue
        s = panel[col].dropna().values.astype(np.float32)
        n_obs = len(s)
        if n_obs < 60:
            print(f"  [{col}] too few obs ({n_obs}), skip")
            continue
        s_detrend = s - s.mean()
        periods = ampd_top_k(s_detrend, top_k=5, max_period=min(360, n_obs // 2),
                             min_period=6)
        periods_years = [p / 12 for p in periods]
        series_periods[col] = {
            "n_obs": n_obs,
            "first_date": panel[col].dropna().index[0].date().isoformat(),
            "last_date": panel[col].dropna().index[-1].date().isoformat(),
            "ampd_periods_months": [round(p, 1) for p in periods],
            "ampd_periods_years": [round(p / 12, 2) for p in periods],
        }
        print(f"  [{col:<18}] n={n_obs:>4} ({series_periods[col]['first_date']} to "
              f"{series_periods[col]['last_date']}): top periods = "
              f"{series_periods[col]['ampd_periods_years']} years")

    # ============================================================
    # 2. Cross-series comparison heatmap
    # ============================================================
    print("\n[2] Cross-series comparison: which periods appear in multiple series?")
    all_periods = []
    for col, d in series_periods.items():
        all_periods.extend(d["ampd_periods_months"])
    unique_periods = sorted(set([round(p, 0) for p in all_periods]))
    print(f"  Unique periods across all series: {len(unique_periods)} bins")

    # Build heatmap: series x period bins
    n_series = len(series_periods)
    n_periods = len(unique_periods)
    heatmap = np.zeros((n_series, n_periods))
    period_idx = {p: i for i, p in enumerate(unique_periods)}
    for s_i, (col, d) in enumerate(series_periods.items()):
        for p in d["ampd_periods_months"]:
            p_bin = round(p, 0)
            # Find nearest bin
            nearest = min(unique_periods, key=lambda x: abs(x - p_bin))
            heatmap[s_i, period_idx[nearest]] += 1

    # Save heatmap data
    heatmap_data = {
        "series": list(series_periods.keys()),
        "period_bins_months": unique_periods,
        "counts": heatmap.tolist(),
    }

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=heatmap.max())
    ax.set_xticks(range(0, n_periods, max(1, n_periods // 20)))
    ax.set_xticklabels([f"{unique_periods[i]:.0f}mo\n({unique_periods[i]/12:.1f}y)"
                          for i in range(0, n_periods, max(1, n_periods // 20))],
                        rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_series))
    ax.set_yticklabels([f"{c}\n({series_periods[c]['n_obs']} obs)"
                          for c in series_periods.keys()], fontsize=9)
    ax.set_xlabel("Period (months / years)")
    ax.set_ylabel("Series")
    ax.set_title("AMPD top-K periods across financial/economic series\n"
                  "(darker = period present in more series)")
    plt.colorbar(im, ax=ax, label="Count of top-K membership")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_ampd_cross_series_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "20_ampd_cross_series_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / '20_ampd_cross_series_heatmap.png'}")

    # ============================================================
    # 3. Spectrum per series (periodogram)
    # ============================================================
    print("\n[3] Spectrum per series (periodogram with AMPD top-K marked)")
    n_cols = 4
    n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for s_i, (col, d) in enumerate(series_periods.items()):
        row, ax_col = s_i // n_cols, s_i % n_cols
        ax = axes[row, ax_col]
        s = panel[col].dropna().values.astype(np.float32)
        s_detrend = s - s.mean()
        n_obs = len(s_detrend)

        # Periodogram via FFT
        fft_vals = np.fft.rfft(s_detrend, n=n_obs * 4)
        amp = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_obs * 4, d=1.0)
        periods_axis = 1.0 / np.where(freqs > 0, freqs, np.inf)
        # Filter to [6, 360]
        mask = (periods_axis >= 6) & (periods_axis <= min(360, n_obs // 2))
        periods_plot = periods_axis[mask]
        amp_plot = amp[mask]
        # Normalize
        amp_plot = amp_plot / amp_plot.max()

        ax.semilogy(periods_plot / 12, amp_plot, color=SERIES_CONFIG[col]["color"], alpha=0.7)
        for p in d["ampd_periods_months"]:
            ax.axvline(p / 12, color="red", alpha=0.5, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"{col}\n({d['n_obs']} obs)", fontsize=10)
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0.5, 35)

    # Hide unused subplots
    for s_i in range(n_series, n_rows * n_cols):
        axes[s_i // n_cols, s_i % n_cols].axis("off")

    plt.suptitle("AMPD Spectrum Analysis on Financial/Economic Series", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_ampd_spectra.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "20_ampd_spectra.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / '20_ampd_spectra.png'}")

    # ============================================================
    # 4. Time evolution (rolling AMPD on long series)
    # ============================================================
    print("\n[4] Time evolution of dominant periods (rolling AMPD, 30y window)")
    evolution = {}
    for col in ["SP500_Return", "M2_Growth", "FedFunds", "CPI"]:
        if col not in panel.columns:
            continue
        s = panel[col].dropna().values.astype(np.float32)
        if len(s) < ROLLING_WINDOW + 12:
            continue
        rolling = rolling_ampd(s, window=ROLLING_WINDOW, step=ROLLING_STEP,
                                top_k=3, min_period=12)
        # Convert start_idx to date
        s_dates = panel[col].dropna().index
        for r in rolling:
            r["start_date"] = s_dates[r["start_idx"]].date().isoformat()
            r["end_date"] = s_dates[min(r["end_idx"] - 1, len(s_dates) - 1)].date().isoformat()
        evolution[col] = rolling
        print(f"  [{col:<14}] {len(rolling)} rolling windows analyzed")

    # Plot evolution
    fig, axes = plt.subplots(len(evolution), 1, figsize=(14, 3 * len(evolution)),
                              sharex=True)
    if len(evolution) == 1:
        axes = [axes]
    theory_colors = {"Kitchin": "#2ca02c", "Juglar": "#1f77b4",
                     "Kuznets": "#d62728", "Kondratiev": "#9467bd"}
    for ax, (col, rolling) in zip(axes, evolution.items()):
        for r in rolling:
            mid_date = pd.Timestamp(r["start_date"]) + (pd.Timestamp(r["end_date"]) - pd.Timestamp(r["start_date"])) / 2
            # Plot dominant period as a point
            for j, p in enumerate(r["periods_years"]):
                color = "red" if j == 0 else "orange" if j == 1 else "yellow"
                ax.scatter(mid_date, p, color=color, s=80, alpha=0.7, edgecolor="black")
        # Theoretical cycle bands
        ax.axhspan(3, 5, alpha=0.15, color=theory_colors["Kitchin"], label="Kitchin (3-5y)")
        ax.axhspan(7, 11, alpha=0.15, color=theory_colors["Juglar"], label="Juglar (7-11y)")
        ax.axhspan(15, 25, alpha=0.15, color=theory_colors["Kuznets"], label="Kuznets (15-25y)")
        ax.axhspan(40, 60, alpha=0.15, color=theory_colors["Kondratiev"], label="Kondratiev (40-60y)")
        ax.set_yscale("log")
        ax.set_ylim(2, 70)
        ax.set_ylabel("Period (years, log scale)")
        ax.set_title(f"{col} — dominant periods over time (30y rolling window)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("Window midpoint date")
    plt.suptitle("Time Evolution of Dominant Periods — Financial & Economic Series", fontsize=13,
                 fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_ampd_time_evolution.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "20_ampd_time_evolution.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIGURES_DIR / '20_ampd_time_evolution.png'}")

    # ============================================================
    # Save results JSON
    # ============================================================
    out = {
        "config": {
            "rolling_window_months": ROLLING_WINDOW,
            "rolling_step_months": ROLLING_STEP,
            "theoretical_cycles_months": THEORETICAL_CYCLES,
            "series_analyzed": list(SERIES_CONFIG.keys()),
        },
        "ampd_per_series": series_periods,
        "heatmap": heatmap_data,
        "time_evolution": evolution,
    }
    out_path = RESULTS_DIR / "20_ampd_cycle_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # ============================================================
    # Markdown summary
    # ============================================================
    md = ["# AMPD Cycle Analysis on Financial/Economic Series", ""]
    md.append(f"**Series analyzed**: {len(series_periods)}")
    md.append("")
    md.append("| Series | n_obs | Range | Top periods (years) |")
    md.append("|---|---|---|---|")
    for col, d in series_periods.items():
        periods_str = ", ".join([f"{p:.1f}" for p in d["ampd_periods_years"]])
        md.append(f"| {col} | {d['n_obs']} | {d['first_date']} to {d['last_date']} | {periods_str} |")
    md.append("")
    md.append("## Theoretical cycle matching")
    md.append("")
    md.append("Expected from economic theory:")
    md.append("- **Kitchin (3-5y)**: short inventory cycle")
    md.append("- **Juglar (7-11y)**: fixed-investment cycle")
    md.append("- **Kuznets (15-25y)**: infrastructure/swing cycle")
    md.append("- **Kondratiev (40-60y)**: long-wave cycle")
    md.append("")
    md.append("AMPD-detected on monthly data (all series, top-K = 5):")
    md.append("")
    for col, d in series_periods.items():
        matched = []
        for cycle_name, (lo, hi) in THEORETICAL_CYCLES.items():
            cycle_name_short = cycle_name.split(" ")[0]
            found = [p for p in d["ampd_periods_years"] if lo / 12 <= p <= hi / 12]
            if found:
                matched.append(f"{cycle_name_short}={found[0]:.1f}y")
        if matched:
            md.append(f"- **{col}**: {', '.join(matched)}")
        else:
            md.append(f"- **{col}**: no theoretical cycles detected in top-5")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `results/20_ampd_cycle_analysis.json` — full results")
    md.append("- `figures/20_ampd_cross_series_heatmap.png` — cross-series heatmap")
    md.append("- `figures/20_ampd_spectra.png` — per-series periodogram with AMPD top-K")
    md.append("- `figures/20_ampd_time_evolution.png` — rolling AMPD dominant periods over time")
    md_path = RESULTS_DIR / "20_ampd_cycle_analysis_summary.md"
    md_path.write_text("\n".join(md))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
