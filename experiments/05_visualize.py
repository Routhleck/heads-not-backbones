"""
Experiment 05: Visualization suite for the paper.

Generates 4 figures:
  Fig 1: FFT spectrum (3 panels: S&P returns 1871, S&P real 1947, multi-asset 2006)
         with top-K periods marked
  Fig 2: Period decomposition of S&P real price (trend + 4y + 20y + 25y + noise)
  Fig 3: Prior-matching heatmap (4 prior sets x N discovered periods)
  Fig 4: Forecast benchmark bar chart (4 horizons x 4 models)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.ampd import AMPD

DATA_PATH = ROOT / "data" / "raw" / "panel_monthly.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_SETS = {
    "None": [],
    "Zhou Jintao": [30 * 12, 20 * 12, 7 * 12],
    "Kondratiev-Juglar": [25 * 12, 10 * 12, 4 * 12],
    "Kuznets": [20 * 12, 10 * 12, 5 * 12],
}


# ============================================================
# Fig 1: Spectrum
# ============================================================

def compute_spectrum(x, n_pad=4):
    """Return (periods, amplitudes) for FFT with n_pad zero-padding."""
    n = len(x)
    n_fft = n * n_pad
    fft_vals = np.fft.rfft(x - np.mean(x), n=n_fft)
    amp = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles per month
    periods = 1.0 / np.where(freqs == 0, np.inf, freqs)
    # Drop DC
    amp = amp[1:]
    periods = periods[1:]
    return periods, amp


def fig1_spectrum(panel, save=True):
    """3-panel FFT spectrum with top-K peaks marked."""
    slices = [
        ("S&P 500 monthly log-returns 1871-2023",
         panel["SP500_Return"].dropna().values),
        ("S&P 500 real price 1947-2023",
         panel["SP500_Real"].dropna().values),
        ("Multi-asset panel 2006-2023\n(SP500+DXY+M2+FedFunds)",
         panel[["SP500_Return", "DXY_Return", "M2_Growth", "FedFunds"]].dropna().mean(axis=1).values),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    for ax, (title, x) in zip(axes, slices):
        periods, amp = compute_spectrum(x, n_pad=4)
        # Cap to 5y..40y for readability
        mask = (periods >= 24) & (periods <= 480)
        p_plot = periods[mask]
        a_plot = amp[mask]
        ax.semilogy(p_plot / 12, a_plot, lw=0.6, color="steelblue", alpha=0.8)
        # Mark AMPD top-3 peaks
        ampd = AMPD(top_k=3, max_period=360, min_period=24)
        top = ampd.fit_discover(x)
        for tp in top:
            ax.axvline(tp / 12, color="crimson", ls="--", lw=1, alpha=0.7)
            ax.text(tp / 12, ax.get_ylim()[1] * 0.5, f"  {tp/12:.1f}y",
                    color="crimson", fontsize=9, rotation=90, va="top")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        ax.set_xlim(2, 40)

    fig.suptitle("Fig 1: FFT amplitude spectrum (4x zero-padded, top-3 AMPD peaks marked)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    if save:
        out = FIG_DIR / "fig1_spectrum.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# Fig 2: Period decomposition
# ============================================================

def fig2_decomposition(panel, save=True):
    """Decompose S&P real price into trend + 3 cycles + noise using FFT pass-band."""
    series = panel["SP500_Real"].dropna()
    # Limit to 1947-2023 (sustained non-NaN range)
    series = series.loc["1947":"2023"]
    x = series.values
    t_index = series.index
    t = np.arange(len(x))

    # Linear trend
    coef = np.polyfit(t, x, 1)
    trend = np.polyval(coef, t)

    # Detrend
    detrended = x - trend

    # FFT to extract cycles around 4y, 20y, 25y
    n_pad = 4
    n = len(detrended)
    n_fft = n * n_pad
    fft_vals = np.fft.rfft(detrended, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    periods_full = 1.0 / np.where(freqs == 0, np.inf, freqs)

    target_periods = [4 * 12, 20 * 12, 25 * 12]  # months
    bandwidth = 6  # ±6 months around each target

    cycles = []
    for tp in target_periods:
        mask = np.abs(periods_full - tp) <= bandwidth
        band_fft = fft_vals * mask
        cycle = np.fft.irfft(band_fft, n=n_fft)[:n]
        cycles.append(cycle)
    cycle_sum = sum(cycles)
    noise = detrended - cycle_sum

    # Plot 4 panels
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(t_index, x, color="black", lw=0.8)
    axes[0].set_title("Original: S&P 500 real price (1947-2023)", fontsize=10)
    axes[0].set_ylabel("Real price (CPI-deflated)")

    axes[1].plot(t_index, trend, color="navy", lw=1.2)
    axes[1].set_title("Linear trend", fontsize=10)
    axes[1].set_ylabel("Trend")

    cycle_labels = ["Cycle ~4 years (Juglar)", "Cycle ~20 years (Kuznets)", "Cycle ~25 years (Kondratiev)"]
    cycle_colors = ["#e74c3c", "#27ae60", "#2980b9"]
    for i, (cyc, lab, col) in enumerate(zip(cycles, cycle_labels, cycle_colors)):
        axes[2 + i].plot(t_index, cyc, color=col, lw=0.7)
        axes[2 + i].set_title(lab, fontsize=10)
        axes[2 + i].set_ylabel("Amplitude")

    axes[4].plot(t_index, noise, color="gray", lw=0.3, alpha=0.7)
    axes[4].set_title("Residual (noise)", fontsize=10)
    axes[4].set_ylabel("Residual")
    axes[4].set_xlabel("Year")

    fig.suptitle("Fig 2: S&P 500 real price = Linear trend + 3 cycles + noise",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    if save:
        out = FIG_DIR / "fig2_decomposition.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# Fig 3: Prior-matching heatmap
# ============================================================

def fig3_heatmap(panel, save=True):
    """4 prior sets × N discovered periods, color = relative error (0 = perfect match)."""
    # Use S&P real price 1947-2023 (longest signal with multiple theory-relevant cycles)
    series = panel["SP500_Real"].dropna()
    series = series.loc["1947":"2023"]
    x = series.values

    ampd = AMPD(top_k=6, max_period=360, min_period=24)
    discovered = ampd.fit_discover(x)  # in months
    discovered_y = discovered / 12.0  # in years

    prior_names = list(PRIOR_SETS.keys())
    n_priors = len(prior_names)
    n_disc = len(discovered_y)

    heat = np.full((n_priors, n_disc), np.nan)
    for i, pname in enumerate(prior_names):
        priors = [p / 12 for p in PRIOR_SETS[pname]]  # in years
        for j in range(n_disc):
            d = discovered_y[j]
            if not priors:
                # None: only check that discovered is valid
                heat[i, j] = 0.0
            else:
                # find closest prior
                rel_errs = [abs(d - p) / p for p in priors]
                heat[i, j] = min(rel_errs) * 100  # in %

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=20)
    ax.set_yticks(range(n_priors))
    ax.set_yticklabels(prior_names)
    ax.set_xticks(range(n_disc))
    ax.set_xticklabels([f"{d:.1f}y" for d in discovered_y], rotation=0)
    ax.set_xlabel("Discovered period (years)")
    ax.set_title("Fig 3: AMPD-discovered periods vs theory prior (rel err %) — S&P real price 1947-2023",
                 fontsize=11)
    for i in range(n_priors):
        for j in range(n_disc):
            v = heat[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if v > 10 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative error to closest prior (%)")
    fig.tight_layout()
    if save:
        out = FIG_DIR / "fig3_prior_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# Fig 4: Forecast benchmark
# ============================================================

def fig4_forecast(panel, save=True):
    """Re-plot results from experiments/04_real_baseline.py as a grouped bar chart."""
    res_path = RESULTS_DIR / "04_real_baseline.json"
    if not res_path.exists():
        print(f"[WARN] {res_path} not found, skipping Fig 4")
        return
    res = json.load(open(res_path))
    horizons = [1, 3, 6, 12]
    models = ["naive", "linear", "arima", "timesnet_lite"]
    model_labels = ["Naive", "Linear", "ARIMA", "TimesBlock (ours)"]
    colors = ["#95a5a6", "#f39c12", "#3498db", "#e74c3c"]

    x = np.arange(len(horizons))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, lab, c) in enumerate(zip(models, model_labels, colors)):
        maes = [res[f"h{h}"][m]["mae"] for h in horizons]
        ax.bar(x + i * width, maes, width, label=lab, color=c, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"h={h}m" for h in horizons])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Fig 4: Walk-forward forecast MAE on S&P 500 monthly log-returns",
                 fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if save:
        out = FIG_DIR / "fig4_forecast.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================



# ============================================================
# Fig 5: Ablation bar chart
# ============================================================

def fig5_ablation(panel, save=True):
    """4 prior sets × 4 horizons: MAE comparison."""
    res_path = RESULTS_DIR / "06_ablation.json"
    if not res_path.exists():
        print(f"[WARN] {res_path} not found, skipping Fig 5")
        return
    res = json.load(open(res_path))
    horizons = [1, 3, 6, 12]
    priors = ["None", "Zhou_Jintao", "Kondratiev_Juglar", "Kuznets"]
    colors = ["#27ae60", "#e74c3c", "#3498db", "#9b59b6"]

    x = np.arange(len(horizons))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (p, c) in enumerate(zip(priors, colors)):
        maes = [res[f"h{h}"][p]["mae"] for h in horizons]
        ax.bar(x + i * width, maes, width, label=p, color=c, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"h={h}m" for h in horizons])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Fig 5: Ablation - prior set effect on S&P 500 log-return forecast MAE",
                 fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9, title="Prior set")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if save:
        out = FIG_DIR / "fig5_ablation.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def main():
    print("=" * 70)
    print(" Experiment 05: Visualization suite")
    print("=" * 70)

    panel = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"[load] panel: {panel.shape}")

    print("\n--- Fig 1: spectrum ---")
    fig1_spectrum(panel)
    print("\n--- Fig 2: decomposition ---")
    fig2_decomposition(panel)
    print("\n--- Fig 3: prior heatmap ---")
    fig3_heatmap(panel)
    print("\n--- Fig 4: forecast benchmark ---")
    fig4_forecast(panel)
    print("\n--- Fig 5: ablation ---")
    fig5_ablation(panel)

    print("\nAll figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
