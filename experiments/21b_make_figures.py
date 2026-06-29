"""
Experiment 21b: Generate 4 polish figures from exp 21 results.

Reads:
- results/21_extended_metrics.json (metrics, calibration, DM test)
- results/21_pit_values.npz (PIT arrays)
- results/21_fan_chart_data.npz (samples, truth, preds for last fold)

Produces:
- figures/21_calibration.png: reliability diagram (predicted vs empirical coverage)
- figures/21_pit_histogram.png: PIT histogram (per horizon, both models)
- figures/21_fan_chart.png: P10/P50/P90 forecast fan with truth (h=12 fold 4)
- figures/21_metric_heatmap.png: model x metric heatmap with sign-correct coloring
"""
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Use TrueType fonts in PDF (avoids LaTeX font substitution)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODELS = ["TimesNet", "TimesNet_GMM"]
MODEL_LABELS = {"TimesNet": "TimesNet", "TimesNet_GMM": "TimesNet + GMM"}
COLORS = {"TimesNet": "#888888", "TimesNet_GMM": "#1f77b4"}

HORIZONS = [1, 3, 6, 12]
H_LABELS = [f"h={h}" for h in HORIZONS]


def fig_calibration(metrics):
    """Reliability diagram: predicted coverage (x) vs empirical coverage (y).

    For a well-calibrated model, points lie on the y=x diagonal.
    Over-coverage: y > x (intervals too wide).
    Under-coverage: y < x (intervals too narrow).
    """
    # Use h=1 as the showcase horizon
    h = 1
    cal = metrics[f"h{h}"]
    # Predicted coverage levels (Winkler 1-alpha)
    # For nominal 0.5, 0.8, 0.9, 0.95, predicted = 0.5, 0.8, 0.9, 0.95
    # Plus a few from pinball: we can derive coverage from pinball
    # For simplicity, use the 4 winkler levels
    levels = [0.50, 0.80, 0.90, 0.95]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], [0.0, 1.0], alpha=0.0)  # no-op

    for model in MODELS:
        emp = [cal[model]["coverage"][f"coverage_{lev:.2f}"] for lev in levels]
        ax.plot(levels, emp, "o-", color=COLORS[model], label=MODEL_LABELS[model],
                markersize=10, linewidth=2)

    ax.set_xlabel("Nominal coverage (predicted)", fontsize=12)
    ax.set_ylabel("Empirical coverage (observed)", fontsize=12)
    ax.set_title(f"Reliability diagram (horizon h={h})", fontsize=13)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    ax.text(0.05, 0.95, "Above diagonal: over-coverage\nBelow diagonal: under-coverage",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out = FIG_DIR / "21_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_pit_histogram(pit_arrays):
    """PIT histogram per horizon (4 subplots), with both models overlaid.

    A well-calibrated model has a uniform PIT distribution.
    U-shape: under-dispersion. Inverted-U: over-dispersion.
    Skewed: bias.
    """
    n_bins = 20
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for i, h in enumerate(HORIZONS):
        ax = axes[i]
        for model in MODELS:
            key = f"pit_h{h}_{model}"
            if key not in pit_arrays:
                continue
            pit = pit_arrays[key]
            ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
                    alpha=0.5, color=COLORS[model], label=MODEL_LABELS[model],
                    edgecolor="black", linewidth=0.5)
        # Uniform reference
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Uniform (ideal)")
        ax.set_title(f"h = {h}", fontsize=12)
        ax.set_xlabel("PIT value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("PIT histograms: probability integral transform\n"
                 "(uniform = perfect calibration)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "21_pit_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_fan_chart(fan_data):
    """Forecast fan chart: P10/P50/P90 with truth overlay (h=12, last fold).

    Shows the predictive distribution and how it compares to actuals.
    """
    h = 12
    models_to_show = ["TimesNet", "TimesNet_GMM"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, m in zip(axes, models_to_show):
        truth = fan_data.get(f"h{h}_{m}_truth")
        samples = fan_data.get(f"h{h}_{m}_samples")
        preds = fan_data.get(f"h{h}_{m}_preds")
        if truth is None or samples is None:
            continue
        # Use first 40 test points for visual clarity
        n_show = min(40, truth.shape[0])
        truth = truth[:n_show]
        preds = preds[:n_show]
        samples = samples[:, :n_show, :]

        x = np.arange(n_show)
        p10 = np.quantile(samples, 0.10, axis=0)[:, 0]  # use h=0
        p90 = np.quantile(samples, 0.90, axis=0)[:, 0]
        p25 = np.quantile(samples, 0.25, axis=0)[:, 0]
        p75 = np.quantile(samples, 0.75, axis=0)[:, 0]
        median = np.median(samples, axis=0)[:, 0]
        truth_1 = truth[:, 0]
        pred_1 = preds[:, 0]

        # Fan: 80% interval (P10-P90) in light, 50% interval (P25-P75) in darker
        ax.fill_between(x, p10, p90, color=COLORS[m], alpha=0.2, label="80% interval (P10-P90)")
        ax.fill_between(x, p25, p75, color=COLORS[m], alpha=0.35, label="50% interval (P25-P75)")
        ax.plot(x, median, color=COLORS[m], linewidth=1.5, label="Predictive median")
        ax.plot(x, truth_1, "k-", linewidth=1.2, alpha=0.7, label="Truth")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

        ax.set_title(f"{MODEL_LABELS[m]} (h={h}, first step)", fontsize=12)
        ax.set_xlabel("Test point index", fontsize=10)
        ax.set_ylabel("S&P 500 monthly return", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Forecast fan chart (h={h}, first forecast step)\n"
                 "Predictive intervals from 1500 GMM samples × 3 seeds", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "21_fan_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_metric_heatmap(metrics):
    """Model x metric heatmap with sign-correct coloring.

    Green = lower is better (MAE, MASE, CRPS, Pinball, Winkler).
    Coverage gap |emp - nom| — lower is better.
    DM p-value — lower is more significant.
    """
    # Pick 8 representative metrics
    metric_specs = [
        ("MAE", "mae", "min"),
        ("MASE", "mase", "min"),
        ("CRPS", "crps", "min"),
        ("Pinball 0.05", "pinball_0.05", "min"),
        ("Pinball 0.5", "pinball_0.50", "min"),
        ("Pinball 0.95", "pinball_0.95", "min"),
        ("Winkler 0.90", "winkler_0.90", "min"),
        ("Coverage 0.90", "coverage_0.90", "target_0.90"),
    ]

    n_metrics = len(metric_specs)
    n_models = len(MODELS)
    # For each model, value = mean over horizons
    matrix = np.zeros((n_metrics, n_models))
    for i, (name, key, direction) in enumerate(metric_specs):
        for j, m in enumerate(MODELS):
            vals = []
            for h in HORIZONS:
                v = metrics[f"h{h}"][m]
                if key.startswith("pinball"):
                    val = v["pinball"][key]
                elif key.startswith("winkler"):
                    val = v["winkler"][key]
                elif key.startswith("coverage"):
                    val = v["coverage"][key]
                else:
                    val = v[key]
                vals.append(val)
            mean_v = np.mean(vals)
            if direction == "target_0.90":
                # Use distance to 0.90
                matrix[i, j] = abs(mean_v - 0.90)
            else:
                matrix[i, j] = mean_v

    # Per-row normalization for color (each metric has its own scale)
    row_max = matrix.max(axis=1, keepdims=True)
    row_min = matrix.min(axis=1, keepdims=True)
    norm = (matrix - row_min) / (row_max - row_min + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha="right", fontsize=11)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([s[0] for s in metric_specs], fontsize=11)
    ax.set_title("Model × metric comparison (mean over h=1,3,6,12)\n"
                 "Green = better, Red = worse (per-metric row-normalized)", fontsize=12)

    # Annotate with actual values
    for i in range(n_metrics):
        for j in range(n_models):
            name, key, direction = metric_specs[i]
            v = matrix[i, j]
            if direction == "target_0.90":
                txt = f"±{v:.3f}"
            else:
                txt = f"{v:.4f}"
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Normalized value (0=best, 1=worst)", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Best", "", "Worst"])

    fig.tight_layout()
    out = FIG_DIR / "21_metric_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_dm_pvalues(dm_results):
    """Bar chart of DM test p-values per horizon with significance bands."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pvals = [dm_results[f"h{h}"]["p_value"] for h in HORIZONS]
    dm_stats = [dm_results[f"h{h}"]["dm_stat"] for h in HORIZONS]

    # Color: green if significant (p<0.05), yellow if p<0.10, gray if ns
    colors = []
    for p in pvals:
        if p < 0.01:
            colors.append("#2ca02c")  # dark green
        elif p < 0.05:
            colors.append("#7fc97f")  # light green
        elif p < 0.10:
            colors.append("#ffff66")  # yellow
        else:
            colors.append("#cccccc")  # gray

    bars = ax.bar([f"h={h}" for h in HORIZONS], pvals, color=colors, edgecolor="black")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05 (5% significance)")
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="p=0.10 (10% marginal)")
    ax.set_ylabel("DM test p-value", fontsize=11)
    ax.set_xlabel("Forecast horizon", fontsize=11)
    ax.set_title("Diebold-Mariano test: TimesNet+GMM vs TimesNet\n"
                 "(lower p-value = GMM gain is statistically significant)", fontsize=12)
    ax.set_ylim(0, max(0.25, max(pvals) * 1.2))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate with DM stat and p-value
    for i, (b, p, d) in enumerate(zip(bars, pvals, dm_stats)):
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else " ns"))
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"DM={d:.2f}\np={p:.3f}{sig}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "21_dm_pvalues.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("=" * 70)
    print(" Experiment 21b: Generate polish figures from exp 21 results")
    print("=" * 70)

    with open(RESULTS_DIR / "21_extended_metrics.json") as f:
        data = json.load(f)
    metrics = data["metrics"]
    dm = data["dm_test"]

    pit_npz = np.load(RESULTS_DIR / "21_pit_values.npz")
    pit_arrays = {k: pit_npz[k] for k in pit_npz.files}

    fan_npz = np.load(RESULTS_DIR / "21_fan_chart_data.npz")
    fan_data = {k: fan_npz[k] for k in fan_npz.files}

    fig_calibration(metrics)
    fig_pit_histogram(pit_arrays)
    fig_fan_chart(fan_data)
    fig_metric_heatmap(metrics)
    fig_dm_pvalues(dm)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
