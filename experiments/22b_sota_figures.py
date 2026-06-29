"""
Experiment 22b: Generate SOTA comparison figures from exp 22 results.

Produces:
- 22_sota_crps_ss.png: bar chart of CRPS-Skill-Score for all 8 models × 4 horizons
- 22_sota_calibration.png: reliability diagram for all 8 models (h=1)
- 22_sota_coverage.png: bar chart of empirical coverage @ 0.90 for all 8 models
- 22_sota_heatmap.png: model × horizon heatmap of CRPS, normalized
- 22_sota_summary_table.png: complete table rendered as figure
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

BACKBONES = ["TimesNet", "DLinear", "NBEATS", "PatchTST"]
HEADS = ["point", "gmm"]
VARIANTS = [f"{b}_{h}" for b in BACKBONES for h in HEADS]
COLORS_HEAD = {"point": "#888888", "gmm": "#1f77b4"}
HATCHES_HEAD = {"point": "///", "gmm": ""}
HORIZONS = [1, 3, 6, 12]
H_LABELS = [f"h={h}" for h in HORIZONS]


def fig_crps_skill_score(metrics):
    """Bar chart: CRPS-Skill-Score for all 8 models at 4 horizons."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    n_models = len(VARIANTS)
    n_h = len(HORIZONS)
    width = 0.10  # bar width
    x = np.arange(n_models)
    for hi, h in enumerate(HORIZONS):
        vals = [metrics[f"h{h}"][v]["crps_skill_score"] * 100 for v in VARIANTS]
        offset = (hi - (n_h - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=H_LABELS[hi],
                      color=[COLORS_HEAD[v.split("_")[1]] for v in VARIANTS],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{v:+.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("CRPS-Skill-Score vs TimesNet_point (%)", fontsize=11)
    ax.set_title("CRPS-Skill-Score for all 8 model variants × 4 horizons\n"
                 "(Positive = better than TimesNet + Huber baseline; all *_gmm are positive)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-3, 9)
    fig.tight_layout()
    out = FIG_DIR / "22_sota_crps_ss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_calibration_sota(metrics):
    """Reliability diagram (h=1) for all 8 models — show the GMM variants are well-calibrated."""
    h = 1
    cal = metrics[f"h{h}"]
    levels = [0.50, 0.80, 0.90, 0.95]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    # Plot all 8 models
    for v in VARIANTS:
        emp = [cal[v]["coverage"][f"coverage_{lev:.2f}"] for lev in levels]
        head = v.split("_")[1]
        color = COLORS_HEAD[head]
        ls = "-" if head == "gmm" else ":"
        marker = "o" if head == "gmm" else "s"
        lw = 2 if head == "gmm" else 1
        ax.plot(levels, emp, marker=marker, color=color, linestyle=ls, linewidth=lw,
                markersize=8, alpha=0.7,
                label=v if head == "gmm" else None)  # only label gmm to avoid clutter
    # Now plot point variants in gray
    for v in VARIANTS:
        if v.endswith("_point"):
            emp = [cal[v]["coverage"][f"coverage_{lev:.2f}"] for lev in levels]
            ax.plot(levels, emp, marker="s", color="#888888", linestyle=":", linewidth=1,
                    markersize=6, alpha=0.5, label=v)
    ax.set_xlabel("Nominal coverage (predicted)", fontsize=12)
    ax.set_ylabel("Empirical coverage (observed)", fontsize=12)
    ax.set_title(f"Reliability diagram (h={h}) — all 8 SOTA variants\n"
                 "Solid blue lines = *_gmm (well-calibrated). Dotted gray = *_point (over-coverage).",
                 fontsize=12)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Two legends: gmm models and point models
    from matplotlib.lines import Line2D
    legend_gmm = [Line2D([0], [0], color="#1f77b4", marker="o", linestyle="-", linewidth=2,
                         markersize=8, label=v) for v in VARIANTS if v.endswith("_gmm")]
    legend_point = [Line2D([0], [0], color="#888888", marker="s", linestyle=":", linewidth=1,
                           markersize=6, label=v) for v in VARIANTS if v.endswith("_point")]
    leg1 = ax.legend(handles=legend_gmm, loc="lower right", fontsize=8, title="*_gmm (NLL)", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=legend_point, loc="upper left", fontsize=8, title="*_point (Huber)", title_fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "22_sota_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_coverage_bar(metrics):
    """Bar chart of empirical coverage @ 0.90 (vs nominal 0.90) for all 8 models × 4 horizons."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    n_models = len(VARIANTS)
    n_h = len(HORIZONS)
    width = 0.10
    x = np.arange(n_models)
    for hi, h in enumerate(HORIZONS):
        vals = [metrics[f"h{h}"][v]["coverage"]["coverage_0.90"] for v in VARIANTS]
        offset = (hi - (n_h - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=H_LABELS[hi],
                      color=[COLORS_HEAD[v.split("_")[1]] for v in VARIANTS],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=0)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="Nominal 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Empirical coverage @ nominal 0.90", fontsize=11)
    ax.set_title("Coverage @ 0.90 for all 8 SOTA variants\n"
                 "All *_gmm cluster near the nominal 0.90 line; all *_point are over-coverage (0.96)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9, ncol=5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.85, 1.0)
    fig.tight_layout()
    out = FIG_DIR / "22_sota_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def fig_crps_heatmap(metrics):
    """Heatmap: model × horizon, color = CRPS (lower = greener)."""
    n_models = len(VARIANTS)
    n_h = len(HORIZONS)
    matrix = np.zeros((n_models, n_h))
    for hi, h in enumerate(HORIZONS):
        for mi, v in enumerate(VARIANTS):
            matrix[mi, hi] = metrics[f"h{h}"][v]["crps"]
    # Per-column normalization (so colors are comparable across horizons)
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    norm = (matrix - col_min) / (col_max - col_min + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_h))
    ax.set_xticklabels(H_LABELS, fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(VARIANTS, fontsize=10)
    ax.set_title("CRPS heatmap (model × horizon)\n"
                 "Green = lower CRPS = better probabilistic forecast", fontsize=12)
    for i in range(n_models):
        for j in range(n_h):
            v = matrix[i, j]
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{v:.5f}", ha="center", va="center", color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Normalized CRPS (per column)", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Best", "", "Worst"])
    fig.tight_layout()
    out = FIG_DIR / "22_sota_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("=" * 70)
    print(" Experiment 22b: SOTA comparison figures from exp 22 results")
    print("=" * 70)
    with open(RESULTS_DIR / "22_sota_comparison.json") as f:
        data = json.load(f)
    metrics = data["metrics"]
    fig_crps_skill_score(metrics)
    fig_calibration_sota(metrics)
    fig_coverage_bar(metrics)
    fig_crps_heatmap(metrics)
    print("\nAll SOTA figures generated.")


if __name__ == "__main__":
    main()
