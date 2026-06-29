"""
Generate the 3 new figures for the paper:

1. fig_12variant_heatmap.pdf  — 12-variant CRPS-SS heatmap (the "head dominates" view)
2. fig_gmm_vs_gauss_regime.pdf — GMM vs Gaussian incremental CRPS-SS per regime
3. fig_coverage_12variants.pdf  — 12-variant coverage @ 90%

Reads from results/35_combined_12variants.json and results/35b_regime_with_gauss.json.
Outputs are .pdf (vector) + .png (preview) in figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
with open(ROOT / "results/35_combined_12variants.json") as f:
    combined = json.load(f)
with open(ROOT / "results/35b_regime_with_gauss.json") as f:
    regime = json.load(f)

HORIZONS = [1, 3, 6, 12]
BACKBONES = ["TimesNet", "DLinear", "NBEATS", "PatchTST"]
HEADS = ["point", "gauss", "gmm"]
VARIANTS = [f"{bb}_{h}" for h in HEADS for bb in BACKBONES]
HEAD_LABELS = {"point": "Point (Huber)", "gauss": "Gaussian (NLL)", "gmm": "GMM K=4 (NLL)"}

# Heatmap layout: rows = variants (12), cols = horizons (4)
# Order: all points, then all gauss, then all gmm (grouped by head)
ROW_LABELS = [f"{HEAD_LABELS[h.split('_')[1]]} — {h.split('_')[0]}" for h in VARIANTS]

# ---------- Compute CRPS-SS matrix ----------
ss_matrix = np.zeros((len(VARIANTS), len(HORIZONS)))
for i, variant in enumerate(VARIANTS):
    for j, h in enumerate(HORIZONS):
        v = combined["metrics"][f"h{h}"].get(variant, {})
        ss = v.get("crps_skill_score_vs_TimesNet_point")
        if ss is not None:
            ss_matrix[i, j] = ss * 100  # percentage

# Color: red=negative, white=zero, blue=positive
# Use diverging colormap centered at 0
vmax = max(abs(ss_matrix.min()), abs(ss_matrix.max()))
vlim = max(vmax, 5.0)


def heatmap(ax, matrix, row_labels, col_labels, title, vlim=5.0):
    """Draw heatmap with red-white-blue diverging colormap, head-row separators."""
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"h={h}" for h in col_labels], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if abs(v) > vlim * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    # Horizontal separators between head groups (after row 3 and row 7)
    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="black", linewidth=1.2)
    ax.set_title(title, fontsize=11)
    return im


# ============================================================
# Figure 1: 12-variant heatmap (the centerpiece)
# Designed for two-column display (figure* in ACM format).
# ============================================================
# Shorter row labels for the heatmap: just "Backbone (Head)"
SHORT_LABELS = []
HEAD_SHORT = {"point": "P", "gauss": "G", "gmm": "M"}
BB_SHORT = {"TimesNet": "TimesNet", "DLinear": "DLinear", "NBEATS": "N-BEATS", "PatchTST": "PatchTST"}
for h in HEADS:
    for bb in BACKBONES:
        SHORT_LABELS.append(f"{BB_SHORT[bb]} ({HEAD_SHORT[h]})")

fig, ax = plt.subplots(figsize=(7.0, 5.5))
heatmap(ax, ss_matrix, SHORT_LABELS, HORIZONS,
        "CRPS-Skill-Score vs TimesNet$_{\\mathrm{point}}$ (%): "
        "the head dominates the backbone", vlim=vlim)
cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("CRPS-Skill-Score (%)", fontsize=9)
# Head group labels: position them just past the heatmap (at x=4 in data coords,
# well within the 0..Ncols-1 axis). Use a single bracket-style label per group.
# Place them between the last heatmap column (x=3) and the colorbar (x≈4).
from matplotlib.patches import FancyBboxPatch
# Add a small text annotation just to the right of column 3
for y_center, label in [(1.0, "Point"), (5.0, "Gaussian"), (9.0, "GMM")]:
    ax.annotate(label, xy=(3.7, y_center), xytext=(3.85, y_center),
                ha="left", va="center", fontsize=9, fontweight="bold",
                rotation=90, annotation_clip=False)
plt.tight_layout()
out = FIG_DIR / "fig_12variant_heatmap.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")


# ============================================================
# Figure 2: GMM-vs-Gaussian incremental per regime
# ============================================================
regime_data = regime["per_regime_per_horizon"]
regime_order = ["high_vol_1970s", "dotcom_bust", "2008_gfc", "secular_bull", "covid_crash"]
regime_labels = {
    "high_vol_1970s": "1970s\nstagflation",
    "dotcom_bust": "Dot-com\nbust",
    "2008_gfc": "2008\nGFC",
    "secular_bull": "Secular\nbull",
    "covid_crash": "COVID\ncrash",
}

# Build matrix: rows = regimes, cols = horizons
incr_matrix = np.full((len(regime_order), len(HORIZONS)), np.nan)
n_matrix = np.zeros((len(regime_order), len(HORIZONS)), dtype=int)
for i, reg in enumerate(regime_order):
    if reg not in regime_data:
        continue
    for j, h in enumerate(HORIZONS):
        hdata = regime_data[reg]["horizons"].get(f"h{h}", {})
        incr = hdata.get("crps_ss_pct_gmm_vs_gauss")
        n = hdata.get("n_windows", 0)
        if incr is not None:
            incr_matrix[i, j] = incr
            n_matrix[i, j] = n

fig, ax = plt.subplots(figsize=(7.0, 3.6))
# Bar plot, grouped by regime
n_reg = len(regime_order)
n_h = len(HORIZONS)
x = np.arange(n_reg)
bar_width = 0.18
colors = ["#3b6fb6", "#62a0d8", "#f0a868", "#d56a4a"]
for j, h in enumerate(HORIZONS):
    vals = incr_matrix[:, j]
    # Replace nan with 0 for plotting (will be skipped)
    bars = ax.bar(x + (j - 1.5) * bar_width, np.where(np.isnan(vals), 0, vals),
                  bar_width, label=f"h={h}", color=colors[j],
                  edgecolor="black", linewidth=0.4)
    # Annotate non-zero bars
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(x[i] + (j - 1.5) * bar_width, v + (0.3 if v >= 0 else -0.6),
                    f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([regime_labels[r] for r in regime_order], fontsize=8.5)
ax.set_ylabel("CRPS-Skill-Score: GMM vs Gaussian (%)", fontsize=9)
ax.axhline(0, color="black", linewidth=0.6)
ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
ax.set_title("GMM's incremental value over single Gaussian, by regime",
             fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = FIG_DIR / "fig_gmm_vs_gauss_regime.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")


# ============================================================
# Figure 3: 12-variant coverage @ 90%
# ============================================================
# Bar plot: x = horizons, grouped bars per variant
cov_matrix = np.zeros((len(VARIANTS), len(HORIZONS)))
for i, variant in enumerate(VARIANTS):
    for j, h in enumerate(HORIZONS):
        v = combined["metrics"][f"h{h}"].get(variant, {})
        cov = v.get("coverage", {}).get("coverage_0.90")
        if cov is not None:
            cov_matrix[i, j] = cov

# Plot as grouped bar
fig, ax = plt.subplots(figsize=(7.0, 3.8))
n_v = len(VARIANTS)
n_h = len(HORIZONS)
x = np.arange(n_h)
bar_width = 0.08
# Colors: gray for point, blue for gauss, orange for gmm
palette = (["#cccccc"] * 4 + ["#3b6fb6"] * 4 + ["#d56a4a"] * 4)
# Plot 3 groups (one per head), with 4 backbone bars per group
for head_idx, head in enumerate(HEADS):
    for bb_idx, bb in enumerate(BACKBONES):
        i = head_idx * 4 + bb_idx
        vals = cov_matrix[i, :]
        color = {"point": "#cccccc", "gauss": "#3b6fb6", "gmm": "#d56a4a"}[head]
        # Position: each head group spans 1.0 unit, 4 bars of 0.08 width centered
        group_offset = (head_idx - 1) * 1.05
        bar_offset = (bb_idx - 1.5) * bar_width
        ax.bar(x + group_offset + bar_offset, vals, bar_width, color=color,
               edgecolor="black", linewidth=0.3,
               label={"point": "Point (Huber)", "gauss": "Gaussian (NLL)", "gmm": "GMM (NLL)"}[head] if bb_idx == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=9)
ax.set_ylabel("Empirical 90% coverage", fontsize=9)
ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8)
ax.set_ylim(0.85, 1.0)
ax.set_title("Predictive-interval coverage at the 90% nominal level", fontsize=10.5)
# Custom legend: 3 entries (one per head) + nominal line
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor="#cccccc", edgecolor="black", label="Point (Huber)"),
    Patch(facecolor="#3b6fb6", edgecolor="black", label="Gaussian (NLL)"),
    Patch(facecolor="#d56a4a", edgecolor="black", label="GMM (NLL)"),
    Line2D([0], [0], color="red", linestyle="--", label="Nominal 0.90"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2, frameon=True)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = FIG_DIR / "fig_coverage_12variants.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

print("\nAll 3 figures generated.")
