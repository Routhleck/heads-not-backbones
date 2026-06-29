"""
Combine 22 (point+gmm) + 34 (gauss) into a 35-style 12-variant JSON, then
regenerate the 3 paper figures with the CURRENT 4-backbone lineup
(TimesNet, DLinear, NBEATS, iTransformer — no PatchTST).

Reads from results/22_sota_comparison.json and results/34_gaussian_head.json.
Outputs:
- results/35_combined_12variants.json  (replaces the old PatchTST-based one)
- figures/fig_12variant_heatmap.pdf
- figures/fig_gmm_vs_gauss_regime.pdf
- figures/fig_coverage_12variants.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "paper/wpn_acm/figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============= Load source data =============
with open(RESULT_DIR / "22_sota_comparison.json") as f:
    sota = json.load(f)
with open(RESULT_DIR / "34_gaussian_head.json") as f:
    gauss = json.load(f)
# Regime data: same source as before (35b_regime_with_gauss.json). We rebuild
# it by aggregating from 22 (point+gmm) + 34 (gauss) per regime — but the
# regime experiment is in a separate script. We'll fall back to reading the
# existing 35b file for regime data (the regime test uses only TimesNet
# backbone, which is the same regardless of 4th-backbone choice).

# 22 has point+gmm cells: TimesNet_point, DLinear_point, NBEATS_point, iTransformer_point,
#                         TimesNet_gmm,   DLinear_gmm,   NBEATS_gmm,   iTransformer_gmm
# 34 has gauss cells:   TimesNet,       DLinear,       NBEATS,       iTransformer

BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
HEADS = ["point", "gauss", "gmm"]
HORIZONS = [1, 3, 6, 12]

combined = {"config": {"source": "22_sota + 34_gaussian"}, "metrics": {}}

for h in HORIZONS:
    hkey = f"h{h}"
    combined["metrics"][hkey] = {}

def _normalize_crps_ss(cell):
    """34 stores CRPS-SS under 'crps_skill_score_vs_TimesNet_point';
    22 stores it under 'crps_skill_score'. Returns decimal (e.g. 0.054 = +5.4%)."""
    return cell.get("crps_skill_score_vs_TimesNet_point",
                    cell.get("crps_skill_score"))

for bb in BACKBONES:
    # point + gmm from 22
    for head in ["point", "gmm"]:
        key = f"{bb}_{head}"
        for h in HORIZONS:
            hkey = f"h{h}"
            if hkey in sota["metrics"] and key in sota["metrics"][hkey]:
                combined["metrics"][hkey][key] = sota["metrics"][hkey][key]
    # gauss from 34
    for h in HORIZONS:
        hkey = f"h{h}"
        if hkey in gauss["metrics"] and bb in gauss["metrics"][hkey]:
            combined["metrics"][hkey][f"{bb}_gauss"] = gauss["metrics"][hkey][bb]

# Sanity check: all 12 cells × 4 horizons should be present
for h in HORIZONS:
    hkey = f"h{h}"
    cells = combined["metrics"][hkey]
    expected = [f"{bb}_{head}" for head in HEADS for bb in BACKBONES]
    missing = [e for e in expected if e not in cells]
    if missing:
        print(f"  WARN h={h}: missing {missing}")

with open(RESULT_DIR / "35_combined_12variants.json", "w") as f:
    json.dump(combined, f, indent=2, default=float)
print(f"Wrote {RESULT_DIR}/35_combined_12variants.json")

# ============= Figure 1: 12-variant CRPS-SS heatmap =============
VARIANTS = [f"{bb}_{head}" for head in HEADS for bb in BACKBONES]
crps_matrix = np.zeros((len(VARIANTS), len(HORIZONS)))
for i, v in enumerate(VARIANTS):
    for j, h in enumerate(HORIZONS):
        d = combined["metrics"][f"h{h}"].get(v, {})
        crps = _normalize_crps_ss(d)
        if crps is None:
            crps = 0.0
        crps_matrix[i, j] = crps * 100  # convert to percent

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(crps_matrix, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=8)
ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=9)
ax.set_yticks(range(len(VARIANTS)))
ax.set_yticklabels(VARIANTS, fontsize=8)
# Group labels on the right side
for j, head in enumerate(HEADS):
    ax.axhline(y=j*4 - 0.5, color="black", linewidth=0.5)
# Annotate each cell
for i in range(len(VARIANTS)):
    for j in range(len(HORIZONS)):
        val = crps_matrix[i, j]
        color = "white" if abs(val) > 4 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.5, color=color)
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("CRPS-Skill-Score vs TimesNet_point (%)", fontsize=9)
ax.set_title("CRPS-Skill-Score: 4 backbones × 3 heads (monthly S&P 500)", fontsize=10.5)
plt.tight_layout()
out = FIG_DIR / "fig_12variant_heatmap.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ============= Figure 2: 12-variant coverage @ 90% (FIXED LAYOUT) =============
# Bug in old layout: bars were positioned with group_offset = (head-1)*1.05,
# which placed point-head bars for horizon h at x ≈ h-1.17 — outside the
# visible plot area (which is [0, 3]). Fix: use a single tight cluster of
# 12 bars per horizon, centered on x=h, with bar_width that fits within
# the 1-unit horizon gap.
cov_matrix = np.zeros((len(VARIANTS), len(HORIZONS)))
for i, v in enumerate(VARIANTS):
    for j, h in enumerate(HORIZONS):
        d = combined["metrics"][f"h{h}"].get(v, {})
        cov = d.get("coverage", {}).get("coverage_0.90")
        if cov is not None:
            cov_matrix[i, j] = cov

fig, ax = plt.subplots(figsize=(7.5, 4.0))
n_v = len(VARIANTS)  # 12
n_h = len(HORIZONS)  # 4
x = np.arange(n_h, dtype=float)
bar_width = 0.075  # 12 * 0.075 = 0.9 < 1 unit horizon gap
palette = (["#cccccc"] * 4 + ["#3b6fb6"] * 4 + ["#d56a4a"] * 4)
# Center all 12 bars on each horizon: bar i offset = (i - 5.5) * bar_width
for i, variant in enumerate(VARIANTS):
    head = variant.split("_")[1]
    color = {"point": "#cccccc", "gauss": "#3b6fb6", "gmm": "#d56a4a"}[head]
    offset = (i - 5.5) * bar_width
    ax.bar(x + offset, cov_matrix[i, :], bar_width, color=color,
           edgecolor="black", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=9)
ax.set_ylabel("Empirical 90% coverage", fontsize=9)
ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8)
ax.set_ylim(0.86, 1.0)
ax.set_title("Predictive-interval coverage at the 90% nominal level", fontsize=10.5)
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

# ============= Figure 5: GMM vs Gauss by regime (uses 35b data, fallback) =============
# The regime test only uses TimesNet backbone, so it's independent of the
# 4th-backbone choice (PatchTST vs iTransformer). Keep using 35b file.
print("Fig 5 (regime) uses 35b_regime_with_gauss.json — unchanged.")