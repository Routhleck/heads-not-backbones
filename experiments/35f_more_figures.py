"""
Generate 2 more figures for the paper:
  4. fig_classical_3heads.pdf — ARIMA/GARCH × 3 heads comparison
  5. fig_pinball_12variants.pdf — Pinball loss at h=1 for 12 variants
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

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load
with open(ROOT / "results/35d_classical_3heads.json") as f:
    classical = json.load(f)
with open(ROOT / "results/35e_pinball_12variants.json") as f:
    pinball = json.load(f)

HORIZONS = [1, 3, 6, 12]


# ============================================================
# Figure 4: classical 3 heads comparison
# ============================================================
# Two panels: ARIMA and GARCH, each shows CRPS-SS for point/gauss/gmm × 4 horizons
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)

baseline_crps = classical["baseline_crps_TimesNet_point"]

for ax_idx, model in enumerate(["ARIMA", "GARCH"]):
    ax = axes[ax_idx]
    heads = ["point", "gauss", "gmm"]
    head_colors = {"point": "#cccccc", "gauss": "#3b6fb6", "gmm": "#d56a4a"}
    head_labels = {"point": "Point (Huber-like)", "gauss": "Gaussian", "gmm": "GMM (K=4)"}
    n_h = len(HORIZONS)
    x = np.arange(n_h)
    bar_width = 0.25
    for h_idx, head in enumerate(heads):
        crps_ss = []
        cov90 = []
        for h in HORIZONS:
            v = classical["results"][model][f"h{h}"].get(head, {})
            ss = v.get("crps_skill_score_vs_TimesNet_point", 0.0) * 100
            cov = v.get("cov90_mean", float("nan"))
            crps_ss.append(ss)
            cov90.append(cov)
        offset = (h_idx - 1) * bar_width
        bars = ax.bar(x + offset, crps_ss, bar_width,
                      color=head_colors[head], label=head_labels[head],
                      edgecolor="black", linewidth=0.3)
        # Annotate values
        for i, v in enumerate(crps_ss):
            ax.text(x[i] + offset, v + (0.3 if v >= 0 else -1.5),
                    f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"{model} (p,q AIC-selected / GARCH(1,1))", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("CRPS-Skill-Score vs TimesNet$_{\\mathrm{point}}$ (%)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if ax_idx == 1:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_ylim(-65, 30)

fig.suptitle("Classical baselines × 3 heads: h-split advantage (short-horizon beats deep; long-horizon does not)",
             fontsize=10, y=1.02)
plt.tight_layout()
out = FIG_DIR / "fig_classical_3heads.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")


# ============================================================
# Figure 5: Pinball loss for 12 variants at h=1
# ============================================================
# Two panels: left tail (P_0.05) and right tail (P_0.95)
# Groups: point rows, gauss rows, gmm rows
HEADS = ["point", "gauss", "gmm"]
BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
VARIANTS = [f"{bb}_{h}" for h in HEADS for bb in BACKBONES]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

# Color by head (point=gray, gaussian=blue, gmm=orange)
head_color = {"point": "#cccccc", "gauss": "#3b6fb6", "gmm": "#d56a4a"}
for ax_idx, q_key in enumerate(["pinball_0.05", "pinball_0.95"]):
    ax = axes[ax_idx]
    q_label = "$\\tau=0.05$ (left tail)" if q_key == "pinball_0.05" else "$\\tau=0.95$ (right tail)"
    ax.set_title(f"Pinball at {q_label} (h=1)", fontsize=10)

    n_v = len(VARIANTS)
    x = np.arange(n_v)
    for i, variant in enumerate(VARIANTS):
        p = pinball["h1"].get(variant, {})
        v = p.get(q_key)
        if v is None:
            continue
        head = variant.rsplit("_", 1)[1]
        ax.bar(x[i], v, color=head_color[head], edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    # X-tick labels show only the backbone name (head labels are at the top)
    backbone_labels = [v.rsplit("_", 1)[0] for v in VARIANTS]
    ax.set_xticklabels(backbone_labels, rotation=30, fontsize=7, ha="right")
    # Group separators
    for sep in [3.5, 7.5]:
        ax.axvline(sep, color="black", linewidth=0.6)
    if ax_idx == 0:
        ax.set_ylabel("Pinball loss (lower is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.5, n_v - 0.5)

# Head group labels in axes coordinates (positioned just above the bars,
# well below the figure title so they don't overlap)
for ax in axes:
    ax.text(0.125, 0.96, "Point head", ha="center", va="top",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))
    ax.text(0.5, 0.96, "Gaussian head", ha="center", va="top",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))
    ax.text(0.875, 0.96, "GMM head", ha="center", va="top",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))

plt.tight_layout()
out = FIG_DIR / "fig_pinball_12variants.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
