"""Phase 3.1: Economic-value / tail-risk quantification.

Uses the existing 12-variant pinball losses (per-quantile proper scoring rule)
and central-CI coverage to quantify:

  (A) Tail pinball improvement: the gain in pinball loss at q=0.01, 0.05 across
      heads. This is the *proper scoring rule* for the VaR task — the loss a
      risk manager would actually pay under the Basel FRTB internal-models
      approach.
  (B) Calibration error: |empirical coverage - nominal| at 90% and 95% central
      CI, across heads. A model with lower calibration error has a more
      trustworthy VaR — the bank doesn't need to over- or under-estimate.

We do NOT need to re-train any model — the proper scores are already stored in
results/22_sota_comparison.json and results/34_gaussian_head.json.

Output: results/38_tail_risk_summary.json (and a .txt table).
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
RES = ROOT / "results"

with open(RES / "22_sota_comparison.json") as f:
    sota = json.load(f)
with open(RES / "34_gaussian_head.json") as f:
    gauss = json.load(f)

VARIANTS_SOTA = sota["config"]["variants"]   # 8 = 4 backbones × {point, gmm}
HORIZONS = sota["config"]["horizons"]        # [1, 3, 6, 12]
BACKBONES = gauss["config"]["backbones"]     # [TimesNet, DLinear, NBEATS, PatchTST]
N_FOLDS = 5

QUANTILES_OF_INTEREST = [0.01, 0.05, 0.10, 0.25, 0.50]
COVERAGE_LEVELS = [0.5, 0.8, 0.9, 0.95]

# ---- Build a unified per-(backbone, head, horizon) record
table = {}
for h in HORIZONS:
    table[f"h{h}"] = {}
    for bb in BACKBONES:
        table[f"h{h}"][bb] = {"point": {}, "gauss": {}, "gmm": {}}

        # Point head: from 22_sota_comparison
        var = f"{bb}_point"
        if var in sota["metrics"][f"h{h}"]:
            m = sota["metrics"][f"h{h}"][var]
            table[f"h{h}"][bb]["point"] = {
                "pinball": {q: m["pinball"][f"pinball_{q:.2f}"] for q in QUANTILES_OF_INTEREST},
                "coverage": m["coverage"],
                "crps": m["crps"],
            }

        # Gaussian head: from 34_gaussian_head
        m = gauss["metrics"][f"h{h}"][bb]
        table[f"h{h}"][bb]["gauss"] = {
            "pinball": {q: m["pinball"][f"pinball_{q:.2f}"] for q in QUANTILES_OF_INTEREST},
            "coverage": m["coverage"],
            "crps": m["crps"],
        }

        # GMM head: from 22_sota_comparison
        var = f"{bb}_gmm"
        if var in sota["metrics"][f"h{h}"]:
            m = sota["metrics"][f"h{h}"][var]
            table[f"h{h}"][bb]["gmm"] = {
                "pinball": {q: m["pinball"][f"pinball_{q:.2f}"] for q in QUANTILES_OF_INTEREST},
                "coverage": m["coverage"],
                "crps": m["crps"],
            }

# ---- Headline A: tail pinball improvement ratio at q=0.01 and 0.05
# Improvement ratio = (point - gmm) / point, positive = gmm better
def pinball_improvement(hkey, bb, q):
    p = table[hkey][bb]["point"].get("pinball", {}).get(q)
    g = table[hkey][bb]["gmm"].get("pinball", {}).get(q)
    ga = table[hkey][bb]["gauss"].get("pinball", {}).get(q)
    if p is None or g is None:
        return None
    return {
        "point": p,
        "gauss": ga,
        "gmm": g,
        "ratio_point_to_gmm": (p - g) / p,
        "ratio_gauss_to_gmm": (ga - g) / ga if ga else None,
    }

# ---- Headline B: calibration error (|empirical - nominal|)
def calibration_error(hkey, bb, head):
    cov = table[hkey][bb][head].get("coverage", {})
    out = {}
    for lev in COVERAGE_LEVELS:
        c = cov.get(f"coverage_{lev:.2f}")
        if c is not None:
            out[f"cov_{lev:.2f}_err"] = abs(c - lev)
    return out

# ---- Aggregate over 4 backbones
def mean_over_backbones(fn, hkey):
    vals = [fn(hkey, bb) for bb in BACKBONES]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else None

# ---- Build summary
summary = {
    "config": {
        "horizons": HORIZONS,
        "backbones": BACKBONES,
        "n_folds": N_FOLDS,
        "quantiles_of_interest": QUANTILES_OF_INTEREST,
        "coverage_levels": COVERAGE_LEVELS,
    },
    "tail_pinball": {},       # per horizon
    "calibration": {},
    "headline": {},           # cross-horizon averages
}

for h in HORIZONS:
    hkey = f"h{h}"
    summary["tail_pinball"][hkey] = {}
    for q in [0.01, 0.05]:
        per_bb = {bb: pinball_improvement(hkey, bb, q) for bb in BACKBONES}
        mean_ratio_pg = mean_over_backbones(
            lambda k, bb: pinball_improvement(k, bb, q)["ratio_point_to_gmm"]
            if pinball_improvement(k, bb, q) else None,
            hkey,
        )
        mean_ratio_gg = mean_over_backbones(
            lambda k, bb: pinball_improvement(k, bb, q)["ratio_gauss_to_gmm"]
            if pinball_improvement(k, bb, q) and pinball_improvement(k, bb, q).get("ratio_gauss_to_gmm") is not None
            else None,
            hkey,
        )
        summary["tail_pinball"][hkey][f"q{q}"] = {
            "per_backbone": per_bb,
            "mean_ratio_point_to_gmm": mean_ratio_pg,
            "mean_ratio_gauss_to_gmm": mean_ratio_gg,
        }
    # Calibration
    summary["calibration"][hkey] = {}
    for head in ["point", "gauss", "gmm"]:
        per_bb = {bb: calibration_error(hkey, bb, head) for bb in BACKBONES}
        # Mean abs error at 0.95
        err_95 = mean_over_backbones(
            lambda k, bb: calibration_error(k, bb, head).get("cov_0.95_err"),
            hkey,
        )
        err_90 = mean_over_backbones(
            lambda k, bb: calibration_error(k, bb, head).get("cov_0.90_err"),
            hkey,
        )
        summary["calibration"][hkey][head] = {
            "per_backbone_calibration": per_bb,
            "mean_abs_err_95": err_95,
            "mean_abs_err_90": err_90,
        }

# Cross-horizon headline: mean over h of "gmm beats point at q=0.05"
ratios_05 = [summary["tail_pinball"][f"h{h}"]["q0.05"]["mean_ratio_point_to_gmm"] for h in HORIZONS]
ratios_01 = [summary["tail_pinball"][f"h{h}"]["q0.01"]["mean_ratio_point_to_gmm"] for h in HORIZONS]
ratios_gg_05 = [summary["tail_pinball"][f"h{h}"]["q0.05"]["mean_ratio_gauss_to_gmm"] for h in HORIZONS]
ratios_gg_01 = [summary["tail_pinball"][f"h{h}"]["q0.01"]["mean_ratio_gauss_to_gmm"] for h in HORIZONS]

summary["headline"] = {
    "mean_pinball_improvement_q0.01_point_to_gmm": float(np.mean(ratios_01)),
    "mean_pinball_improvement_q0.05_point_to_gmm": float(np.mean(ratios_05)),
    "mean_pinball_improvement_q0.01_gauss_to_gmm": float(np.mean([r for r in ratios_gg_01 if r is not None])),
    "mean_pinball_improvement_q0.05_gauss_to_gmm": float(np.mean([r for r in ratios_gg_05 if r is not None])),
    "per_h_pinball_improvement_q0.05": {f"h{h}": r for h, r in zip(HORIZONS, ratios_05)},
}

# Save
with open(RES / "38_tail_risk_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print summary table
print("=" * 78)
print("Phase 3.1: Tail Pinball Loss & Calibration Error (12 variants, monthly S&P 500)")
print("=" * 78)

print("\n[A] Pinball loss at q=0.01 (1% VaR proper score)")
print("    Backbone    h=1(point) h=1(gauss) h=1(gmm)   h=12(point) h=12(gauss) h=12(gmm)")
for bb in BACKBONES:
    cells = []
    for h in [1, 12]:
        p = table[f"h{h}"][bb]["point"].get("pinball", {}).get(0.01)
        ga = table[f"h{h}"][bb]["gauss"].get("pinball", {}).get(0.01)
        g = table[f"h{h}"][bb]["gmm"].get("pinball", {}).get(0.01)
        cells.extend([f"{p:.4f}" if p else "-",
                      f"{ga:.4f}" if ga else "-",
                      f"{g:.4f}" if g else "-"])
    print(f"    {bb:12s}  " + "  ".join(f"{c:>10s}" for c in cells))

print("\n[B] Pinball improvement ratio at q=0.01 (positive = GMM wins)")
print(f"    Backbone       h=1 (P→G)  h=1 (G→GMM)  h=12 (P→G)  h=12 (G→GMM)")
for bb in BACKBONES:
    r1_pg = pinball_improvement("h1", bb, 0.01)
    r12_pg = pinball_improvement("h12", bb, 0.01)
    print(f"    {bb:14s}  "
          f"{(r1_pg['ratio_point_to_gmm']*100 if r1_pg else 0):+8.2f}%  "
          f"{(r1_pg['ratio_gauss_to_gmm']*100 if r1_pg and r1_pg['ratio_gauss_to_gmm'] is not None else 0):+10.2f}%  "
          f"{(r12_pg['ratio_point_to_gmm']*100 if r12_pg else 0):+8.2f}%  "
          f"{(r12_pg['ratio_gauss_to_gmm']*100 if r12_pg and r12_pg['ratio_gauss_to_gmm'] is not None else 0):+10.2f}%")

print("\n[C] Pinball improvement ratio at q=0.05")
print(f"    Backbone       h=1 (P→G)  h=1 (G→GMM)  h=12 (P→G)  h=12 (G→GMM)")
for bb in BACKBONES:
    r1_pg = pinball_improvement("h1", bb, 0.05)
    r12_pg = pinball_improvement("h12", bb, 0.05)
    print(f"    {bb:14s}  "
          f"{(r1_pg['ratio_point_to_gmm']*100 if r1_pg else 0):+8.2f}%  "
          f"{(r1_pg['ratio_gauss_to_gmm']*100 if r1_pg and r1_pg['ratio_gauss_to_gmm'] is not None else 0):+10.2f}%  "
          f"{(r12_pg['ratio_point_to_gmm']*100 if r12_pg else 0):+8.2f}%  "
          f"{(r12_pg['ratio_gauss_to_gmm']*100 if r12_pg and r12_pg['ratio_gauss_to_gmm'] is not None else 0):+10.2f}%")

print("\n[D] Calibration error: mean |empirical coverage - nominal| at 0.95 central CI")
print(f"    Head     h=1      h=3      h=6      h=12")
for head in ["point", "gauss", "gmm"]:
    cells = [f"{summary['calibration'][f'h{h}'][head]['mean_abs_err_95']*100:.2f}%" for h in HORIZONS]
    print(f"    {head:6s}  " + "  ".join(f"{c:>8s}" for c in cells))

print("\n" + "=" * 78)
print("Headline (cross-horizon mean):")
print(f"  Pinball@q=0.01:  point→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.01_point_to_gmm']*100:+.2f}%")
print(f"  Pinball@q=0.05:  point→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.05_point_to_gmm']*100:+.2f}%")
print(f"  Pinball@q=0.01:  gauss→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.01_gauss_to_gmm']*100:+.2f}%")
print(f"  Pinball@q=0.05:  gauss→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.05_gauss_to_gmm']*100:+.2f}%")
print("=" * 78)

# Save text summary
with open(RES / "38_tail_risk_summary.txt", "w") as f:
    f.write("Phase 3.1: Tail Pinball Loss & Calibration Error (12 variants, monthly S&P 500)\n")
    f.write("=" * 78 + "\n\n")
    f.write("Headline (cross-horizon mean over 4 backbones):\n")
    f.write(f"  Pinball@q=0.01:  point→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.01_point_to_gmm']*100:+.2f}%\n")
    f.write(f"  Pinball@q=0.05:  point→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.05_point_to_gmm']*100:+.2f}%\n")
    f.write(f"  Pinball@q=0.01:  gauss→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.01_gauss_to_gmm']*100:+.2f}%\n")
    f.write(f"  Pinball@q=0.05:  gauss→gmm improvement = {summary['headline']['mean_pinball_improvement_q0.05_gauss_to_gmm']*100:+.2f}%\n\n")
    f.write("Per-horizon mean pinball improvement (q=0.05, point→gmm):\n")
    for h, r in summary["headline"]["per_h_pinball_improvement_q0.05"].items():
        f.write(f"  {h}: {r*100:+.2f}%\n")
    f.write("\n\n")
    f.write("Per-backbone tail pinball improvement (q=0.05, h=1, point→gmm):\n")
    for bb in BACKBONES:
        r = pinball_improvement("h1", bb, 0.05)
        if r:
            f.write(f"  {bb:12s}: point={r['point']:.5f}  gmm={r['gmm']:.5f}  improvement={r['ratio_point_to_gmm']*100:+.2f}%\n")
    f.write("\n\n")
    f.write("Per-head calibration error at 95% central CI (|empirical - 0.95|):\n")
    f.write(f"  Head     h=1      h=3      h=6      h=12\n")
    for head in ["point", "gauss", "gmm"]:
        cells = [f"{summary['calibration'][f'h{h}'][head]['mean_abs_err_95']*100:.2f}%" for h in HORIZONS]
        f.write(f"  {head:6s}  " + "  ".join(f"{c:>8s}" for c in cells) + "\n")

print(f"\nSaved {RES/'38_tail_risk_summary.json'}")
print(f"Saved {RES/'38_tail_risk_summary.txt'}")
