"""
Merge exp 22 (point + GMM) and exp 34 (Gaussian) into a single 12-variant result file.

Output: results/35_combined_12variants.json

Schema (per horizon h):
  metrics[variant] = {
    "mae", "mase", "crps", "pinball", "winkler", "coverage",
    "crps_skill_score_vs_TimesNet_point",
  }

Variants (12 total, ordered by head then backbone for the heatmap):
  Point head:   TimesNet_point, DLinear_point, NBEATS_point, iTransformer_point
  Gauss head:   TimesNet_gauss, DLinear_gauss, NBEATS_gauss, iTransformer_gauss
  GMM head:     TimesNet_gmm,   DLinear_gmm,   NBEATS_gmm,   iTransformer_gmm
"""
import json
import sys
from pathlib import Path

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))

with open(ROOT / "results/22_sota_comparison.json") as f:
    exp22 = json.load(f)
with open(ROOT / "results/34_gaussian_head.json") as f:
    exp34 = json.load(f)

HORIZONS = [1, 3, 6, 12]
BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]
HEADS = ["point", "gauss", "gmm"]
VARIANTS = [f"{bb}_{h}" for h in HEADS for bb in BACKBONES]

# Baseline CRPS (TimesNet_point) per horizon
baseline = {}
for h in HORIZONS:
    baseline[f"h{h}"] = exp22["metrics"][f"h{h}"]["TimesNet_point"]["crps"]

# Build per-horizon merged metrics
merged = {f"h{h}": {} for h in HORIZONS}
merged_calibration = {f"h{h}": {} for h in HORIZONS}
merged_dm = {f"h{h}": {} for h in HORIZONS}

for h in HORIZONS:
    for variant in VARIANTS:
        bb, head = variant.rsplit("_", 1)
        if head == "gauss":
            src = exp34["metrics"][f"h{h}"][bb]
        else:
            src = exp22["metrics"][f"h{h}"][variant]
        # Copy fields
        merged[f"h{h}"][variant] = {
            "mae": src.get("mae"),
            "mase": src.get("mase"),
            "crps": src.get("crps"),
            "pinball": src.get("pinball", {}),
            "winkler": src.get("winkler", {}),
            "coverage": src.get("coverage", {}),
            "n_folds": src.get("n_folds", 5),
            "backbone": bb,
            "head": head,
        }
        # CRPS-SS vs TimesNet_point (consistent baseline)
        if src.get("crps") is not None and baseline[f"h{h}"] is not None:
            merged[f"h{h}"][variant]["crps_skill_score_vs_TimesNet_point"] = (
                1.0 - src["crps"] / baseline[f"h{h}"]
            )

    # Calibration data
    for variant in VARIANTS:
        bb, head = variant.rsplit("_", 1)
        if head == "gauss":
            if bb in exp34["calibration"][f"h{h}"]:
                merged_calibration[f"h{h}"][variant] = exp34["calibration"][f"h{h}"][bb]
        else:
            if variant in exp22["calibration"][f"h{h}"]:
                merged_calibration[f"h{h}"][variant] = exp22["calibration"][f"h{h}"][variant]

# DM test results from exp 22 (only GMM variants, not Gauss)
# We can compute the Gauss DM test later (exp 35b)
for h in HORIZONS:
    if f"h{h}" in exp22.get("dm_test", {}):
        merged_dm[f"h{h}"] = exp22["dm_test"][f"h{h}"]

out = {
    "config": {
        "horizons": HORIZONS,
        "backbones": BACKBONES,
        "heads": HEADS,
        "variants": VARIANTS,
        "n_total_variants": len(VARIANTS),
        "baseline": "TimesNet_point",
        "baseline_crps_per_h": baseline,
        "sources": ["experiments/22_sota_comparison.py", "experiments/34_gaussian_head.py"],
    },
    "metrics": merged,
    "calibration": merged_calibration,
    "dm_test": merged_dm,  # only GMM has DM results for now
}

out_path = ROOT / "results/35_combined_12variants.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"Saved {out_path}")
print(f"\n12 variants × 4 horizons = {len(VARIANTS) * len(HORIZONS)} cells")
print(f"Baseline (TimesNet_point CRPS): {baseline}")
print("\nQuick check — h=1 CRPS-SS vs TimesNet_point:")
for variant in VARIANTS:
    v = merged["h1"].get(variant, {})
    ss = v.get("crps_skill_score_vs_TimesNet_point", 0.0) * 100
    crps = v.get("crps", float("nan"))
    cov90 = v.get("coverage", {}).get("coverage_0.90", float("nan"))
    print(f"  {variant:<20} CRPS={crps:.5f}  CRPS-SS={ss:+6.2f}%  Cov@90={cov90:.4f}")
