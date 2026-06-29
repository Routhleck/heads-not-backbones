"""Phase 3.3: FRTB-style economic value quantification.

The Basel FRTB Internal Models Approach (IMA) charges capital based on
the bank's own VaR/ES estimates. If a better forecast model reduces
the mis-pricing of tail risk, the bank needs to hold less capital for
the same risk exposure.

We use the existing pinball losses at q=0.05 (5% VaR proper score)
and q=0.01 (1% VaR proper score) as the bank's "loss function" for
mis-estimating the 1-day-ahead VaR at the 95% / 99% confidence
levels. The improvement ratio (point->gmm, gauss->gmm) is interpreted
as a proportional reduction in the bank's forecasting cost.

Note: this is a *proxy* for capital savings — the true FRTB IMA
formula uses ES at 97.5% with a 12.5-month rolling window. Our
12-variant experiment is monthly S&P 500, so we report the proxy
honestly and note the mapping limitations.
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
with open(RES / "38_tail_risk_summary.json") as f:
    tail = json.load(f)
with open(RES / "37a_garch_monthly.json") as f:
    garch = json.load(f)
with open(RES / "37b_caviar_monthly.json") as f:
    caviar = json.load(f)

HORIZONS = [1, 3, 6, 12]

# Baseline: TimesNet_point h=1 CRPS = 0.01934
# Convert to basis points (1bp = 0.01% = 0.0001 fractional)
# Monthly return typical ~1% = 100bps
baseline_crps_bps = 0.01934 * 10000  # = 193.4 bps

# Pinball at q=0.05 is in fractional units
# Per fold mean of 0.0050 (h=1) = 50 bps monthly
# This is the "expected VaR forecasting cost per period"
# A 6.3% reduction in pinball = 6.3% reduction in this cost

# FRTB IMA capital formula (simplified):
#   K_99_VaR = 3 * VaR_99 (1-day)  [multiplier 3 in stress periods]
#   K_97.5_ES = 1.5 * ES_97.5 (1-day, 12-week average)
# Total K = max(K_99_VaR, K_97.5_ES) + SLA add-on
# In our 5% VaR (q=0.05) world, "VaR_95" is the analog.
# A 6.3% reduction in pinball_0.05 → 6.3% reduction in the bank's
# expected loss from VaR forecasting error → similar reduction in
# capital charge required (under a linearized FRTB approximation)

print("=" * 78)
print("Phase 3.3: FRTB-style Economic Value Quantification")
print("=" * 78)
print(f"\nAssumed baseline: TimesNet_point h=1 CRPS = {baseline_crps_bps:.1f} bps (monthly)")

# Pinball at q=0.05 in bps
print("\n[A] Pinball at q=0.05 (5% VaR proxy, monthly S&P 500, bps)")
print(f"    Head       h=1      h=3      h=6      h=12")
for head_name, source in [("point", sota), ("gauss", gauss), ("gmm", sota)]:
    cells = []
    for h in HORIZONS:
        if head_name == "point":
            v = source["metrics"][f"h{h}"][f"TimesNet_point"]["pinball"]["pinball_0.05"]
        elif head_name == "gauss":
            v = source["metrics"][f"h{h}"]["TimesNet"]["pinball"]["pinball_0.05"]
        else:  # gmm
            v = source["metrics"][f"h{h}"][f"TimesNet_gmm"]["pinball"]["pinball_0.05"]
        cells.append(f"{v*10000:.2f}bp")
    print(f"    {head_name:6s}  " + "  ".join(f"{c:>8s}" for c in cells))

# Pinball at q=0.01 in bps
print("\n[B] Pinball at q=0.01 (1% VaR proxy, monthly S&P 500, bps)")
print(f"    Head       h=1      h=3      h=6      h=12")
for head_name, source in [("point", sota), ("gauss", gauss), ("gmm", sota)]:
    cells = []
    for h in HORIZONS:
        if head_name == "point":
            v = source["metrics"][f"h{h}"][f"TimesNet_point"]["pinball"]["pinball_0.01"]
        elif head_name == "gauss":
            v = source["metrics"][f"h{h}"]["TimesNet"]["pinball"]["pinball_0.01"]
        else:  # gmm
            v = source["metrics"][f"h{h}"][f"TimesNet_gmm"]["pinball"]["pinball_0.01"]
        cells.append(f"{v*10000:.2f}bp")
    print(f"    {head_name:6s}  " + "  ".join(f"{c:>8s}" for c in cells))

# FRTB capital charge proxy
# For each head, h=1:
#   - 5% VaR proxy: average pinball_0.05 (bps)
#   - 1% VaR proxy: average pinball_0.01 (bps)
# Cost function = pinball_0.05 + 1.5 * pinball_0.01 (FRTB-style additive)
print("\n[C] FRTB-style cost function = pinball_0.05 + 1.5 * pinball_0.01 (bps, h=1)")
for head_name, source in [("point", sota), ("gauss", gauss), ("gmm", sota)]:
    if head_name == "point":
        p05 = source["metrics"]["h1"][f"TimesNet_point"]["pinball"]["pinball_0.05"]
        p01 = source["metrics"]["h1"][f"TimesNet_point"]["pinball"]["pinball_0.01"]
    elif head_name == "gauss":
        p05 = source["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.05"]
        p01 = source["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.01"]
    else:  # gmm
        p05 = source["metrics"]["h1"][f"TimesNet_gmm"]["pinball"]["pinball_0.05"]
        p01 = source["metrics"]["h1"][f"TimesNet_gmm"]["pinball"]["pinball_0.01"]
    cost = (p05 + 1.5 * p01) * 10000
    print(f"    {head_name:6s}  {p05*10000:.2f}bp + 1.5*{p01*10000:.2f}bp = {cost:.2f}bp")

# Translate to GMM savings
p05_gmm = sota["metrics"]["h1"]["TimesNet_gmm"]["pinball"]["pinball_0.05"]
p01_gmm = sota["metrics"]["h1"]["TimesNet_gmm"]["pinball"]["pinball_0.01"]
p05_pt = sota["metrics"]["h1"]["TimesNet_point"]["pinball"]["pinball_0.05"]
p01_pt = sota["metrics"]["h1"]["TimesNet_point"]["pinball"]["pinball_0.01"]
cost_gmm = (p05_gmm + 1.5 * p01_gmm) * 10000
cost_pt = (p05_pt + 1.5 * p01_pt) * 10000
saving = cost_pt - cost_gmm
saving_pct = (cost_pt - cost_gmm) / cost_pt * 100
print(f"\n    GMM cost vs point cost: {cost_gmm:.2f}bp vs {cost_pt:.2f}bp")
print(f"    FRTB capital proxy SAVINGS (point -> gmm):  {saving:.2f}bp  ({saving_pct:+.1f}%)")
print(f"    (Linearized FRTB approximation: cost reduction = capital reduction)")

# Now compare to classical baselines
print("\n[D] Cost comparison: Deep NBEATS_gmm vs classical (h=1)")
nb_gmm_cost = (sota["metrics"]["h1"]["NBEATS_gmm"]["pinball"]["pinball_0.05"] +
               1.5 * sota["metrics"]["h1"]["NBEATS_gmm"]["pinball"]["pinball_0.01"]) * 10000
garch_egarch = garch["methods"]["EGARCH_skewt"]["h1"]
garch_cost = garch_egarch.get("pinball_0.05_bps", None)
caviar_h1 = caviar["results"]["h1"]["pinball_mean"]
caviar_cost = (caviar_h1["0.05"] + 1.5 * caviar_h1["0.01"]) * 10000
print(f"    NBEATS_gmm (deep):    {nb_gmm_cost:.2f}bp")
print(f"    EGARCH_skewt:         {garch_cost if garch_cost else 'n/a (no pinball stored)'}bp")
print(f"    CAViaR:               {caviar_cost:.2f}bp")
print(f"    TimesNet_point:       {cost_pt:.2f}bp (baseline)")
print(f"    TimesNet_gmm:         {cost_gmm:.2f}bp")
print()

# Save
out = {
    "config": {
        "frtb_formula_proxy": "K_95 = pinball_0.05 + 1.5 * pinball_0.01 (linearized)",
        "horizons": HORIZONS,
        "data": "monthly S&P 500 (Shiller), n=1832, walk-forward 5 folds",
    },
    "pinball_0.05_bps_per_head_h1": {
        "point": float(sota["metrics"]["h1"]["TimesNet_point"]["pinball"]["pinball_0.05"] * 10000),
        "gauss": float(gauss["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.05"] * 10000),
        "gmm":   float(sota["metrics"]["h1"]["TimesNet_gmm"]["pinball"]["pinball_0.05"] * 10000),
    },
    "pinball_0.01_bps_per_head_h1": {
        "point": float(sota["metrics"]["h1"]["TimesNet_point"]["pinball"]["pinball_0.01"] * 10000),
        "gauss": float(gauss["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.01"] * 10000),
        "gmm":   float(sota["metrics"]["h1"]["TimesNet_gmm"]["pinball"]["pinball_0.01"] * 10000),
    },
    "frtb_cost_proxy_bps_h1": {
        "point": cost_pt,
        "gauss": (gauss["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.05"] +
                  1.5 * gauss["metrics"]["h1"]["TimesNet"]["pinball"]["pinball_0.01"]) * 10000,
        "gmm": cost_gmm,
    },
    "savings_bps_point_to_gmm_h1": saving,
    "savings_pct_point_to_gmm_h1": saving_pct,
    "classical_comparison_h1": {
        "NBEATS_gmm": nb_gmm_cost,
        "CAViaR": caviar_cost,
        "TimesNet_point": cost_pt,
    },
}
with open(RES / "38c_frtb_capital.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {RES / '38c_frtb_capital.json'}")
