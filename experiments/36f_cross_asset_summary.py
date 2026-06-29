"""Update Phase 1.4 cross-asset summary to include VIX.

Pulls the head gradient from each panel's daily/monthly result file
and produces a single consolidated table.

Backbones: TimesNet, DLinear, N-BEATS, iTransformer. iTransformer
(NeurIPS 2023, ICLR 2024 spotlight) replaces PatchTST (which was
dropped due to a patching-sigma incompatibility on fat-tailed
return series; see DECISION_PATCHTST_DROPPED.md).
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
RES = ROOT / "results"

# Daily SP500
with open(RES / "36e_daily_SP500_Return.json") as f:
    sp500_d = json.load(f)
# Daily VIX
with open(RES / "36e_daily_VIX.json") as f:
    vix_d = json.load(f)
# Daily DGS10 (rates)
with open(RES / "36e_daily_DGS10.json") as f:
    dgs10_d = json.load(f)
# Daily EURUSD (FX)
with open(RES / "36e_daily_EURUSD_Return.json") as f:
    eurusd_d = json.load(f)
# Monthly SP500 (existing)
with open(RES / "22_sota_comparison.json") as f:
    sp500_m = json.load(f)
with open(RES / "34_gaussian_head.json") as f:
    sp500_m_g = json.load(f)

# 4 backbones (PatchTST replaced by iTransformer — see module docstring).
BACKBONES = ["TimesNet", "DLinear", "NBEATS", "iTransformer"]


def head_gradient(monthly_dict, gauss_dict, panel_name, horizons):
    """Compute mean CRPS-SS over backbones for each head × horizon."""
    out = {}
    for h in horizons:
        hkey = f"h{h}"
        if hkey not in monthly_dict["metrics"]:
            continue
        # Use TimesNet_point as reference baseline
        baseline = monthly_dict["metrics"][hkey]["TimesNet_point"]["crps"]
        per_head = {}
        for head in ["point", "gauss", "gmm"]:
            cells = []
            for bb in BACKBONES:
                if head == "gauss":
                    crps = gauss_dict["metrics"][hkey][bb]["crps"]
                else:
                    var = f"{bb}_{head}"
                    if var in monthly_dict["metrics"][hkey]:
                        crps = monthly_dict["metrics"][hkey][var]["crps"]
                    else:
                        crps = None
                if crps is not None:
                    cells.append((baseline - crps) / baseline * 100)
            if cells:
                per_head[head] = float(np.mean(cells))
        out[hkey] = per_head
    return out


def head_gradient_daily(panel_dict, panel_name, horizons):
    """Same but for daily results where SS isn't pre-computed."""
    out = {}
    for h in horizons:
        hkey = f"h{h}"
        if hkey not in panel_dict["metrics"]:
            continue
        baseline = panel_dict["metrics"][hkey]["TimesNet_point"]["crps"]
        per_head = {}
        for head in ["point", "gauss", "gmm"]:
            cells = []
            for bb in BACKBONES:
                var = f"{bb}_{head}"
                if var in panel_dict["metrics"][hkey]:
                    crps = panel_dict["metrics"][hkey][var]["crps"]
                    cells.append((baseline - crps) / baseline * 100)
            if cells:
                per_head[head] = float(np.mean(cells))
        out[hkey] = per_head
    return out


monthly_grad = head_gradient(sp500_m, sp500_m_g, "monthly_sp500", [1, 3, 6, 12])
sp500_d_grad = head_gradient_daily(sp500_d, "daily_sp500", [1, 5, 10, 20])
vix_d_grad = head_gradient_daily(vix_d, "daily_vix", [1, 5, 10, 20])
dgs10_d_grad = head_gradient_daily(dgs10_d, "daily_dgs10", [1, 5, 10, 20])
dgs10_d_grad_full = dgs10_d_grad  # backward-compat alias; same as dgs10_d_grad now
eurusd_d_grad = head_gradient_daily(eurusd_d, "daily_eurusd", [1, 5, 10, 20])

# Print
print("=" * 78)
print("Phase 1.4 (updated): Cross-asset head gradient")
print("=" * 78)
print()
print("Backbones: TimesNet, DLinear, N-BEATS, iTransformer (NeurIPS 2023).")
print("PatchTST replaced by iTransformer (patching-sigma incompatibility on")
print("fat-tailed return series; see DECISION_PATCHTST_DROPPED.md).")
print()

# Monthly SP500
print("Monthly S&P 500 (Shiller, 1871-2023, n=1832)")
print(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}")
for h in [1, 3, 6, 12]:
    hkey = f"h{h}"
    g = monthly_grad.get(hkey, {})
    print(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%")
print()

# Daily SP500
print("Daily S&P 500 returns (FRED, 2014-2024, n=2142)")
print(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}")
for h in [1, 5, 10, 20]:
    hkey = f"h{h}"
    g = sp500_d_grad.get(hkey, {})
    print(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%")
print()

# Daily VIX
print("Daily VIX level (FRED, 1990-2024, n=8834)")
print(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}")
for h in [1, 5, 10, 20]:
    hkey = f"h{h}"
    g = vix_d_grad.get(hkey, {})
    print(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%")
print()

# Daily DGS10
print("Daily 10Y UST yield (FRED, 1962-2024, n=15735)")
print(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}")
for h in [1, 5, 10, 20]:
    hkey = f"h{h}"
    g = dgs10_d_grad.get(hkey, {})
    print(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%")
print()

# Daily EURUSD
print("Daily EUR/USD return (FRED, 1999-2024, n=6519)")
print(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}")
for h in [1, 5, 10, 20]:
    hkey = f"h{h}"
    g = eurusd_d_grad.get(hkey, {})
    print(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%")
print()

# Cross-panel mean
print("=" * 78)
print("Cross-panel mean (over horizons)")
print("=" * 78)
all_panels = [
    ("monthly_sp500", monthly_grad),
    ("daily_sp500", sp500_d_grad),
    ("daily_vix", vix_d_grad),
    ("daily_dgs10", dgs10_d_grad),
    ("daily_eurusd", eurusd_d_grad),
]
for panel_name, grad in all_panels:
    point_mean = np.mean([grad[k].get('point', 0) for k in grad if 'point' in grad[k]])
    gauss_mean = np.mean([grad[k].get('gauss', 0) for k in grad if 'gauss' in grad[k]])
    gmm_mean = np.mean([grad[k].get('gmm', 0) for k in grad if 'gmm' in grad[k]])
    print(f"  {panel_name:18s} point={point_mean:+6.2f}% gauss={gauss_mean:+6.2f}% gmm={gmm_mean:+6.2f}%")

# Save
out = {
    "config": {
        "backbones": BACKBONES,
        "panels": ["monthly_SP500", "daily_SP500", "daily_VIX", "daily_DGS10", "daily_EURUSD"],
        "exclusions": {
            "patchtst": (
                "PatchTST replaced by iTransformer (NeurIPS 2023, ICLR 2024 "
                "spotlight) due to a patching-sigma incompatibility on "
                "fat-tailed return series (MAE explodes 0.025/0.063 on daily "
                "SP500 h=1/h=5; CRPS-SS -1.95% at monthly h=12). See "
                "DECISION_PATCHTST_DROPPED.md for the full diagnostic."
            ),
        },
    },
    "monthly_sp500": monthly_grad,
    "daily_sp500": sp500_d_grad,
    "daily_vix": vix_d_grad,
    "daily_dgs10": dgs10_d_grad,
    "daily_eurusd": eurusd_d_grad,
    "headline": {
        "monthly_sp500_gmm_mean": float(np.mean([monthly_grad[k].get('gmm', 0) for k in monthly_grad])),
        "daily_sp500_gmm_mean": float(np.mean([sp500_d_grad[k].get('gmm', 0) for k in sp500_d_grad])),
        "daily_vix_gmm_mean": float(np.mean([vix_d_grad[k].get('gmm', 0) for k in vix_d_grad])),
        "daily_dgs10_gmm_mean": float(np.mean([dgs10_d_grad[k].get('gmm', 0) for k in dgs10_d_grad])),
        "daily_eurusd_gmm_mean": float(np.mean([eurusd_d_grad[k].get('gmm', 0) for k in eurusd_d_grad])),
    },
}

with open(RES / "36f_cross_asset_summary.json", "w") as f:
    json.dump(out, f, indent=2)

# Save text
with open(RES / "36f_cross_asset_summary.txt", "w") as f:
    f.write("Phase 1.4 (updated): Cross-asset head gradient\n")
    f.write("=" * 78 + "\n\n")
    f.write("Monthly S&P 500 (Shiller, 1871-2023, n=1832)\n")
    f.write(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}\n")
    for h in [1, 3, 6, 12]:
        hkey = f"h{h}"
        g = monthly_grad.get(hkey, {})
        f.write(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%\n")
    f.write("\n")
    f.write("Daily S&P 500 returns (FRED, 2014-2024, n=2142)\n")
    f.write(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}\n")
    for h in [1, 5, 10, 20]:
        hkey = f"h{h}"
        g = sp500_d_grad.get(hkey, {})
        f.write(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%\n")
    f.write("\n")
    f.write("Daily VIX level (FRED, 1990-2024, n=8834)\n")
    f.write(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}\n")
    for h in [1, 5, 10, 20]:
        hkey = f"h{h}"
        g = vix_d_grad.get(hkey, {})
        f.write(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%\n")
    f.write("\n")
    f.write("Daily 10Y UST yield (FRED, 1962-2024, n=15735)\n")
    f.write(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}\n")
    for h in [1, 5, 10, 20]:
        hkey = f"h{h}"
        g = dgs10_d_grad.get(hkey, {})
        f.write(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%\n")
    f.write("\n")
    f.write("Daily EUR/USD return (FRED, 1999-2024, n=6519)\n")
    f.write(f"  {'Horizon':10s} {'Point':>10s} {'Gauss':>10s} {'GMM':>10s}\n")
    for h in [1, 5, 10, 20]:
        hkey = f"h{h}"
        g = eurusd_d_grad.get(hkey, {})
        f.write(f"  h={h:<8d} {g.get('point', 0):+9.2f}% {g.get('gauss', 0):+9.2f}% {g.get('gmm', 0):+9.2f}%\n")
    f.write("\n")
    f.write("=" * 78 + "\n")
    f.write("Cross-panel mean (over horizons)\n")
    f.write("=" * 78 + "\n")
    for panel_name, grad in all_panels:
        point_mean = np.mean([grad[k].get('point', 0) for k in grad if 'point' in grad[k]])
        gauss_mean = np.mean([grad[k].get('gauss', 0) for k in grad if 'gauss' in grad[k]])
        gmm_mean = np.mean([grad[k].get('gmm', 0) for k in grad if 'gmm' in grad[k]])
        f.write(f"  {panel_name:18s} point={point_mean:+6.2f}% gauss={gauss_mean:+6.2f}% gmm={gmm_mean:+6.2f}%\n")

print(f"\nSaved {RES/'36f_cross_asset_summary.json'}")
print(f"Saved {RES/'36f_cross_asset_summary.txt'}")
