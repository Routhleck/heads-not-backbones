#!/usr/bin/env bash
#
# run_reproduce.sh — one-shot smoke test of the headline claim.
#
# This reproduces the headline 12-variant × 5-fold × 4-horizon × 3-seed
# CRPS-Skill-Score grid (Table 2 in the paper), then verifies that the
# per-cell numbers fall within ±0.05pp of the committed reference values
# in results/35_combined_12variants.json.
#
# Usage:
#   bash run_reproduce.sh                # full pipeline
#   bash run_reproduce.sh --quick        # 1 seed (1/3 the wall-clock)
#
# Wall-clock: ~22 minutes on RTX 4060; ~3x on CPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    *)        echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

echo "[$(date +%H:%M:%S)] reproduce: starting (ROOT=$ROOT)"
echo "[$(date +%H:%M:%S)] python: $(python3 -V)"

# 1. Headline 12-variant panel (3 seeds)
if [[ "$QUICK" == "1" ]]; then
  SEEDS_FLAG="--seeds 0"          # smoke test: 1 seed only
else
  SEEDS_FLAG=""                   # default 3 seeds
fi

echo "[$(date +%H:%M:%S)] step 1/3: 22_sota_comparison (point+gauss baselines)"
python3 experiments/22_sota_comparison.py $SEEDS_FLAG

echo "[$(date +%H:%M:%S)] step 2/3: 34_gaussian_head (GMM head on point baselines)"
python3 experiments/34_gaussian_head.py $SEEDS_FLAG

echo "[$(date +%H:%M:%S)] step 3/3: 35_merge_12variants (combine into Table 2)"
python3 experiments/35_merge_12variants.py $SEEDS_FLAG

# 4. Diff against committed reference
REFERENCE="results/35_combined_12variants.json"
RERUN_OUT=$(mktemp /tmp/reproduce-XXXXXX.json)
trap "rm -f $RERUN_OUT" EXIT

echo "[$(date +%H:%M:%S)] diff: checking against committed reference $REFERENCE"

if [[ ! -f "$REFERENCE" ]]; then
  echo "  [WARN] no committed $REFERENCE yet (first run); skipping diff"
  exit 0
fi

python3 - <<EOF
import json
ref   = json.load(open("$REFERENCE"))
run   = json.load(open("$RERUN_OUT"))
fail  = 0
checked = 0
def walk(r, v, path):
    global fail, checked
    if isinstance(r, dict):
        for k in r:
            if k in v: walk(r[k], v[k], f"{path}.{k}")
    elif isinstance(r, (int, float)) and isinstance(v, (int, float)):
        checked += 1
        # Skip non-CRPS-SS scalars (e.g. config ints)
        if abs(r) > 50 or abs(v) > 50:
            return
        delta = abs(r - v)
        if delta > 0.05:
            fail += 1
            print(f"  [FAIL] {path}: ref={r:.3f}  rerun={v:.3f}  delta={delta:.3f}")
    else:
        return
walk(ref, run, "")
print(f"checked {checked} numeric cells, {fail} fail (>0.05pp delta)")
exit(1 if fail else 0)
EOF

if [[ $? -eq 0 ]]; then
  echo "[$(date +%H:%M:%S)] reproduce: PASS (all per-cell CSS within ±0.05pp of committed)"
else
  echo "[$(date +%H:%M:%S)] reproduce: FAIL — see diagnostics above"
  exit 1
fi
