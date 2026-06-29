"""
Experiment 41: Directional Accuracy and Out-of-Sample R^2 for the 12-cell story.

Purpose
-------
The main 12-variant experiments (22, 27, 34, 36e) report CRPS, MAE, MASE,
calibration, and significance tests (DM, MCS). They do NOT report two
"absolute" metrics that the finance literature treats as the primary
sanity check:

  - Directional Accuracy (DA): P(sign(forecast) == sign(realized))
    Naive random walk benchmark = 50% (assuming zero mean returns)
  - Out-of-sample R^2 (R^2_OOS): 1 - MSE(model) / MSE(random walk)
    Positive = beats random walk; 0 = same as RW; <0 = worse

We re-train each (backbone, head) cell on each asset with 1 seed x 5 folds
and record per-fold (truth, preds) to compute DA and R^2_OOS.

Cost
----
1 seed x 5 folds x 12 cells x 4 horizons x 5 assets = 1200 runs.
On RTX 4060 ~ 35 min total (4 loky workers).

Why 1 seed only
---------------
DA and R^2_OOS are aggregate statistics; the per-cell standard error
across 5 walk-forward folds is much larger than across 3 seeds. We
trade seed variance for fold variance coverage.

Outputs
-------
results/41_da_r2_summary.json — full numbers
results/41_da_r2_table.tex — LaTeX-ready table
"""
import os
import sys
import json
import time
import importlib.util
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ============= Reuse training stack from 36e =============
# 36e has all inline backbones + heads + training loop. Same config
# (SEQ_LEN=60, EPOCHS=80, LR=1e-3, BS=128) is what 22_sota uses too.
spec = importlib.util.spec_from_file_location(
    "exp_36e", ROOT / "experiments" / "36e_daily_12variant.py"
)
exp_36e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_36e)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ============= Asset registry =============
# 5 assets x 4 horizons each
DAILY_HORIZONS = [1, 5, 10, 20]
MONTHLY_HORIZONS = [1, 3, 6, 12]

ASSETS = [
    {
        "name": "monthly_SP500_Return",
        "path": ROOT / "data" / "raw" / "panel_monthly.csv",
        "column": "SP500_Return",
        "horizons": MONTHLY_HORIZONS,
    },
    {
        "name": "daily_SP500_Return",
        "path": ROOT / "data" / "raw" / "panel_daily.csv",
        "column": "SP500_Return",
        "horizons": DAILY_HORIZONS,
    },
    {
        "name": "daily_VIX",
        "path": ROOT / "data" / "raw" / "panel_daily.csv",
        "column": "VIX",
        "horizons": DAILY_HORIZONS,
    },
    {
        "name": "daily_DGS10",
        "path": ROOT / "data" / "raw" / "panel_daily.csv",
        "column": "DGS10",
        "horizons": DAILY_HORIZONS,
    },
    {
        "name": "daily_EURUSD_Return",
        "path": ROOT / "data" / "raw" / "panel_daily.csv",
        "column": "EURUSD_Return",
        "horizons": DAILY_HORIZONS,
    },
]

VARIANTS = [
    "TimesNet_point", "TimesNet_gauss", "TimesNet_gmm",
    "DLinear_point",  "DLinear_gauss",  "DLinear_gmm",
    "NBEATS_point",   "NBEATS_gauss",   "NBEATS_gmm",
    "iTransformer_point", "iTransformer_gauss", "iTransformer_gmm",
]

# Single seed for DA / R^2 (5-fold walk-forward already gives rich
# fold-level variance)
SEED = 0
N_JOBS = 4

# ============= Per-task worker =============
def run_one(asset, variant, horizon, fold_idx, seed):
    """Train one (variant, horizon, fold, seed) and return DA + R2_OOS."""
    df = pd.read_csv(asset["path"], index_col=0, parse_dates=True)
    series = df[asset["column"]].dropna().values.astype(np.float32)
    n = len(series)

    # Build the same walk-forward fold as 36e/22.
    init_train = int(n * exp_36e.INITIAL_TRAIN_FRAC)
    test_window = int(n * exp_36e.TEST_FRAC)
    step = int(n * exp_36e.STEP_FRAC)

    seq_len = exp_36e.SEQ_LEN

    # Build supervised pairs.
    X_all = []
    Y_all = []
    for t in range(n - seq_len - horizon + 1):
        X_all.append(series[t:t + seq_len])
        Y_all.append(series[t + seq_len:t + seq_len + horizon])
    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32)

    # Walk forward.
    fold_data = []
    train_end_series = init_train
    fold_i = 0
    while train_end_series + test_window + horizon <= n:
        test_end_series = train_end_series + test_window
        train_end_pair = train_end_series - seq_len
        test_end_pair = test_end_series - seq_len
        if train_end_pair <= 0 or test_end_pair <= train_end_pair:
            train_end_series += step
            continue
        X_train = X_all[:train_end_pair]
        Y_train = Y_all[:train_end_pair]
        X_test = X_all[train_end_pair:test_end_pair]
        Y_test = Y_all[train_end_pair:test_end_pair]
        if len(X_test) == 0:
            break
        fold_data.append((fold_i, X_train, Y_train, X_test, Y_test))
        fold_i += 1
        train_end_series += step

    if fold_idx >= len(fold_data):
        return None

    f, X_train, Y_train, X_test, Y_test = fold_data[fold_idx]

    # Train and get predictions (reuse 36e worker).
    args = (variant, horizon, f, seed, X_train, Y_train, X_test, Y_test)
    out = exp_36e.train_and_save_predictions(args)
    preds = out["preds"]   # (n_test, H) — point prediction per step
    truth = out["truth"]   # (n_test, H)

    # Directional Accuracy: for h-step forecast, sign of total return over h.
    # This matches the "h-month-ahead direction" convention.
    pred_dir = np.sign(preds.sum(axis=1))   # (n_test,)
    true_dir = np.sign(truth.sum(axis=1))   # (n_test,)
    da = float((pred_dir == true_dir).mean())

    # Out-of-sample R^2 = 1 - MSE(model) / MSE(random walk)
    # Random walk benchmark depends on series type:
    #   - LEVEL series (VIX, DGS10): RW = y_t (random walk in prices, perfect for persistent)
    #   - RETURN series (SP500_Return, EURUSD_Return): RW = sample mean (random walk in prices
    #     implies zero log-return)
    # We compute and return BOTH so the paper can show which benchmark it's comparing to.
    # The "primary" r2_oos uses the appropriate benchmark per asset type.
    is_return = asset["column"].endswith("_Return")
    y_t = X_test[:, -1]
    if is_return:
        # For return series, RW in log-prices → predicted return = sample mean.
        train_mean = float(Y_train.mean())
        rw_pred_primary = np.tile(np.full((X_test.shape[0], 1), train_mean, dtype=np.float32),
                                  (1, horizon))
        # Also compute y_t-based RW for reference
        rw_pred_y = np.tile(y_t[:, None], (1, horizon))
    else:
        # For level series, RW in levels → predicted = last observed.
        rw_pred_primary = np.tile(y_t[:, None], (1, horizon))
        rw_pred_y = rw_pred_primary
    mse_model = float(np.mean((truth - preds) ** 2))
    mse_rw_primary = float(np.mean((truth - rw_pred_primary) ** 2))
    mse_rw_y = float(np.mean((truth - rw_pred_y) ** 2))
    r2_oos = 1.0 - mse_model / mse_rw_primary if mse_rw_primary > 1e-12 else float("nan")
    r2_oos_vs_y = 1.0 - mse_model / mse_rw_y if mse_rw_y > 1e-12 else float("nan")

    return {
        "asset": asset["name"],
        "variant": variant,
        "horizon": horizon,
        "fold": f,
        "seed": seed,
        "da": da,
        "r2_oos": r2_oos,                  # vs appropriate benchmark (mean for returns, y_t for levels)
        "r2_oos_vs_y": r2_oos_vs_y,        # vs y_t (for paper appendix comparison)
        "is_return": is_return,
        "n_test": int(truth.shape[0]),
    }


# ============= Driver =============
def main(dry_run=False):
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "41_da_r2_summary.json"
    out_tex = out_dir / "41_da_r2_table.tex"

    if dry_run:
        # Tiny smoke test: 1 asset x 1 variant x 1 horizon x 1 fold = 1 task.
        # Verifies: verbose log, raw.pkl save, JSON write, methodology fix.
        tasks = [(ASSETS[0], VARIANTS[0], ASSETS[0]["horizons"][0], 0, SEED)]
        out_json = out_dir / "41_da_r2_dryrun.json"
        raw_path = out_dir / "41_da_r2_dryrun.raw.pkl"
    else:
        tasks = []
        for asset in ASSETS:
            for variant in VARIANTS:
                for horizon in asset["horizons"]:
                    for fold_idx in range(5):
                        tasks.append((asset, variant, horizon, fold_idx, SEED))
        raw_path = out_json.with_suffix(".raw.pkl")

    print(f"[41] total tasks: {len(tasks)}" + (" [DRY RUN]" if dry_run else ""))
    print(f"[41] raw results will be saved to {raw_path}")
    t0 = time.time()
    # verbose=10 -> joblib prints "Done N / 1200 | elapsed: X.Xmin" lines so
    # the log shows real-time progress (no silent 30-min gaps).
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(run_one)(*t) for t in tasks
    )
    elapsed = time.time() - t0
    print(f"[41] all done in {elapsed/60:.1f} min")
    # Save raw results immediately so a post-aggregation hang doesn't lose
    # the trained predictions (memory v2: joblib silent-aggregation can hang).
    import pickle
    with open(raw_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[41] wrote raw results: {raw_path}")

    # Filter None (shouldn't happen unless config drift) and aggregate.
    rows = [r for r in results if r is not None]
    summary = {}
    for r in rows:
        key = (r["asset"], r["variant"], r["horizon"])
        if key not in summary:
            summary[key] = {
                "da_per_fold": [], "r2_oos_per_fold": [],
                "r2_oos_vs_y_per_fold": [], "n_test": []
            }
        summary[key]["da_per_fold"].append(r["da"])
        summary[key]["r2_oos_per_fold"].append(r["r2_oos"])
        summary[key]["r2_oos_vs_y_per_fold"].append(r["r2_oos_vs_y"])
        summary[key]["n_test"].append(r["n_test"])

    # Compute aggregate stats.
    final = {}
    for (asset_name, variant, horizon), v in summary.items():
        da_arr = np.array(v["da_per_fold"])
        r2_arr = np.array(v["r2_oos_per_fold"])
        r2_y_arr = np.array(v["r2_oos_vs_y_per_fold"])
        final.setdefault(asset_name, {}).setdefault(variant, {})[str(horizon)] = {
            "da_mean": float(da_arr.mean()),
            "da_std": float(da_arr.std(ddof=1)) if len(da_arr) > 1 else 0.0,
            "r2_oos_mean": float(r2_arr.mean()),
            "r2_oos_std": float(r2_arr.std(ddof=1)) if len(r2_arr) > 1 else 0.0,
            "r2_oos_vs_y_mean": float(r2_y_arr.mean()),
            "r2_oos_vs_y_std": float(r2_y_arr.std(ddof=1)) if len(r2_y_arr) > 1 else 0.0,
            "n_folds": len(da_arr),
            "n_test_total": int(sum(v["n_test"])),
        }

    # Save JSON.
    with open(out_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"[41] wrote {out_json}")

    # Print headline table: monthly SP500 only (the spine).
    if "monthly_SP500_Return" in final:
        print("\n[41] monthly S&P 500 summary:")
        print(f"{'variant':22s} {'h':>3s}  {'DA%':>7s}  {'R2_OOS%':>9s}  {'R2_OOS%_vs_y':>12s}")
        for variant, by_h in final["monthly_SP500_Return"].items():
            for h in sorted(by_h.keys(), key=lambda x: int(x)):
                v = by_h[h]
                print(f"{variant:22s} {h:>3s}  {v['da_mean']*100:6.2f}%  "
                      f"{v['r2_oos_mean']*100:+8.3f}%  "
                      f"{v['r2_oos_vs_y_mean']*100:+11.3f}%")
    else:
        print(f"\n[41] monthly_SP500_Return not in final; assets present: {list(final.keys())}")

    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 task only, output to 41_da_r2_dryrun.json")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
