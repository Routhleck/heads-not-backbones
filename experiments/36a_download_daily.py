"""
Download daily panel from FRED (no rate limits, free, official).

Series:
  SP500   — S&P 500 Index (level, daily)
  VIXCLS  — CBOE Volatility Index (level, daily)
  DGS10   — 10-Year Treasury Constant Maturity Rate (%, daily)
  GOLDAMGBD228NLBM — Gold Fixing Price (USD/oz, daily, business days)
  DEXUSEU — U.S. / Euro Foreign Exchange Rate (level, daily)

We compute log-returns for prices (SP500, Gold, EURUSD) and keep
VIX and DGS10 as levels (they ARE the variable of interest for risk
management).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_datareader.data as web

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "raw"
DATA.mkdir(parents=True, exist_ok=True)

START = "1950-01-01"
END = "2024-12-31"

FRED_SERIES = {
    "SP500": "SP500_Return",   # log return
    "VIXCLS": "VIX",           # level
    "DGS10": "DGS10",          # level (10Y yield, %)
    "GOLDAMGBD228NLBM": "GOLD_Return",  # log return
    "DEXUSEU": "EURUSD_Return",         # log return
}


def main():
    series = {}
    for fred_id, name in FRED_SERIES.items():
        print(f"  [{name}] downloading {fred_id} from FRED...")
        try:
            s = web.DataReader(fred_id, "fred", START, END)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            continue
        s = s.iloc[:, 0]  # one column
        s.name = name
        s = s.dropna()
        # Compute log return if requested
        if name.endswith("_Return"):
            s = np.log(s.astype(float) / s.astype(float).shift(1))
            s = s.dropna()
        series[name] = s
        print(f"  [{name}] {len(s)} non-NaN obs, range: {s.index[0].date()} to {s.index[-1].date()}, "
              f"mean={s.mean():.5f}, std={s.std():.5f}")

    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    print(f"\n[panel] {len(panel)} rows, {panel.shape[1]} columns")
    print(f"[panel] non-NaN counts:\n{panel.notna().sum()}")

    out_path = DATA / "panel_daily.csv"
    panel.to_csv(out_path, float_format="%.6f")
    print(f"\nSaved {out_path}")
    return panel


if __name__ == "__main__":
    main()
