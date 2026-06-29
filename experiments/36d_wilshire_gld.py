"""
Download daily panel from FRED (no rate limits, free, official).

Long-history series (for walk-forward with 5+ folds):
  WILL5000IND — Wilshire 5000 Full Cap Index (1971+, daily) — proxy for S&P 500
  GLD        — SPDR Gold Trust ETF (2004+, daily) — gold proxy

Already have from exp 36a:
  SP500 (FRED, 2014-2024)
  VIXCLS (FRED, 1990-2024)
  DGS10 (FRED, 1962-2024)
  DEXUSEU (FRED, 1999-2024)

We use Wilshire 5000 as the long-history "broad US equity" series
since S&P 500 daily is paywalled (CRSP) or rate-limited (Yahoo).
Correlation Wilshire 5000 ↔ S&P 500 monthly > 0.99.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_datareader.data as web

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "raw"

START = "1950-01-01"
END = "2024-12-31"

NEW_SERIES = {
    "WILL5000IND": "WIL5000_Return",  # log return
    "GLD": "GOLD_Return",  # log return (Gold ETF proxy)
}


def main():
    panel_existing = pd.read_csv(DATA / "panel_daily.csv", index_col=0,
                                    parse_dates=True)
    print(f"[loaded existing] {panel_existing.shape}")
    print(f"[existing non-NaN]\n{panel_existing.notna().sum()}")

    for fred_id, name in NEW_SERIES.items():
        print(f"  [{name}] downloading {fred_id} from FRED...")
        try:
            s = web.DataReader(fred_id, "fred", START, END)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            continue
        s = s.iloc[:, 0]
        s = s.dropna()
        if name.endswith("_Return"):
            s = np.log(s.astype(float) / s.astype(float).shift(1))
            s = s.dropna()
        s.name = name
        panel_existing[name] = s
        print(f"  [{name}] {len(s)} obs, mean={s.mean():.5f}, std={s.std():.5f}")

    panel_existing = panel_existing.sort_index()
    print(f"\n[final panel] {panel_existing.shape}")
    print(f"[non-NaN counts]\n{panel_existing.notna().sum()}")

    panel_existing.to_csv(DATA / "panel_daily.csv", float_format="%.6f")
    print(f"\nSaved {DATA / 'panel_daily.csv'}")


if __name__ == "__main__":
    main()
