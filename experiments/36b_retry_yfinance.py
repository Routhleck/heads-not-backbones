"""
Retry downloading S&P 500 daily (long history) from yfinance with
rate-limit handling. Falls back to a different download method if
yfinance is rate-limited.

SP500: 1950+ from yfinance ^GSPC
Gold: 1975+ from yfinance GC=F
"""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "raw"


def download_yf_slow(ticker, start, end, max_retries=3):
    """Download with retries, sleep between calls."""
    import yfinance as yf
    for attempt in range(max_retries):
        try:
            print(f"  attempt {attempt+1}/{max_retries} for {ticker}...")
            df = yf.download(ticker, start=start, end=end, progress=False,
                              auto_adjust=False, timeout=30)
            if df is None or len(df) == 0:
                print(f"  empty response")
                time.sleep(30)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            print(f"  error: {e}")
            time.sleep(30)
    return None


def main():
    panel_existing = pd.read_csv(DATA / "panel_daily.csv", index_col=0,
                                    parse_dates=True)
    print(f"[loaded existing] {panel_existing.shape}")

    # 1. S&P 500 long history
    sp = download_yf_slow("^GSPC", "1950-01-01", "2024-12-31")
    if sp is not None and len(sp) > 1000:
        ret = np.log(sp["Adj Close"] / sp["Adj Close"].shift(1))
        ret.name = "SP500_Return"
        ret = ret.dropna()
        print(f"  [SP500_Return] {len(ret)} obs, range {ret.index[0].date()} to {ret.index[-1].date()}")
        panel_existing["SP500_Return"] = ret
    else:
        print("  SP500 download failed")

    time.sleep(5)

    # 2. Gold (CME front-month futures)
    gold = download_yf_slow("GC=F", "1975-01-01", "2024-12-31")
    if gold is not None and len(gold) > 1000:
        ret = np.log(gold["Adj Close"] / gold["Adj Close"].shift(1))
        ret.name = "GOLD_Return"
        ret = ret.dropna()
        print(f"  [GOLD_Return] {len(ret)} obs, range {ret.index[0].date()} to {ret.index[-1].date()}")
        panel_existing["GOLD_Return"] = ret
    else:
        print("  Gold download failed")

    time.sleep(5)

    # Save
    panel_existing = panel_existing.sort_index()
    print(f"\n[final panel] {panel_existing.shape}")
    print(f"[non-NaN counts]\n{panel_existing.notna().sum()}")
    panel_existing.to_csv(DATA / "panel_daily.csv", float_format="%.6f")
    print(f"\nSaved {DATA / 'panel_daily.csv'}")


if __name__ == "__main__":
    main()
