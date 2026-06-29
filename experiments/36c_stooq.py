"""
Download S&P 500 daily from Stooq (free, no rate limit, 1950+).
https://stooq.com/q/d/?s=^spx&i=d
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import io

ROOT = Path("/Users/sichaohe/Documents/GitHub/finance-x-timesnet")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "raw"


def stooq_csv(ticker_with_caret, name):
    """Download from stooq.com — returns DataFrame with date-indexed Adj Close."""
    url = f"https://stooq.com/q/d/l/?s={ticker_with_caret}&i=d"
    print(f"  [{name}] downloading {url}...")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        print(f"  [{name}] HTTP {r.status_code}")
        return None
    if "No data" in r.text or len(r.text) < 100:
        print(f"  [{name}] empty response")
        return None
    df = pd.read_csv(io.StringIO(r.text))
    if df is None or len(df) == 0:
        print(f"  [{name}] empty dataframe")
        return None
    # stooq format: Date, Open, High, Low, Close, Volume
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        print(f"  [{name}] columns: {df.columns.tolist()}")
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    print(f"  [{name}] {len(df)} obs, range {df.index[0].date()} to {df.index[-1].date()}")
    return df


def main():
    panel_existing = pd.read_csv(DATA / "panel_daily.csv", index_col=0,
                                    parse_dates=True)
    print(f"[loaded existing] {panel_existing.shape}")

    # 1. S&P 500 from Stooq (^SPX)
    sp = stooq_csv("^spx", "SP500_Return")
    if sp is not None and len(sp) > 1000:
        ret = np.log(sp["Close"] / sp["Close"].shift(1))
        ret.name = "SP500_Return"
        ret = ret.dropna()
        panel_existing["SP500_Return"] = ret
        print(f"  [SP500_Return] {len(ret)} obs, mean={ret.mean():.5f}, std={ret.std():.5f}")

    # 2. Gold from Stooq (^GC.F or xauusd)
    for tk in ["^gc.f", "xauusd"]:
        gold = stooq_csv(tk, "GOLD_Return")
        if gold is not None and len(gold) > 1000:
            ret = np.log(gold["Close"] / gold["Close"].shift(1))
            ret.name = "GOLD_Return"
            ret = ret.dropna()
            panel_existing["GOLD_Return"] = ret
            print(f"  [GOLD_Return] {len(ret)} obs, mean={ret.mean():.5f}, std={ret.std():.5f}")
            break

    # Save
    panel_existing = panel_existing.sort_index()
    print(f"\n[final panel] {panel_existing.shape}")
    print(f"[non-NaN counts]\n{panel_existing.notna().sum()}")
    panel_existing.to_csv(DATA / "panel_daily.csv", float_format="%.6f")
    print(f"\nSaved {DATA / 'panel_daily.csv'}")


if __name__ == "__main__":
    main()
