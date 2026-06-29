# Raw data loader for multi-asset financial time series.
#
# Supports:
#   - S&P 500 (Shiller dataset) 1871-present, monthly
#   - USD index DXY / DTWEXBGS (FRED) 1973-present, monthly
#   - Gold spot (WGC) 1968-present, monthly
#   - Generic FRED series via FredApi-like direct CSV
#
# CLI usage:
#   python -m src.data.raw_loader --series sp500
#   python -m src.data.raw_loader --series dxy --start 1980
#   python -m src.data.raw_loader --all
#
# Shiller data note: chapt26.xlsx is ANNUAL. For MONTHLY data 1871-present,
# use ie_data.xls (the Irrational Exuberance update dataset).
import argparse
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


# ============================================================
# Shiller S&P 500 (1871-present)
# ============================================================

SHILLER_URLS = [
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "https://shillerdata.com/",  # fallback (will need different parsing)
]


def load_shiller_sp500() -> pd.DataFrame:
    """
    Download Shiller's Irrational Exuberance monthly dataset.

    URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    Includes monthly S&P price (real & nominal), dividends, earnings, CPI,
    10y bond yield, CAPE ratio. 1871-present.

    Returns:
        DataFrame with columns detected from raw Excel header.
    """
    print(f"[Shiller] Downloading from primary URL...")
    resp = None
    for url in SHILLER_URLS:
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                print(f"  Attempt {attempt+1}/3 failed for {url}: {e}")
                time.sleep(2)
        if resp is not None and resp.status_code == 200:
            break
    if resp is None or resp.status_code != 200:
        raise RuntimeError(f"Failed to download Shiller data from any mirror")
    print(f"[Shiller] Downloaded from {resp.url} ({len(resp.content)} bytes)")
    # Old .xls format; try header=None first to inspect
    df = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data", header=None)
    print(f"[Shiller] Raw shape: {df.shape}")
    print(f"[Shiller] First 10 rows preview:")
    for i in range(min(10, len(df))):
        print(f"  row {i}: {list(df.iloc[i, :10])}")
    return df


# ============================================================
# FRED series (DXY / DTWEXBGS / CPI / etc.)
# ============================================================

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def load_fred_series(series_id: str, start: str = "1900-01-01", freq: str = "ME") -> pd.DataFrame:
    """
    Download a FRED series directly via CSV download (no API key required).

    Args:
        series_id: FRED series ID, e.g. 'DTWEXBGS', 'CPIAUCSL', 'DGS10'
        start: YYYY-MM-DD
        freq: 'ME' (month end) or 'D' (daily). Resamples accordingly.

    Returns:
        DataFrame with Date index and one column = series_id
    """
    url = f"{FRED_BASE}?id={series_id}&cosd={start}&coec=9999-12-31"
    print(f"[FRED] Downloading {series_id} from {url[:80]}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    # FRED uses '.' for missing values
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    # Resample to desired frequency if needed
    if freq == "ME":
        # Some series are daily (DTWEXBGS, DFF), some monthly (CPIAUCSL).
        # Resample to month-end using last value in each month.
        df = df.resample("ME").last()
    elif freq == "D":
        pass  # already daily
    return df


# ============================================================
# Convenience loaders for known series
# ============================================================

def load_dxy(start: str = "1973-01-01") -> pd.DataFrame:
    """FRED DTWEXBGS — Trade Weighted USD Index: Broad Goods (1973-present)."""
    return load_fred_series("DTWEXBGS", start=start)


def load_cpi(start: str = "1913-01-01") -> pd.DataFrame:
    """FRED CPIAUCSL — Consumer Price Index All Urban (monthly, 1913-present)."""
    return load_fred_series("CPIAUCSL", start=start)


def load_fed_funds(start: str = "1954-01-01") -> pd.DataFrame:
    """FRED DFF — Federal Funds Effective Rate."""
    return load_fred_series("DFF", start=start)


def load_m2(start: str = "1959-01-01") -> pd.DataFrame:
    """FRED M2SL — M2 Money Stock."""
    return load_fred_series("M2SL", start=start)


# ============================================================
# FRED commodities (Global Price of Index, Not Seasonally Adjusted)
# ============================================================

def load_copper(start: str = "1960-01-01") -> pd.DataFrame:
    """FRED PCOPPUSDM — Global price of Copper (US$/metric ton), monthly."""
    return load_fred_series("PCOPPUSDM", start=start)


def load_oil_wti(start: str = "1986-01-01") -> pd.DataFrame:
    """FRED MCOILWTICO — WTI Crude Oil (US$/bbl), monthly."""
    return load_fred_series("MCOILWTICO", start=start)


def load_wheat(start: str = "1960-01-01") -> pd.DataFrame:
    """FRED PWHEAMTUSDM — Global price of Wheat (US$/metric ton), monthly."""
    return load_fred_series("PWHEAMTUSDM", start=start)


# NOTE: Gold spot price (LBMA PM) is not in FRED's free tier. Series ID
# 'PGOLDUSDM' was 404 at time of writing. To include gold in future work,
# pull from LBMA (https://www.lbma.org.uk/prices-and-data/precious-metal-prices)
# or Nasdaq Data Link (Quandl LPPM/PM). For this iteration we drop gold.

def build_unified_panel(save_csv: bool = True) -> pd.DataFrame:
    """
    Build a unified monthly panel of:
      - SP500 price + dividend yield (Shiller)
      - DTWEXBGS (FRED) — USD index
      - CPIAUCSL (FRED) — for real-term conversion
      - DFF (FRED) — Fed funds rate
      - M2SL (FRED) — money supply
    """
    print("\n=== Building unified monthly panel ===")
    # Shiller
    shiller = load_shiller_sp500()
    # FRED series
    cpi = load_cpi()
    dxy = load_dxy()
    fed = load_fed_funds()
    m2 = load_m2()

    # Process Shiller ie_data.xls — manually assign clean column names
    # Shiller's raw file has multi-row headers (rows 6+7) with duplicate names.
    # Standard Shiller column schema (from his published data dictionary):
    #   0: Date (YYYY.MM format, e.g., 1871.01 = Jan 1871)
    #   1: P — S&P Composite Price Index
    #   2: D — Dividend
    #   3: E — Earnings
    #   4: CPI — Consumer Price Index
    #   5: Date Fraction (decimal, e.g., 1871.042 = mid-Jan 1871)
    #   6: Rate GS10 — Long Interest Rate (10y bond yield)
    #   7-13: Real Price/Dividend/Earnings
    #   14-21: CAPE, Total Return CAPE, Returns
    SHILLER_COLS = [
        "Date", "P", "D", "E", "CPI", "DateFrac", "RateGS10",
        "RealPrice", "RealDividend", "RealEarnings",
        "CAPE", "TR_CAPE", "Yield", "Returns_Nominal",
        "Returns_Real", "Returns_Total",
    ]
    # Use header row 7 for the actual data start
    header_row = 7
    print(f"[Shiller] Using header row {header_row}")
    # Truncate to known columns (drop trailing extra columns)
    n_cols = min(len(SHILLER_COLS), shiller.shape[1])
    shiller = shiller.iloc[header_row + 1:, :n_cols].reset_index(drop=True)
    shiller.columns = SHILLER_COLS[:n_cols]

    # Parse Date column — Shiller uses YYYY.MM format (e.g., 2024.07)
    # Drop rows where Date is NaN first
    shiller = shiller.dropna(subset=["Date"]).copy()
    date_col = shiller["Date"].astype(float)
    # 1871.01 → Jan 1871; 1871.02 → Feb 1871; 1871.12 → Dec 1871
    # Decimal part is month-as-2-digit-integer, not month/12
    year = date_col.astype(int)
    month = ((date_col - year) * 100).round().astype(int).clip(1, 12)
    dates = pd.to_datetime(dict(year=year, month=month, day=1))
    shiller["_date"] = dates
    shiller = shiller.set_index("_date")
    # Drop duplicate dates (keep first)
    shiller = shiller[~shiller.index.duplicated(keep="first")]
    # Resample to month-end to align with FRED ME frequency
    shiller = shiller.resample("ME").last()

    sp_col = "P"
    cpi_col = "CPI"
    print(f"[Shiller] S&P price col = {sp_col}, CPI col = {cpi_col}")
    print(f"[Shiller] Shiller date range: {shiller.index.min()} → {shiller.index.max()}, n={len(shiller)}")

    # FRED commodities
    copper = load_copper()
    oil = load_oil_wti()
    wheat = load_wheat()

    # FRED series already resampled to month-end inside load_fred_series
    # Build unified monthly panel
    panel = pd.DataFrame(index=pd.date_range("1871-01-01", "2025-12-31", freq="ME"))
    if sp_col:
        panel["SP500_Price"] = shiller[sp_col].astype(float)
    if cpi_col:
        # Use Shiller's CPI first (longer history), fallback to FRED
        panel["CPI_Shiller"] = shiller[cpi_col].astype(float)
    panel["CPI"] = cpi["CPIAUCSL"].astype(float)
    panel["DXY"] = dxy["DTWEXBGS"].astype(float)
    panel["FedFunds"] = fed["DFF"].astype(float)
    panel["M2"] = m2["M2SL"].astype(float)
    panel["Copper"] = copper["PCOPPUSDM"].astype(float)
    panel["Oil_WTI"] = oil["MCOILWTICO"].astype(float)
    panel["Wheat"] = wheat["PWHEAMTUSDM"].astype(float)

    # Log-returns + real terms
    panel["SP500_Real"] = panel["SP500_Price"] / panel["CPI"] * 100
    panel["SP500_Return"] = np.log(panel["SP500_Price"] / panel["SP500_Price"].shift(1))
    panel["DXY_Return"] = np.log(panel["DXY"] / panel["DXY"].shift(1))
    panel["M2_Growth"] = panel["M2"] / panel["M2"].shift(12) - 1  # YoY
    # Commodity log-returns
    for col in ["Copper", "Oil_WTI", "Wheat"]:
        panel[f"{col}_Return"] = np.log(panel[col] / panel[col].shift(1))

    # Quality stats
    print(f"\n=== Panel summary ===")
    print(f"Shape: {panel.shape}")
    print(f"Date range: {panel.index.min()} → {panel.index.max()}")
    for col in panel.columns:
        n_valid = panel[col].notna().sum()
        first_valid = panel[col].first_valid_index()
        last_valid = panel[col].last_valid_index()
        print(f"  {col}: {n_valid} valid ({100*n_valid/len(panel):.1f}%), {first_valid} → {last_valid}")

    if save_csv:
        out_path = DATA_RAW / "panel_monthly.csv"
        panel.to_csv(out_path)
        print(f"\nSaved → {out_path}")

    return panel


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=["sp500", "dxy", "cpi", "fedfunds", "m2", "shiller"],
                        help="Which single series to download")
    parser.add_argument("--start", default="1900-01-01", help="Start date for FRED")
    parser.add_argument("--all", action="store_true", help="Build full unified panel")
    args = parser.parse_args()

    if args.all:
        build_unified_panel()
        return

    if args.series == "shiller" or args.series == "sp500":
        df = load_shiller_sp500()
        out = DATA_RAW / "shiller_sp500_raw.csv"
        df.to_csv(out, index=False)
        print(f"Saved → {out}")
        return

    loader = {
        "dxy": load_dxy,
        "cpi": load_cpi,
        "fedfunds": load_fed_funds,
        "m2": load_m2,
    }[args.series]
    df = loader(start=args.start)
    out = DATA_RAW / f"fred_{args.series}.csv"
    df.to_csv(out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()