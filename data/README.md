# Data sources

Two committed CSV panels (small, in `data/raw/`):

| File                        | Rows   | Source                                             |
|-----------------------------|--------|----------------------------------------------------|
| `panel_monthly.csv`         | 1,832  | Shiller (Yale) — S&P 500 monthly log-returns 1871-2023 |
| `panel_daily.csv`           | ~37k   | Yahoo Finance + FRED — daily panel 1950-2024       |

Both CSVs are committed so reviewers don't need network access to
reproduce the headline numbers. To fetch fresh daily data over network:

```bash
python3 experiments/36a_download_daily.py     # pulls ^GSPC, ^VIX, DGS10, EURUSD
```

Series in `panel_daily.csv`:

| Column          | Source                | Notes                          |
|-----------------|------------------------|--------------------------------|
| SP500_Return    | Yahoo ^GSPC            | log-return                    |
| VIX             | Yahoo ^VIX             | level (in §5.3)              |
| DGS10           | FRED                   | 10Y Treasury yield (in §5.3) |
| EURUSD_Return   | FRED DEXUSEU           | log-return                    |

If the daily panel needs re-pulling, run the script above; it writes to
`data/raw/panel_daily.csv` after joining Yahoo Finance and FRED.
