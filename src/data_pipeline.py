"""
data_pipeline.py
----------------
Phase 2 of the multi-factor equity strategy research project.

Institutional-grade market data pipeline:
    1. Downloads adjusted close prices from Yahoo Finance via yfinance
    2. Validates data quality with configurable thresholds
    3. Handles missing data using forward-fill + back-fill convention
    4. Computes log returns and simple returns
    5. Persists all artefacts to /data with checksums for reproducibility

Why log returns?
    Log returns are time-additive (r_total = sum of daily r), approximately
    normally distributed, and numerically stable — all preferred properties
    for cross-sectional factor research.

Why adjusted close?
    Adjusted close accounts for dividends and stock splits, giving a
    total-return price series.  Using raw close would introduce artificial
    return spikes on ex-dividend and split dates.

Author  : Quant Research Team
Created : 2025-05
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from universe import get_universe

# ---------------------------------------------------------------------------
# Logging — institutional pipelines always log; never use bare print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all magic numbers in one place
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Lookback: 5 full calendar years of daily data
END_DATE: str = datetime.today().strftime("%Y-%m-%d")
START_DATE: str = (datetime.today() - timedelta(days=5 * 365 + 2)).strftime("%Y-%m-%d")

# Quality thresholds
MAX_MISSING_PCT: float = 0.02   # Drop ticker if > 2 % of observations are NaN
MIN_PRICE: float = 1.0          # Drop observations with price < $1 (likely bad data)
MIN_TRADING_DAYS: int = 20      # Minimum days to consider a ticker usable

# Download parameters
BATCH_SIZE: int = 20            # Tickers per yfinance batch call
REQUEST_TIMEOUT: int = 30       # Seconds before giving up


# ---------------------------------------------------------------------------
# Step 1 – Download raw prices
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers in batches.

    Batching avoids Yahoo Finance's undocumented rate limits.  A single
    yf.download() call with 100 tickers sometimes silently returns partial
    data; batching makes failures explicit and recoverable.

    Parameters
    ----------
    tickers   : list of ticker strings
    start     : ISO date string, inclusive
    end       : ISO date string, exclusive
    batch_size: number of tickers per API call

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index = trading dates (DatetimeIndex),
        columns = ticker symbols, values = adjusted close prices (USD).
    """
    log.info(f"Starting download: {len(tickers)} tickers | {start} → {end}")

    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        log.info(f"  Batch {batch_num}/{len(batches)}: {batch}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress yfinance deprecation noise
                raw = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    auto_adjust=True,   # gives us the adjusted prices directly
                    progress=False,
                    timeout=REQUEST_TIMEOUT,
                )

            if raw.empty:
                log.warning(f"  Batch {batch_num}: empty response — skipping")
                failed_tickers.extend(batch)
                continue

            # yfinance returns MultiIndex columns when len(batch) > 1
            # Extract just the "Close" level (= adjusted close when auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                # Single-ticker case: flat columns
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            frames.append(close)

        except Exception as exc:
            log.error(f"  Batch {batch_num} failed: {exc}")
            failed_tickers.extend(batch)

    if not frames:
        raise RuntimeError("All batches failed — check network / API access.")

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices.columns.name = "ticker"

    if failed_tickers:
        log.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

    log.info(
        f"Raw download complete: {prices.shape[0]} trading days × "
        f"{prices.shape[1]} tickers"
    )
    return prices


# ---------------------------------------------------------------------------
# Step 2 – Validate & clean prices
# ---------------------------------------------------------------------------

def validate_and_clean(
    prices: pd.DataFrame,
    max_missing_pct: float = MAX_MISSING_PCT,
    min_price: float = MIN_PRICE,
    min_days: int = MIN_TRADING_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """
    Institutional data cleaning in three ordered passes.

    Pass 1 — Remove clearly bad observations
        Prices below $1 are set to NaN.  This catches data-vendor errors
        (e.g., a split not reflected in the adjustment factor) that would
        create huge spurious return spikes.

    Pass 2 — Drop tickers with too much missing data
        If a ticker is missing > max_missing_pct of all trading days, it is
        removed from the universe.  Imputing large gaps introduces look-ahead
        bias and distorts factor exposures.

    Pass 3 — Forward-fill then back-fill residual NaNs
        Forward-fill: carry the last known price forward (e.g., a trading halt
        or a brief data gap).  This is the most conservative assumption —
        price did not change.
        Back-fill: fill any leading NaNs (ticker started trading after the
        sample start date) with the first available observation.  This avoids
        survivorship bias from simply dropping late-starting tickers.

    Returns
    -------
    pd.DataFrame
        Cleaned price matrix.
    dict
        Data-quality report for logging and downstream auditing.
    """
    log.info("Running data validation and cleaning …")

    report: dict = {
        "raw_shape": prices.shape,
        "tickers_dropped_low_price": [],
        "tickers_dropped_missing": [],
        "missing_pct_before": {},
        "missing_pct_after": {},
    }

    # --- Pass 1: Clip implausibly low prices ---
    suspect_mask = prices < min_price
    n_suspect = int(suspect_mask.sum().sum())
    if n_suspect:
        log.warning(f"  Zeroing {n_suspect} observations below ${min_price:.2f}")
        prices = prices.where(~suspect_mask, other=np.nan)

    # --- Pass 2: Drop tickers with excessive missing data ---
    missing_pct = prices.isna().mean()
    report["missing_pct_before"] = missing_pct.to_dict()

    bad_tickers = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if bad_tickers:
        log.warning(
            f"  Dropping {len(bad_tickers)} tickers with >{max_missing_pct:.0%} "
            f"missing data: {bad_tickers}"
        )
        report["tickers_dropped_missing"] = bad_tickers
        prices = prices.drop(columns=bad_tickers)

    # --- Pass 3: Forward-fill then back-fill ---
    prices = prices.ffill().bfill()

    # --- Final sanity checks ---
    prices = prices.dropna(axis=1, how="any")   # drop anything still NaN
    prices = prices.sort_index()                 # ensure chronological order

    # Drop tickers with too few trading days (e.g. very recent IPOs)
    short_tickers = prices.columns[prices.count() < min_days].tolist()
    if short_tickers:
        log.warning(f"  Dropping {len(short_tickers)} tickers with < {min_days} days: {short_tickers}")
        prices = prices.drop(columns=short_tickers)

    report["missing_pct_after"] = prices.isna().mean().to_dict()
    report["clean_shape"] = prices.shape

    log.info(
        f"Cleaning complete: {prices.shape[0]} days × {prices.shape[1]} tickers "
        f"(dropped {report['raw_shape'][1] - prices.shape[1]} tickers)"
    )
    return prices, report


# ---------------------------------------------------------------------------
# Step 3 – Compute return matrices
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute both simple and log return matrices.

    Simple returns  : r_t = (P_t / P_{t-1}) - 1
        Used for performance attribution, portfolio construction, and
        reporting (because they compound correctly across assets).

    Log returns     : r_t = ln(P_t / P_{t-1})
        Used for statistical analysis, factor model estimation, and
        risk modelling (additive across time, better distributional properties).

    Both matrices have the same shape: (T-1) × N, where T is the number
    of price observations and N is the number of tickers.

    Returns
    -------
    dict with keys 'simple' and 'log', each a pd.DataFrame.
    """
    log.info("Computing return matrices …")

    simple_returns = prices.pct_change(fill_method=None).iloc[1:]
    log_returns    = np.log(prices / prices.shift(1)).iloc[1:]

    # Trim extreme return outliers that survive price cleaning
    # Winsorise at 1st / 99th percentile — standard in institutional research
    # NOTE: we winsorise *after* computing returns, not prices, to avoid
    #       introducing artificial price levels.
    for col in simple_returns.columns:
        lo, hi = simple_returns[col].quantile([0.001, 0.999])
        simple_returns[col] = simple_returns[col].clip(lo, hi)
        lo_l, hi_l = log_returns[col].quantile([0.001, 0.999])
        log_returns[col] = log_returns[col].clip(lo_l, hi_l)

    log.info(
        f"Returns computed: {simple_returns.shape[0]} periods × "
        f"{simple_returns.shape[1]} tickers"
    )
    return {"simple": simple_returns, "log": log_returns}


# ---------------------------------------------------------------------------
# Step 4 – Persist artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    prices: pd.DataFrame,
    returns: dict[str, pd.DataFrame],
    report: dict,
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Save all pipeline outputs with metadata for reproducibility.

    File layout
    -----------
    data/
        prices_adjusted.parquet      — cleaned price matrix
        returns_simple.parquet       — simple (arithmetic) daily returns
        returns_log.parquet          — log daily returns
        pipeline_metadata.json       — run metadata + quality report
        checksums.json               — SHA-256 of every parquet file

    Why Parquet?
        Parquet is the institutional standard for columnar financial data.
        It preserves dtypes (including DatetimeIndex), compresses ~5× vs CSV,
        and reads 10–50× faster.  Never use CSV for production pipelines.

    Why checksums?
        Checksums let you verify that data files have not been corrupted or
        accidentally overwritten between pipeline runs.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving artefacts to {data_dir} …")

    files: dict[str, Path] = {
        "prices_adjusted": data_dir / "prices_adjusted.parquet",
        "returns_simple":  data_dir / "returns_simple.parquet",
        "returns_log":     data_dir / "returns_log.parquet",
    }

    prices.to_parquet(files["prices_adjusted"])
    returns["simple"].to_parquet(files["returns_simple"])
    returns["log"].to_parquet(files["returns_log"])

    # --- Checksums ---
    checksums: dict[str, str] = {}
    for name, path in files.items():
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[name] = sha256
        log.info(f"  {path.name}: {sha256[:16]}…")

    (data_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )

    # --- Pipeline metadata ---
    metadata = {
        "pipeline_version": "2.0.0",
        "run_timestamp":    datetime.utcnow().isoformat() + "Z",
        "start_date":       str(prices.index.min().date()),
        "end_date":         str(prices.index.max().date()),
        "n_trading_days":   int(prices.shape[0]),
        "n_tickers":        int(prices.shape[1]),
        "tickers":          prices.columns.tolist(),
        "quality_report":   {
            k: v for k, v in report.items()
            if k not in ("missing_pct_before", "missing_pct_after")  # too verbose
        },
    }
    (data_dir / "pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    log.info("All artefacts saved successfully.")


# ---------------------------------------------------------------------------
# Step 5 – Descriptive diagnostics (logged, not printed)
# ---------------------------------------------------------------------------

def log_summary_statistics(prices: pd.DataFrame, returns: dict[str, pd.DataFrame]) -> None:
    """Print a concise quality summary to the log."""
    sr = returns["simple"]
    ann_vol = sr.std() * np.sqrt(252)

    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    log.info(f"  Price matrix  : {prices.shape[0]:,} days × {prices.shape[1]} tickers")
    log.info(f"  Date range    : {prices.index[0].date()} → {prices.index[-1].date()}")
    log.info(f"  Median NaN %  : {prices.isna().mean().median():.4%}")
    log.info(f"  Median ann vol: {ann_vol.median():.2%}")
    log.info(f"  Min  ann vol  : {ann_vol.min():.2%}  ({ann_vol.idxmin()})")
    log.info(f"  Max  ann vol  : {ann_vol.max():.2%}  ({ann_vol.idxmax()})")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    tickers: Optional[list[str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run the full Phase-2 data pipeline end-to-end.

    Returns
    -------
    prices  : cleaned adjusted-close price DataFrame
    returns : dict with 'simple' and 'log' return DataFrames
    """
    if tickers is None:
        tickers = get_universe()

    log.info(f"Phase 2 pipeline starting | universe = {len(tickers)} tickers")

    # 1. Download
    raw_prices = download_prices(tickers, start=start, end=end)

    # 2. Validate & clean
    prices, report = validate_and_clean(raw_prices)

    # 3. Compute returns
    returns = compute_returns(prices)

    # 4. Save
    save_artefacts(prices, returns, report, data_dir=data_dir)

    # 5. Diagnostics
    log_summary_statistics(prices, returns)

    return prices, returns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices, returns = run_pipeline()
