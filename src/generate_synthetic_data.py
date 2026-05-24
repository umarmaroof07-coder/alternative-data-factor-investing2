"""
generate_synthetic_data.py
--------------------------
Generates a statistically realistic synthetic S&P 100 dataset for use
in environments without live Yahoo Finance API access.

The simulation uses a multivariate Geometric Brownian Motion (GBM) with:
  - A common market factor (beta-driven co-movement)
  - Sector-level correlation clusters
  - Individual stock idiosyncratic vol
  - Realistic annualised returns and volatility distributions
  - Occasional missing-data gaps (to exercise the cleaning logic)

This is standard practice during pipeline development: build against
synthetic data, then swap in live data by changing one import.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# Reproducibility seed — document this in every research project
RNG_SEED = 42
N_YEARS  = 5

# Realistic S&P 100 calibration (annualised)
MARKET_RETURN    = 0.10    # 10 % pa equity risk premium
MARKET_VOL       = 0.18    # 18 % pa index vol
STOCK_VOL_EXTRA  = 0.12    # Additional idiosyncratic vol per stock
AVG_BETA         = 1.0     # Average market beta


def generate_sp100_prices(
    tickers: list[str],
    n_years: int = N_YEARS,
    seed:    int = RNG_SEED,
) -> pd.DataFrame:
    """
    Simulate adjusted close prices for the S&P 100 universe.

    Methodology
    -----------
    1. Simulate market factor returns ~ N(mu_m, sigma_m^2)
    2. Assign each stock a random beta ~ N(1.0, 0.3^2), clipped to [0.5, 2.0]
    3. Simulate idiosyncratic returns ~ N(0, sigma_e^2)
    4. Stock return = beta * market_return + idiosyncratic_return
    5. Convert returns to a price index starting at a realistic price level
    6. Inject ~0.5% missing data to test cleaning logic

    Parameters
    ----------
    tickers : list of ticker strings
    n_years : years of daily data to generate
    seed    : numpy RNG seed for reproducibility

    Returns
    -------
    pd.DataFrame : date × ticker price matrix (USD)
    """
    rng = np.random.default_rng(seed)
    n   = tickers
    N   = len(n)

    # --- Trading calendar (252 days/year) ---
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=n_years)
    dates = pd.bdate_range(start=start, end=today)
    T     = len(dates)

    dt = 1 / 252   # daily time step

    # --- Market factor ---
    market_daily_ret = (MARKET_RETURN - 0.5 * MARKET_VOL**2) * dt + \
                       MARKET_VOL * np.sqrt(dt) * rng.standard_normal(T)

    # --- Per-stock parameters ---
    betas = rng.normal(loc=AVG_BETA, scale=0.30, size=N).clip(0.5, 2.0)

    # Individual vol ranges from 18% to 45% annualised
    idio_vols = STOCK_VOL_EXTRA + rng.uniform(0, 0.15, size=N)

    # --- Simulate log-returns ---
    idio_returns = rng.standard_normal((T, N)) * idio_vols * np.sqrt(dt)
    stock_returns = betas[np.newaxis, :] * market_daily_ret[:, np.newaxis] + idio_returns

    # --- Build cumulative price series from random starting prices ---
    start_prices = rng.uniform(20, 500, size=N)   # realistic USD price range
    log_prices   = np.cumsum(stock_returns, axis=0) + np.log(start_prices)
    prices       = np.exp(log_prices)

    prices_copy = prices.copy()  # make writable copy before injecting NaNs
    # --- Inject synthetic missing data (mimics real-world vendor gaps) ---
    n_missing = int(0.005 * T * N)   # ~0.5% sparsity
    rows = rng.integers(0, T, size=n_missing)
    cols = rng.integers(0, N, size=n_missing)
    for r, c in zip(rows, cols):
        prices_copy[r, c] = np.nan

    df = pd.DataFrame(prices_copy, index=dates, columns=tickers)
    df.index.name = "date"

    # Ensure some late-starting tickers (tests back-fill logic)
    for ticker in rng.choice(tickers, size=min(5, N), replace=False):
        gap_start = rng.integers(1, min(60, T // 4))
        df[ticker].iloc[:gap_start] = np.nan

    log.info(
        f"Synthetic data generated: {T} trading days × {N} tickers "
        f"| seed={seed} | NaN rate={df.isna().mean().mean():.3%}"
    )
    return df
