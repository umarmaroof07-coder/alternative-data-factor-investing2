"""
sentiment_factor.py
-------------------
Phase 5b — Alternative Data Factor Construction

This module builds two alternative data factors:

FACTOR 1: Hiring Trends
-----------------------
  Raw signal  : Monthly job posting volume change (YoY %) per company
  Economic logic:
    Companies that are hiring aggressively tend to be:
    (a) Experiencing strong demand and revenue growth
    (b) Investing in future capacity (leading indicator)
    (c) Confident in their business outlook (management signal)

  Academic support:
    Eisfeldt & Papanikolaou (2013): "Organization Capital and the
    Cross-Section of Expected Returns" — human capital investment
    predicts stock returns.  Subsequent work using LinkedIn data
    (Green et al. 2019) confirms job posting growth predicts sales
    growth and positive stock returns.

  Data cadence: Monthly (job boards update continuously but we
    aggregate to month-end to avoid look-ahead bias)

  Look-ahead bias control:
    We use postings from month t-1 to predict returns in month t+1.
    The extra 1-month lag beyond the natural data lag is conservative
    but necessary because job posting aggregators often revise their
    data retroactively.  We lose 1 month of signal strength in exchange
    for cleaner out-of-sample attribution.

FACTOR 2: News Sentiment
------------------------
  Raw signal  : Rolling 21-day volume-weighted news sentiment score
  Economic logic:
    Persistent positive news coverage predicts short-to-medium term
    price appreciation through:
    (a) Analyst attention → higher analyst coverage → better information
        diffusion (Merton 1987 investor recognition hypothesis)
    (b) Retail investor attention → demand pressure (Barber & Odean 2008)
    (c) Genuine information content about business fundamentals

  Academic support:
    Tetlock (2007): "Giving Content to Investor Sentiment" — negative
    media sentiment predicts lower stock returns.
    Loughran & McDonald (2011): Finance-specific word lists improve
    textual analysis of 10-Ks.
    RavenPack and Bloomberg studies: 1-day and 1-month IC of 0.02–0.06.

  Data cadence: Daily (we aggregate to monthly for the factor panel)

  Look-ahead bias control:
    We use sentiment from trading days [t-21, t] to predict returns
    starting at t+1.  The sentiment aggregation window ends on the
    decision date, so no future news is included.

NORMALISATION PIPELINE
-----------------------
Both factors go through the same 3-step normalisation as Phase 3:
  1. Winsorise at 5th / 95th percentile (same threshold as traditional factors)
  2. Cross-sectional percentile rank → (0, 1)
  3. Z-score → approximately N(0, 1)

This ensures the alt-data factors are on the same scale as the
traditional factors when they are combined in the composite.

Author  : Quant Research Team
Phase   : 5 — Alternative Data Alpha Layer
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alternative_data import AltDataSimulator
from factor_utils import normalise_factor, factor_autocorrelation, factor_coverage

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ===========================================================================
# Factor 1: Hiring Trends
# ===========================================================================

def build_hiring_factor(
    tickers:        list[str],
    start_date:     str | pd.Timestamp,
    end_date:       str | pd.Timestamp,
    daily_returns:  pd.DataFrame,
    seed:           int   = 101,
    alpha_corr:     float = 0.09,
    publication_lag_months: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Build the Hiring Trends factor from synthetic job posting data.

    Construction pipeline
    ---------------------
    1. Simulate monthly job posting volume via AltDataSimulator (AR(1) + sectors)
    2. Compute YoY growth rate: (postings_t / postings_{t-12}) - 1
       WHY YoY? Eliminates seasonality (hiring always spikes in Jan; that's
       not a signal, it's a calendar effect).
    3. Apply publication lag: shift forward by publication_lag_months
       WHY? Job posting aggregators typically publish with a 2-4 week lag.
       We use a 1-month lag to be conservative — we can only trade on
       data we actually have in hand.
    4. Interpolate to daily frequency via forward-fill
       WHY? The rest of the pipeline (portfolio.py, backtest.py) operates
       on daily dates.  FFILL is correct here: we hold the last known
       monthly reading until the next one arrives.
    5. Cross-sectional normalisation: winsorise → rank → z-score

    Parameters
    ----------
    tickers                 : universe of tickers
    start_date / end_date   : date range (matches price data)
    daily_returns           : daily return panel (for alpha injection)
    seed                    : RNG seed for reproducibility
    alpha_corr              : target rank-IC between signal and fwd returns
    publication_lag_months  : months to lag the signal (data availability lag)

    Returns
    -------
    dict with:
        'raw_monthly'  : T_monthly × N raw hiring growth rates
        'raw_daily'    : T_daily × N forward-filled raw signal
        'normalised'   : T_daily × N normalised z-scores
        'monthly_norm' : T_monthly × N normalised scores (for diagnostics)
    """
    log.info(
        f"Building Hiring Trends factor | "
        f"lag={publication_lag_months}m | alpha_corr={alpha_corr:.2f}"
    )

    # --- Step 1: Simulate raw monthly posting volume ---
    simulator = AltDataSimulator(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        signal_name="HiringVolume",
        seed=seed,
        alpha_corr=alpha_corr,
        ar_coef=0.82,      # hiring is sticky: companies don't flip hiring plans monthly
    )
    # The base level represents log-scale job posting volume (mean-0, std-1)
    raw_level = simulator.generate(future_returns=daily_returns)

    # --- Step 2: YoY growth (12-month difference of the log-level) ---
    # In practice: YoY_growth = postings_t / postings_{t-12} - 1
    # In log-space: log(p_t) - log(p_{t-12}) ≈ YoY growth rate
    # We compute on the standardised level directly (equivalent)
    yoy_growth = raw_level - raw_level.shift(12)

    # Need at least 12 months of history → first 12 rows are NaN (correct)
    log.info(
        f"YoY hiring growth computed | "
        f"valid rows: {yoy_growth.dropna(how='all').shape[0]} / {len(yoy_growth)}"
    )

    # --- Step 3: Publication lag ---
    # Shift forward: data available at month t+lag, not month t
    yoy_lagged = yoy_growth.shift(publication_lag_months)

    # --- Step 4: Normalise at monthly frequency ---
    monthly_norm = yoy_lagged.apply(normalise_factor, axis=1)

    # --- Step 5: Interpolate to daily (forward-fill monthly → daily) ---
    # Build a daily index matching the price data
    daily_index = daily_returns.index

    # Reindex monthly to daily, then ffill
    raw_daily = (
        yoy_lagged
        .reindex(daily_index, method="ffill")
    )
    normalised_daily = (
        monthly_norm
        .reindex(daily_index, method="ffill")
    )

    log.info(
        f"Hiring factor built | daily shape: {normalised_daily.shape} | "
        f"NaN rate: {normalised_daily.isna().mean().mean():.1%}"
    )

    return {
        "raw_monthly":  yoy_lagged,
        "raw_daily":    raw_daily,
        "normalised":   normalised_daily,
        "monthly_norm": monthly_norm,
    }


# ===========================================================================
# Factor 2: News Sentiment
# ===========================================================================

def build_sentiment_factor(
    tickers:       list[str],
    start_date:    str | pd.Timestamp,
    end_date:      str | pd.Timestamp,
    daily_returns: pd.DataFrame,
    seed:          int   = 202,
    alpha_corr:    float = 0.07,
    smoothing_days: int  = 21,
) -> dict[str, pd.DataFrame]:
    """
    Build the News Sentiment factor from synthetic NLP sentiment scores.

    Construction pipeline
    ---------------------
    1. Simulate daily article-level sentiment (range: -1 to +1)
       via AltDataSimulator with higher frequency and lower persistence
       than hiring (news moves faster than HR decisions).

    2. Compute volume-weighted rolling 21-day mean
       WHY? A single headline has high noise.  Rolling aggregation:
         (a) Reduces noise while preserving signal persistence
         (b) Mimics real vendor products (e.g. RavenPack CSS-S score)
         (c) 21 days = 1 month aligns with our rebalancing horizon

    3. Momentum correction: subtract 3-month lagged sentiment
       WHY? Raw sentiment has a mean-reversion tendency at horizons
       longer than 1 month (the "news reversal" documented by Da et al.
       2014).  We subtract the 3-month lagged score to create a sentiment
       *change* factor that has better persistence than raw level.

    4. Cross-sectional normalisation: winsorise → rank → z-score

    Parameters
    ----------
    tickers         : universe of tickers
    start_date/end  : date range
    daily_returns   : daily return panel (for alpha injection)
    seed            : RNG seed
    alpha_corr      : target rank-IC
    smoothing_days  : rolling mean window for noise reduction

    Returns
    -------
    dict with:
        'raw_daily'   : T_daily × N raw smoothed sentiment scores
        'normalised'  : T_daily × N normalised z-scores
    """
    log.info(
        f"Building News Sentiment factor | "
        f"smoothing={smoothing_days}d | alpha_corr={alpha_corr:.2f}"
    )

    # --- Step 1: Simulate daily sentiment on a daily grid ---
    # Sentiment moves faster than hiring → lower AR coefficient
    daily_index = daily_returns.index
    start       = daily_index[0]
    end         = daily_index[-1]

    simulator = AltDataSimulator(
        tickers=tickers,
        start_date=start,
        end_date=end,
        signal_name="NewsSentiment",
        seed=seed,
        alpha_corr=alpha_corr,
        ar_coef=0.65,    # sentiment mean-reverts faster than hiring plans
    )

    # Override: generate on daily index (not monthly)
    # We adapt the simulator's monthly output to daily by directly
    # re-simulating at daily frequency here
    rng = np.random.default_rng(seed + 99)
    T   = len(daily_index)
    N   = len(tickers)
    phi = 0.65
    sigma = np.sqrt(1 - phi**2)

    # Simple AR(1) daily panel with sector factor
    from alternative_data import TICKER_TO_SECTOR
    sector_labels  = [TICKER_TO_SECTOR.get(t, "Other") for t in tickers]
    unique_sectors = list(set(sector_labels))

    # Market factor
    mkt = np.zeros(T)
    mkt[0] = rng.standard_normal()
    for t in range(1, T):
        mkt[t] = phi * mkt[t-1] + sigma * rng.standard_normal()

    # Sector factors
    sect_factors = {}
    for s in unique_sectors:
        sf = np.zeros(T)
        sf[0] = rng.standard_normal()
        for t in range(1, T):
            sf[t] = phi * sf[t-1] + sigma * rng.standard_normal()
        sect_factors[s] = sf

    # Company quality
    quality = rng.normal(0, 0.4, N)

    # Idiosyncratic
    idio = np.zeros((T, N))
    idio[0] = rng.standard_normal(N)
    for t in range(1, T):
        idio[t] = phi * idio[t-1] + sigma * rng.standard_normal(N)

    panel = np.zeros((T, N))
    for i, (ticker, sector) in enumerate(zip(tickers, sector_labels)):
        panel[:, i] = (
            0.25 * mkt
            + 0.20 * sect_factors[sector]
            + 0.15 * quality[i]
            + 0.40 * idio[:, i]   # more idiosyncratic than hiring
        )

    # Alpha injection using forward returns
    if alpha_corr > 0:
        fwd_ret = daily_returns.shift(-21)   # 21-day forward returns
        for i, ticker in enumerate(tickers):
            if ticker in fwd_ret.columns:
                fwd = fwd_ret[ticker].reindex(daily_index).fillna(0).values
                fwd_rank = pd.Series(fwd).rank(pct=True).values
                panel[:, i] = (
                    (1 - alpha_corr) * panel[:, i]
                    + alpha_corr * (fwd_rank - 0.5) / 0.3
                )

    # Add measurement noise
    panel += 0.10 * rng.standard_normal(panel.shape)

    # 2% random missing
    missing = rng.random(panel.shape) < 0.02
    panel[missing] = np.nan

    raw_df = pd.DataFrame(panel, index=daily_index, columns=tickers)
    raw_df.index.name = "date"
    raw_df.columns.name = "ticker"

    # --- Step 2: Rolling 21-day smoothing ---
    smoothed = raw_df.rolling(window=smoothing_days, min_periods=10).mean()

    # --- Step 3: Sentiment momentum (change vs 63-day lag) ---
    # Captures "improving sentiment" rather than "high absolute sentiment"
    # Companies with rising sentiment outperform those with falling sentiment
    sent_change = smoothed - smoothed.shift(63)

    # --- Step 4: Cross-sectional normalisation (applied row-wise) ---
    normalised = sent_change.apply(normalise_factor, axis=1)

    log.info(
        f"Sentiment factor built | shape: {normalised.shape} | "
        f"NaN rate: {normalised.isna().mean().mean():.1%} | "
        f"AC(21d): {factor_autocorrelation(normalised, lag=21):.3f}"
    )

    return {
        "raw_daily":  smoothed,
        "sent_change": sent_change,
        "normalised": normalised,
    }


# ===========================================================================
# Combined alternative data score
# ===========================================================================

def build_alt_composite(
    hiring_norm:   pd.DataFrame,
    sentiment_norm: pd.DataFrame,
    hiring_weight:   float = 0.50,
    sentiment_weight: float = 0.50,
) -> pd.DataFrame:
    """
    Combine hiring and sentiment into a single alternative data score.

    WHY EQUAL-WEIGHT BETWEEN THE TWO ALT FACTORS?
    -----------------------------------------------
    Same rationale as Phase 3: in-sample optimised weights over-fit.
    Equal weight is the max-entropy prior when we have no strong a priori
    reason to prefer one alt signal over the other.

    The combined alt-data score is then combined with the traditional
    composite in phase5_alt_factors.py with weights 60% traditional / 40% alt.

    Parameters
    ----------
    hiring_norm       : T × N normalised hiring z-scores (daily)
    sentiment_norm    : T × N normalised sentiment z-scores (daily)
    hiring_weight     : weight for hiring factor (default 0.5)
    sentiment_weight  : weight for sentiment factor (default 0.5)

    Returns
    -------
    pd.DataFrame : T × N combined alternative data z-scores
    """
    log.info(
        f"Combining alt-data factors | "
        f"hiring={hiring_weight:.0%} / sentiment={sentiment_weight:.0%}"
    )

    # Normalise weights to sum to 1
    total = hiring_weight + sentiment_weight
    w_h   = hiring_weight / total
    w_s   = sentiment_weight / total

    # Align on common dates and tickers
    common_dates   = hiring_norm.index.intersection(sentiment_norm.index)
    common_tickers = hiring_norm.columns.intersection(sentiment_norm.columns)

    h = hiring_norm.loc[common_dates, common_tickers]
    s = sentiment_norm.loc[common_dates, common_tickers]

    # Weighted average; require both to be non-NaN for a valid combined score
    combined = w_h * h + w_s * s

    # Mark NaN where either input is NaN (conservative: don't impute)
    both_valid = h.notna() & s.notna()
    combined   = combined.where(both_valid, other=np.nan)

    # Re-normalise the composite (cross-sectionally) so it remains ≈ N(0,1)
    from factor_utils import cross_sectional_zscore
    combined = combined.apply(cross_sectional_zscore, axis=1)

    log.info(
        f"Alt composite built | shape: {combined.shape} | "
        f"NaN rate: {combined.isna().mean().mean():.1%}"
    )
    return combined


# ===========================================================================
# Diagnostics
# ===========================================================================

def alt_factor_diagnostics(
    factors: dict[str, pd.DataFrame],
    trad_composite: pd.DataFrame,
) -> dict:
    """
    Compute and log key diagnostics for the alt-data factors.

    Includes cross-factor correlations with the traditional composite
    (want low correlation → diversification value).
    """
    results = {}

    for name, df in factors.items():
        cov  = factor_coverage(df).mean()
        ac1  = factor_autocorrelation(df, lag=1)
        ac21 = factor_autocorrelation(df, lag=21)

        # Correlation with traditional composite
        common   = df.index.intersection(trad_composite.index)
        flat_alt = df.loc[common].stack().dropna()
        flat_trd = trad_composite.loc[common].stack().dropna()
        common_idx = flat_alt.index.intersection(flat_trd.index)

        if len(common_idx) > 50:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(flat_alt.loc[common_idx], flat_trd.loc[common_idx])
        else:
            rho = np.nan

        results[name] = {
            "coverage_mean":    round(float(cov), 4),
            "autocorr_1d":      round(float(ac1),  4),
            "autocorr_21d":     round(float(ac21), 4),
            "corr_vs_trad":     round(float(rho), 4) if not np.isnan(rho) else None,
        }

        log.info(
            f"[{name:>18}] cov={cov:.1%}  AC(1d)={ac1:.3f}  "
            f"AC(21d)={ac21:.3f}  ρ_vs_trad={rho:.3f}"
        )

    return results
