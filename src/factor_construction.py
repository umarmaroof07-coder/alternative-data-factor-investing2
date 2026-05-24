"""
factor_construction.py
----------------------
Phase 3: Institutional-grade factor construction for a multi-factor
equity strategy on the S&P 100 universe.

Factors built
-------------
1. Momentum (12-1)      — 12-month total return, skipping last month
2. Low Volatility       — rolling 63-day realised volatility (inverted)
3. Composite Score      — equal-weighted combination of normalised factors

All factors are:
  - Computed using only information available at each date (no look-ahead)
  - Normalised via the standard quant pipeline: winsorise → rank → z-score
  - Saved as Parquet for Phase 4 (portfolio construction)

Academic references
-------------------
Momentum : Jegadeesh & Titman (1993), Carhart (1997)
Low Vol  : Baker, Bradley & Wurgler (2011), Frazzini & Pedersen (2014)
Composite: Fama & French (2015), Asness, Frazzini & Pedersen (2019)

Author   : Quant Research Team
Phase    : 3 — Factor Construction
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from factor_utils import (
    combine_factors,
    cross_sectional_rank,
    factor_autocorrelation,
    factor_coverage,
    factor_dispersion,
    normalise_factor,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Configuration — all look-back windows in one place
# ---------------------------------------------------------------------------

# Momentum: 12-month formation, skip 1 month
MOM_FORMATION_DAYS : int = 252   # ~12 calendar months of trading days
MOM_SKIP_DAYS      : int = 21    # ~1 month skip (avoids short-term reversal)

# Volatility: 63-day rolling window ≈ 1 quarter
VOL_WINDOW_DAYS    : int = 63

# Minimum non-NaN observations required to compute a valid signal
MIN_OBS_MOMENTUM   : int = 200   # need at least 200 of 252 days
MIN_OBS_VOL        : int = 42    # need at least 42 of 63 days (≈ 2/3)

# Factor weights in composite (equal by default)
FACTOR_WEIGHTS: dict[str, float] = {
    "momentum": 1.0,
    "low_vol":  1.0,
}


# ===========================================================================
# Factor 1: 12-1 Momentum
# ===========================================================================

def build_momentum_factor(
    prices: pd.DataFrame,
    formation_days: int = MOM_FORMATION_DAYS,
    skip_days:      int = MOM_SKIP_DAYS,
    min_obs:        int = MIN_OBS_MOMENTUM,
) -> pd.DataFrame:
    """
    Build the 12-1 momentum factor (raw and normalised).

    ECONOMIC RATIONALE
    ------------------
    Momentum is the empirical finding that stocks that have performed well
    over the past 12 months tend to continue outperforming over the next
    1–3 months.  This is one of the most robust anomalies in finance,
    documented in 40+ years of global equity data.

    CONSTRUCTION
    ------------
    Raw signal = P(t - skip) / P(t - formation) - 1

    The 1-month skip ("12-1" rather than "12-0") is critical.
    The most recent month exhibits SHORT-TERM REVERSAL — stocks that went
    up last month tend to mean-revert.  Including the last month actually
    *hurts* the signal.  This was documented by Jegadeesh (1990) and is
    why every serious momentum factor skips the most recent 21 trading days.

    LOOK-AHEAD BIAS CHECK
    ---------------------
    At date t we use:
        P(t - skip_days)          ← last known price before recent reversal
        P(t - formation_days)     ← price 12 months ago
    Both are strictly in the past at date t.  No future prices are used.

    Parameters
    ----------
    prices         : T × N adjusted close price DataFrame
    formation_days : look-back window (default 252 = ~12 months)
    skip_days      : recent period to skip (default 21 = ~1 month)
    min_obs        : minimum non-NaN prices in window for valid signal

    Returns
    -------
    dict with:
        'raw'        : T × N DataFrame of raw momentum returns
        'normalised' : T × N DataFrame of cross-sectionally normalised scores
    """
    log.info(
        f"Building 12-1 momentum | "
        f"formation={formation_days}d, skip={skip_days}d, min_obs={min_obs}"
    )

    # Step 1: compute raw 12-1 momentum
    # P(t - skip) / P(t - formation) - 1
    # Using shift(): shift(n) gives you the value from n periods ago.
    # shift(skip) = price 1 month ago  (numerator of return)
    # shift(formation) = price 12 months ago (denominator of return)
    price_end   = prices.shift(skip_days)       # "end" of formation window
    price_start = prices.shift(formation_days)  # "start" of formation window

    raw_momentum = price_end / price_start - 1

    # Step 2: Zero out signals where we don't have enough price history.
    # We require at least min_obs valid prices in the formation window.
    # rolling().count() counts non-NaN values — a vectorised, look-ahead-safe
    # way to enforce the minimum-observations rule.
    valid_obs = prices.shift(skip_days).rolling(
        window=formation_days - skip_days,
        min_periods=1,
    ).count()

    raw_momentum = raw_momentum.where(valid_obs >= min_obs, other=np.nan)

    # Step 3: Cross-sectional normalisation (winsorise → rank → z-score)
    # Apply normalise_factor() row-wise: each row is one date's cross-section.
    # axis=1 means "apply function to each row" — i.e. across the N tickers.
    normalised = raw_momentum.apply(normalise_factor, axis=1)

    log.info(
        f"Momentum built | "
        f"valid rows: {raw_momentum.dropna(how='all').shape[0]} / {len(prices)}"
    )
    return {"raw": raw_momentum, "normalised": normalised}


# ===========================================================================
# Factor 2: Rolling Low Volatility
# ===========================================================================

def build_low_vol_factor(
    log_returns: pd.DataFrame,
    window:  int = VOL_WINDOW_DAYS,
    min_obs: int = MIN_OBS_VOL,
) -> dict[str, pd.DataFrame]:
    """
    Build the Low Volatility factor (rolling 63-day realised vol, inverted).

    ECONOMIC RATIONALE
    ------------------
    The Low Volatility anomaly is the finding that *lower risk* stocks earn
    *higher* risk-adjusted returns than theory (CAPM) predicts.  This violates
    the basic Finance 101 risk-return trade-off and is one of the most
    economically puzzling effects in asset pricing.

    Explanations include: leverage constraints (Frazzini & Pedersen 2014),
    lottery demand (Bali, Brown & Murray 2017), and benchmark-relative mandates
    that force institutional managers to hold high-beta stocks.

    CONSTRUCTION
    ------------
    Raw vol  = std(log_returns, window=63 days) × sqrt(252)   [annualised]
    Raw signal = -raw_vol   (invert so higher score = lower vol = more desirable)

    WHY LOG RETURNS FOR VOL?
    ------------------------
    Log returns are used here (not simple returns) because:
    1. Variance of log returns is stationary and additive across time.
    2. The formula sqrt(252) × daily_std correctly annualises log-return std.
    3. Simple return variance is not time-additive, making annualisation
       approximate rather than exact.

    WHY 63 DAYS (~1 QUARTER)?
    -------------------------
    - Short windows (e.g. 21 days) are too noisy; one earnings release
      dominates the estimate.
    - Long windows (e.g. 252 days) are too slow; they don't detect
      regime changes in a stock's risk.
    - 63 days balances stability with responsiveness.  This matches the
      window used in most Barra and MSCI factor models.

    LOOK-AHEAD BIAS CHECK
    ---------------------
    rolling(window=63).std() on date t uses returns from [t-63, t-1].
    Return on day t = P(t)/P(t-1) - 1. Both prices are known at close of t.
    No future returns are used.  ✓

    Parameters
    ----------
    log_returns : T × N log-return DataFrame
    window      : rolling window in trading days (default 63)
    min_obs     : minimum valid returns in window (default 42)

    Returns
    -------
    dict with:
        'raw_vol'    : T × N annualised volatility (positive, lower = better)
        'raw'        : T × N inverted signal (higher = lower vol = better)
        'normalised' : T × N cross-sectionally normalised scores
    """
    log.info(
        f"Building Low Volatility factor | "
        f"window={window}d, min_obs={min_obs}"
    )

    # Step 1: Rolling realised volatility (annualised)
    raw_vol = (
        log_returns
        .rolling(window=window, min_periods=min_obs)
        .std()
        .mul(np.sqrt(252))   # annualise: multiply daily std by sqrt(252)
    )

    # Step 2: Invert — low vol stocks get high scores
    # We negate so the factor loads positively on "low volatility"
    raw_signal = -raw_vol

    # Step 3: Cross-sectional normalisation
    normalised = raw_signal.apply(normalise_factor, axis=1)

    # Vol stats for logging
    mean_vol = raw_vol.mean().mean()
    log.info(
        f"Low vol built | mean annualised vol across universe: {mean_vol:.2%} | "
        f"valid rows: {raw_vol.dropna(how='all').shape[0]} / {len(log_returns)}"
    )
    return {"raw_vol": raw_vol, "raw": raw_signal, "normalised": normalised}


# ===========================================================================
# Factor 3: Composite Score
# ===========================================================================

def build_composite_score(
    momentum_norm: pd.DataFrame,
    low_vol_norm:  pd.DataFrame,
    weights: dict[str, float] = FACTOR_WEIGHTS,
) -> pd.DataFrame:
    """
    Combine normalised factor scores into a composite signal.

    COMBINATION METHODOLOGY
    -----------------------
    For each date t and each stock i:
        composite(i,t) = w_mom × mom_score(i,t) + w_vol × vol_score(i,t)

    After weighting, we re-normalise the composite to ≈ N(0,1).
    This ensures the composite behaves like any individual factor when
    passed to a portfolio construction step.

    WHY NOT JUST AVERAGE THE RAW SIGNALS?
    --------------------------------------
    Momentum and low-volatility have different raw scales, distributions,
    and sign conventions.  Averaging raw signals would give implicit weight
    to the factor with the largest absolute values.  By normalising each
    factor to z-scores first and then averaging, we give each factor equal
    statistical influence — which is what "equal weighting" actually means.

    EQUAL-WEIGHT RATIONALE
    ----------------------
    In-sample optimised factor weights almost always over-fit.  Equal
    weighting is the max-entropy prior: we assign equal credence to each
    signal because we have no in-sample edge in distinguishing which will
    outperform out-of-sample.  This is documented in DeMiguel et al. (2007)
    "Optimal Versus Naive Diversification" — the result extends to factor
    weights as well as asset weights.

    Parameters
    ----------
    momentum_norm : T × N normalised momentum z-scores
    low_vol_norm  : T × N normalised low-vol z-scores
    weights       : dict of factor weights (normalised internally)

    Returns
    -------
    pd.DataFrame — T × N composite z-scores
    """
    log.info(f"Building composite score | weights: {weights}")

    # Align on the intersection of dates where both factors exist
    common_dates = momentum_norm.index.intersection(low_vol_norm.index)
    mom = momentum_norm.loc[common_dates]
    vol = low_vol_norm.loc[common_dates]

    # Apply combine_factors row-by-row
    composite_rows = []
    for date in common_dates:
        row_mom = mom.loc[date]
        row_vol = vol.loc[date]

        # Both factors must be non-NaN for a stock to get a composite score
        valid = row_mom.notna() & row_vol.notna()
        if valid.sum() < 5:
            composite_rows.append(pd.Series(np.nan, index=mom.columns))
            continue

        row_composite = combine_factors(
            {"momentum": row_mom[valid], "low_vol": row_vol[valid]},
            weights=weights,
        )
        # Re-index to full universe (NaN for stocks missing either factor)
        composite_rows.append(row_composite.reindex(mom.columns))

    composite = pd.DataFrame(composite_rows, index=common_dates)
    composite.index.name = "date"
    composite.columns.name = "ticker"

    log.info(f"Composite built | shape: {composite.shape}")
    return composite


# ===========================================================================
# Diagnostics & reporting
# ===========================================================================

def compute_factor_diagnostics(
    factors: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Compute and log key diagnostics for each factor panel.

    Metrics
    -------
    coverage_mean   : average fraction of universe with valid scores
    dispersion_mean : average cross-sectional std of z-scores (should ≈ 1)
    autocorr_1d     : 1-day rank autocorrelation (measures signal turnover)
    autocorr_21d    : 21-day rank autocorrelation (monthly turnover)
    """
    diagnostics = {}

    for name, df in factors.items():
        cov  = factor_coverage(df).mean()
        disp = factor_dispersion(df).mean()
        ac1  = factor_autocorrelation(df, lag=1)
        ac21 = factor_autocorrelation(df, lag=21)

        diagnostics[name] = {
            "coverage_mean":    round(float(cov),  4),
            "dispersion_mean":  round(float(disp), 4),
            "autocorr_1d":      round(float(ac1),  4),
            "autocorr_21d":     round(float(ac21), 4),
        }

        log.info(
            f"[{name:>12}] coverage={cov:.1%}  disp={disp:.3f}  "
            f"AC(1d)={ac1:.3f}  AC(21d)={ac21:.3f}"
        )

    return diagnostics


# ===========================================================================
# Persistence
# ===========================================================================

def save_factors(
    factors: dict[str, pd.DataFrame],
    diagnostics: dict[str, dict],
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Save all factor panels and diagnostics to disk.

    File layout
    -----------
    data/
        factor_momentum.parquet     — normalised momentum z-scores
        factor_low_vol.parquet      — normalised low-vol z-scores
        factor_composite.parquet    — composite factor z-scores
        factor_raw_momentum.parquet — raw (un-normalised) momentum returns
        factor_raw_vol.parquet      — raw annualised volatility
        factor_diagnostics.json     — quality metrics for each factor
        factor_metadata.json        — run metadata + config
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving factor panels to {data_dir} …")

    file_map = {
        "factor_momentum":     factors["momentum"],
        "factor_low_vol":      factors["low_vol"],
        "factor_composite":    factors["composite"],
        "factor_raw_momentum": factors["raw_momentum"],
        "factor_raw_vol":      factors["raw_vol"],
    }

    for name, df in file_map.items():
        path = data_dir / f"{name}.parquet"
        df.to_parquet(path)
        log.info(f"  Saved {path.name} | {df.shape}")

    # Diagnostics JSON
    (data_dir / "factor_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    # Metadata
    meta = {
        "phase": "3 - Factor Construction",
        "pipeline_version": "3.0.0",
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "factors": list(file_map.keys()),
        "config": {
            "mom_formation_days": MOM_FORMATION_DAYS,
            "mom_skip_days":      MOM_SKIP_DAYS,
            "vol_window_days":    VOL_WINDOW_DAYS,
            "factor_weights":     FACTOR_WEIGHTS,
        },
        "diagnostics": diagnostics,
    }
    (data_dir / "factor_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    log.info("Factor metadata saved.")


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_factor_pipeline(
    prices:      pd.DataFrame | None = None,
    log_returns: pd.DataFrame | None = None,
    data_dir:    Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Run the full Phase-3 factor construction pipeline.

    1. Load Phase-2 data (if not passed directly)
    2. Build 12-1 momentum factor
    3. Build rolling low-volatility factor
    4. Build composite score
    5. Compute diagnostics
    6. Save everything to data/

    Returns
    -------
    dict with keys: momentum, low_vol, composite, raw_momentum, raw_vol
    """
    # --- Load data ---
    if prices is None:
        log.info("Loading Phase-2 price data …")
        prices = pd.read_parquet(data_dir / "prices_adjusted.parquet")

    if log_returns is None:
        log.info("Loading Phase-2 log return data …")
        log_returns = pd.read_parquet(data_dir / "returns_log.parquet")

    log.info(
        f"Input: {prices.shape[0]:,} days × {prices.shape[1]} tickers | "
        f"range {prices.index[0].date()} → {prices.index[-1].date()}"
    )

    # --- Factor 1: Momentum ---
    mom_out = build_momentum_factor(prices)

    # --- Factor 2: Low Volatility ---
    vol_out = build_low_vol_factor(log_returns)

    # --- Factor 3: Composite ---
    composite = build_composite_score(
        mom_out["normalised"],
        vol_out["normalised"],
    )

    # --- Package ---
    factors = {
        "momentum":     mom_out["normalised"],
        "low_vol":      vol_out["normalised"],
        "composite":    composite,
        "raw_momentum": mom_out["raw"],
        "raw_vol":      vol_out["raw_vol"],
    }

    # --- Diagnostics ---
    log.info("Computing factor diagnostics …")
    diag_factors = {
        k: factors[k] for k in ("momentum", "low_vol", "composite")
    }
    diagnostics = compute_factor_diagnostics(diag_factors)

    # --- Save ---
    save_factors(factors, diagnostics, data_dir=data_dir)

    log.info("=" * 60)
    log.info("PHASE 3 COMPLETE")
    log.info("=" * 60)

    return factors


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    factors = run_factor_pipeline()
