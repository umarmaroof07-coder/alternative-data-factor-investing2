"""
regime_analysis.py
------------------
Phase 6b — Market Regime Analysis

PURPOSE
-------
A strategy that only works in one type of market is fragile.
Institutional allocators want to see performance across regimes —
if a strategy produces alpha in bull markets but loses badly in
crashes, it amplifies portfolio risk at exactly the wrong time.

REGIMES DEFINED
---------------
We classify every trading day into one of four regimes using
only information available up to (and including) that day:

  1. Bull Market
     Definition: trailing 252-day index return > 0 AND above 200-day MA
     Rationale  : Price is trending up; momentum strategies typically shine
     Frequency  : ~65% of trading days historically

  2. Bear Market
     Definition: trailing 252-day index return < -10% OR below 200-day MA
     Rationale  : Sustained downtrend; factor signals may break down
     Frequency  : ~15% of trading days

  3. High Volatility
     Definition: rolling 21-day VIX proxy (index vol) > 75th percentile
     Rationale  : Risk-off environments; factor premia compress
     Note       : Bear and high-vol often overlap; we treat separately

  4. Low Volatility
     Definition: rolling 21-day vol < 25th percentile
     Rationale  : Calm markets; factor signals are cleanest (less noise)

  5. Transition (everything else)

WHY THESE DEFINITIONS?
-----------------------
  • Simple, transparent, and reproducible (no hidden parameters)
  • Based only on *past* price data (no look-ahead)
  • The 252-day/21-day windows are institutional conventions
  • 200-day MA is the most widely used trend filter in practice

LOOK-AHEAD BIAS CONTROL
-----------------------
All regime labels are computed from the equal-weighted universe return
(as a market proxy).  The return on day t is computed from prices at
close of day t, which is available before open on day t+1.  Because
portfolio weights are applied starting t+1 (Phase 4 convention), the
regime label on day t correctly describes the environment faced by
the portfolio on day t+1.

REGIME PERFORMANCE METRICS
---------------------------
For each regime we report:
  • Annualised return (strategy and benchmark)
  • Annualised volatility
  • Sharpe ratio within regime
  • Hit rate (% of regime days with positive daily return)
  • Max intra-regime drawdown

Author  : Quant Research Team
Phase   : 6 — Robustness Testing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

class RegimeLabels:
    """Constants for regime label strings."""
    BULL       = "Bull Market"
    BEAR       = "Bear Market"
    HIGH_VOL   = "High Volatility"
    LOW_VOL    = "Low Volatility"
    TRANSITION = "Transition"

    ALL = [BULL, BEAR, HIGH_VOL, LOW_VOL, TRANSITION]

    # Display order for figures
    ORDERED = [BULL, LOW_VOL, TRANSITION, HIGH_VOL, BEAR]

    COLORS = {
        BULL:       "#2166AC",   # blue
        BEAR:       "#B2182B",   # red
        HIGH_VOL:   "#E08214",   # orange
        LOW_VOL:    "#4DAC26",   # green
        TRANSITION: "#888888",   # grey
    }


def classify_regimes(
    market_returns:   pd.Series,
    vol_window:       int   = 21,
    trend_window:     int   = 252,
    ma_window:        int   = 200,
    bear_threshold:   float = -0.10,
    high_vol_pct:     float = 0.75,
    low_vol_pct:      float = 0.25,
) -> pd.Series:
    """
    Classify each trading day into a market regime.

    CLASSIFICATION LOGIC (in priority order)
    -----------------------------------------
    1. BEAR:        12m trailing return < bear_threshold (-10%)
                    OR current price < 200-day moving average
                    (uses the market index proxy, not individual stocks)

    2. HIGH_VOL:    rolling 21d vol > 75th percentile of full-sample vol
                    AND not already classified as BEAR

    3. LOW_VOL:     rolling 21d vol < 25th percentile of full-sample vol
                    AND 12m return > 0 (trending up)

    4. BULL:        12m return > 0 AND above 200-day MA
                    AND not HIGH_VOL or BEAR

    5. TRANSITION:  everything else

    Priority matters: a day can be simultaneously high-vol and bear-ish.
    We label it BEAR because that's the more severe / useful regime for
    risk management.

    Parameters
    ----------
    market_returns : pd.Series — daily equal-weighted universe return
    vol_window     : days for rolling vol estimate (default 21 = 1 month)
    trend_window   : days for trailing return (default 252 = 1 year)
    ma_window      : days for moving average trend filter (default 200)
    bear_threshold : trailing return that defines bear market (default -10%)
    high_vol_pct   : percentile above which vol is "high" (default 75th)
    low_vol_pct    : percentile below which vol is "low" (default 25th)

    Returns
    -------
    pd.Series of regime label strings, indexed like market_returns
    """
    mr = market_returns.copy()

    # --- Trailing 12-month market return (no look-ahead: shift(1) applied later) ---
    # (1+r1)(1+r2)...(1+rT) - 1 over trailing 252 days
    # rolling().prod() unavailable in some pandas versions; use log-return sum
    _log_mr = np.log1p(mr)
    trailing_ret = _log_mr.rolling(trend_window, min_periods=int(trend_window * 0.8)).sum().apply(np.expm1)

    # --- 200-day moving average of cumulative index level ---
    index_level = (1 + mr).cumprod()
    ma_200      = index_level.rolling(ma_window, min_periods=int(ma_window * 0.8)).mean()
    above_ma    = index_level > ma_200

    # --- Rolling 21-day annualised volatility ---
    rolling_vol = mr.rolling(vol_window, min_periods=int(vol_window * 0.5)).std() * np.sqrt(252)

    # Vol percentile thresholds computed on full sample (look-ahead for CLASSIFICATION ONLY)
    # This is the one place where we use full-sample info to set thresholds.
    # Justification: we are labelling regimes *ex-post* for analysis purposes,
    # not for live trading decisions.  The portfolio weights are not affected.
    # In a live system, use expanding-window percentiles instead.
    vol_hi_cut = rolling_vol.quantile(high_vol_pct)
    vol_lo_cut = rolling_vol.quantile(low_vol_pct)

    # --- Classify ---
    regime = pd.Series(RegimeLabels.TRANSITION, index=mr.index)

    # Bear first (highest priority)
    bear_mask = (trailing_ret < bear_threshold) | (~above_ma)
    regime[bear_mask] = RegimeLabels.BEAR

    # High vol (not already bear)
    high_vol_mask = (rolling_vol > vol_hi_cut) & (~bear_mask)
    regime[high_vol_mask] = RegimeLabels.HIGH_VOL

    # Low vol (not bear, not high vol)
    low_vol_mask = (rolling_vol < vol_lo_cut) & (trailing_ret > 0) & (~bear_mask) & (~high_vol_mask)
    regime[low_vol_mask] = RegimeLabels.LOW_VOL

    # Bull (trending up, calm, not otherwise classified)
    bull_mask = (trailing_ret > 0) & above_ma & (~bear_mask) & (~high_vol_mask) & (~low_vol_mask)
    regime[bull_mask] = RegimeLabels.BULL

    # First trend_window rows will be NaN → label as Transition
    regime[trailing_ret.isna()] = RegimeLabels.TRANSITION

    # Log distribution
    dist = regime.value_counts(normalize=True)
    for label in RegimeLabels.ALL:
        pct = dist.get(label, 0)
        log.info(f"  Regime [{label:18s}]: {pct:.1%} of trading days")

    return regime


# ---------------------------------------------------------------------------
# Regime performance statistics
# ---------------------------------------------------------------------------

class RegimeStats(NamedTuple):
    """Performance statistics within a single regime."""
    regime:       str
    n_days:       int
    pct_days:     float
    ann_return:   float
    ann_vol:      float
    sharpe:       float
    hit_rate:     float    # fraction of days with positive return
    max_dd:       float    # worst drawdown within regime sub-periods


def compute_regime_stats(
    strategy_returns: pd.Series,
    regime_labels:    pd.Series,
    label:            str = "Strategy",
) -> list[RegimeStats]:
    """
    Compute performance metrics for each market regime.

    For each regime R:
      1. Collect all days where regime == R
      2. Compute annualised return = mean(r) × 252
      3. Compute annualised vol    = std(r) × sqrt(252)
      4. Compute Sharpe            = ann_return / ann_vol
      5. Compute hit rate          = mean(r > 0)
      6. Compute max drawdown      = worst peak-to-trough within regime

    Note: regime days are NOT necessarily consecutive, so the max
    drawdown is computed on a "synthetic" NAV built from only those days.
    This is conservative — it underestimates max drawdown relative to
    actual lived experience (which includes multi-day streaks).

    Parameters
    ----------
    strategy_returns : daily net returns (pd.Series with DatetimeIndex)
    regime_labels    : regime classification for each date
    label            : strategy name for display

    Returns
    -------
    list[RegimeStats] — one per regime, in ORDERED display order
    """
    common = strategy_returns.index.intersection(regime_labels.index)
    ret    = strategy_returns.loc[common]
    reg    = regime_labels.loc[common]

    total_days = len(ret)
    stats_list = []

    for regime in RegimeLabels.ORDERED:
        mask = reg == regime
        r    = ret[mask]

        if len(r) < 10:
            log.warning(f"  [{label}] Regime '{regime}': only {len(r)} days — skipping")
            stats_list.append(RegimeStats(
                regime=regime, n_days=len(r), pct_days=len(r)/total_days,
                ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                hit_rate=np.nan, max_dd=np.nan,
            ))
            continue

        ann_ret = r.mean() * 252
        ann_vol = r.std(ddof=1) * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
        hit     = (r > 0).mean()

        # Regime max drawdown (on synthetic regime-only NAV)
        nav_r   = (1 + r).cumprod()
        dd      = float((nav_r / nav_r.cummax() - 1).min())

        stats_list.append(RegimeStats(
            regime=regime,
            n_days=int(len(r)),
            pct_days=float(len(r) / total_days),
            ann_return=float(ann_ret),
            ann_vol=float(ann_vol),
            sharpe=float(sharpe),
            hit_rate=float(hit),
            max_dd=float(dd),
        ))

        log.info(
            f"  [{label:20s}|{regime:18s}] "
            f"n={len(r):4d}  ret={ann_ret:.2%}  SR={sharpe:.3f}  "
            f"hit={hit:.1%}  MDD={dd:.2%}"
        )

    return stats_list


def regime_stats_to_df(stats_list: list[RegimeStats]) -> pd.DataFrame:
    """Convert list of RegimeStats to a tidy DataFrame."""
    return pd.DataFrame([s._asdict() for s in stats_list])


# ---------------------------------------------------------------------------
# Full regime analysis
# ---------------------------------------------------------------------------

def run_regime_analysis(
    strategy_ret_trad: pd.Series,
    strategy_ret_enh:  pd.Series,
    benchmark_ret:     pd.Series,
    market_returns:    pd.Series,
) -> dict:
    """
    Run complete regime analysis for both strategy variants and benchmark.

    Parameters
    ----------
    strategy_ret_trad : daily net returns, traditional strategy
    strategy_ret_enh  : daily net returns, enhanced strategy
    benchmark_ret     : daily returns, EW benchmark
    market_returns    : daily EW universe return (used for regime classification)

    Returns
    -------
    dict with:
        'regimes'            : pd.Series of regime labels
        'stats_traditional'  : list[RegimeStats]
        'stats_enhanced'     : list[RegimeStats]
        'stats_benchmark'    : list[RegimeStats]
        'regime_counts'      : pd.Series of regime day counts
    """
    log.info("Classifying market regimes …")
    regimes = classify_regimes(market_returns)

    log.info("Computing Traditional strategy regime stats …")
    stats_trad = compute_regime_stats(strategy_ret_trad, regimes, "Traditional")

    log.info("Computing Enhanced strategy regime stats …")
    stats_enh  = compute_regime_stats(strategy_ret_enh,  regimes, "Enhanced")

    log.info("Computing Benchmark regime stats …")
    stats_bm   = compute_regime_stats(benchmark_ret,     regimes, "Benchmark")

    return {
        "regimes":           regimes,
        "stats_traditional": stats_trad,
        "stats_enhanced":    stats_enh,
        "stats_benchmark":   stats_bm,
        "regime_counts":     regimes.value_counts(),
    }
