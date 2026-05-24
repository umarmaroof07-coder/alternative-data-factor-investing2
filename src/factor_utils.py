"""
factor_utils.py
---------------
Cross-sectional factor processing utilities.

Every function here operates on a single cross-section (one date, N stocks).
The factor_construction.py module applies these via DataFrame.apply() or
rolling windows to produce the full T × N factor panel.

Design philosophy
-----------------
Institutional factor research separates three concerns:
  1. Signal construction   — what raw economic quantity are we measuring?
  2. Cross-sectional processing — winsorise → rank → normalise
  3. Aggregation           — combine signals into a composite score

Keeping utilities here lets you swap normalisation schemes for the same
signal without touching the economic logic.

Author  : Quant Research Team
Phase   : 3 — Factor Construction
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Winsorisation
# ---------------------------------------------------------------------------

def winsorise(
    series: pd.Series,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.Series:
    """
    Clip extreme values to percentile bounds within a cross-section.

    WHY THIS MATTERS
    ----------------
    A single outlier stock can dominate a factor score.  Imagine computing
    12-month momentum: a biotech whose drug got FDA approval returned +900%.
    Without winsorisation that one stock pulls the entire cross-section mean
    upward, compressing every other stock's z-score toward zero.

    We clip at the 5th / 95th percentile by default.  This is more aggressive
    than the 1/99 used in Phase 2 (which handled price data).  Factor scores
    are derived quantities that can have fatter tails than raw prices, so
    tighter winsorisation is standard in Barra / Axioma style factor models.

    Parameters
    ----------
    series : pd.Series — one row of the factor matrix (one cross-section date)
    lower  : float     — lower percentile to clip to (default 5th)
    upper  : float     — upper percentile to clip to (default 95th)

    Returns
    -------
    pd.Series with the same index, values clipped to [lower_bound, upper_bound]
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


# ---------------------------------------------------------------------------
# 2. Cross-sectional ranking → uniform [0, 1]
# ---------------------------------------------------------------------------

def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """
    Convert raw factor values to uniform percentile ranks within a date.

    WHY RANK INSTEAD OF RAW VALUES?
    --------------------------------
    Raw momentum values depend on the market regime: in a bull market every
    stock has positive momentum, so the raw signal conveys less information
    about *relative* attractiveness.  Ranks remove this level effect.

    Ranks are scaled to [0, 1] by dividing by (N + 1).  Dividing by (N + 1)
    rather than N ensures no stock ever receives rank 0 or 1 exactly, which
    would cause numerical problems in subsequent transformations.

    This is the "percentile rank" used by AQR, Two Sigma, and most major
    systematic equity houses.

    Parameters
    ----------
    series : pd.Series — one cross-section of raw factor values

    Returns
    -------
    pd.Series — percentile ranks in (0, 1), NaN preserved
    """
    n = series.count()
    if n < 2:
        return pd.Series(np.nan, index=series.index)
    return series.rank(method="average", na_option="keep") / (n + 1)


# ---------------------------------------------------------------------------
# 3. Z-score normalisation
# ---------------------------------------------------------------------------

def cross_sectional_zscore(
    series: pd.Series,
    winsorise_first: bool = True,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.Series:
    """
    Standardise a cross-section to zero mean, unit standard deviation.

    Pipeline: [optional winsorise] → subtract mean → divide by std

    WHY Z-SCORE AFTER RANKING?
    --------------------------
    Ranks give us a uniform distribution.  The z-score of a uniform
    distribution is approximately normal, which is what most portfolio
    optimisers and risk models assume.  Running z-score on raw (unranked)
    factor values often yields a skewed or bimodal distribution.

    The two-step rank → z-score is sometimes called "normal-score transform"
    or "inverse normal transform" and is the industry standard for feeding
    signals into mean-variance optimisers.

    Parameters
    ----------
    series          : cross-section of factor values (raw or ranked)
    winsorise_first : whether to clip before standardising
    lower / upper   : winsorisation percentiles

    Returns
    -------
    pd.Series — z-scores with mean ≈ 0, std ≈ 1
    """
    if winsorise_first:
        series = winsorise(series, lower, upper)

    mu  = series.mean()
    sig = series.std(ddof=1)

    if sig < 1e-10:          # degenerate cross-section (e.g. all same value)
        return pd.Series(0.0, index=series.index)

    return (series - mu) / sig


# ---------------------------------------------------------------------------
# 4. Full normalisation pipeline (rank → z-score)
# ---------------------------------------------------------------------------

def normalise_factor(
    raw: pd.Series,
    winsorise_pcts: tuple[float, float] = (0.05, 0.95),
) -> pd.Series:
    """
    Standard two-step institutional factor normalisation.

    Step 1 — Winsorise at 5th / 95th percentile
    Step 2 — Cross-sectional percentile rank → (0, 1)
    Step 3 — Z-score the ranks → approximately N(0, 1)

    This three-step pipeline is used by Barra, Axioma, and most major
    systematic factor providers.  The key property: the output is always
    approximately standard normal, regardless of the raw signal's distribution.

    Parameters
    ----------
    raw            : pd.Series — raw cross-sectional factor values
    winsorise_pcts : (lower, upper) percentiles for winsorisation

    Returns
    -------
    pd.Series — normalised factor scores, approximately N(0, 1)
    """
    lo, hi = winsorise_pcts
    step1 = winsorise(raw, lo, hi)
    step2 = cross_sectional_rank(step1)
    step3 = cross_sectional_zscore(step2, winsorise_first=False)
    return step3


# ---------------------------------------------------------------------------
# 5. Composite score combination
# ---------------------------------------------------------------------------

def combine_factors(
    factor_dict: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Combine multiple normalised factor scores into a single composite score.

    Combination method: weighted average of z-scores.

    WHY EQUAL-WEIGHT AS DEFAULT?
    ----------------------------
    Optimising factor weights in-sample almost always leads to over-fitting.
    Equal-weighting is a robust baseline that outperforms in-sample optimised
    weights out-of-sample in most academic and practitioner studies.

    If weights are provided, they are normalised to sum to 1.0 so the caller
    doesn't need to worry about scaling.  The final composite is re-z-scored
    so it remains approximately N(0, 1) regardless of weight scheme.

    Parameters
    ----------
    factor_dict : {factor_name: normalised_scores (pd.Series)}
    weights     : {factor_name: weight}  — None = equal weight

    Returns
    -------
    pd.Series — composite z-score
    """
    names = list(factor_dict.keys())

    if weights is None:
        weights = {n: 1.0 for n in names}

    # Normalise weights to sum = 1
    total_w = sum(weights[n] for n in names)
    w = {n: weights[n] / total_w for n in names}

    # Stack into a matrix for vectorised dot product
    frame = pd.DataFrame(factor_dict)
    w_vec = pd.Series(w)

    composite = frame.mul(w_vec).sum(axis=1, min_count=1)

    # Re-normalise composite so it stays ≈ N(0,1)
    composite = cross_sectional_zscore(composite, winsorise_first=False)
    return composite


# ---------------------------------------------------------------------------
# 6. Diagnostic helpers
# ---------------------------------------------------------------------------

def factor_coverage(factor_df: pd.DataFrame) -> pd.Series:
    """Fraction of universe with valid (non-NaN) scores on each date."""
    return factor_df.notna().mean(axis=1)


def factor_autocorrelation(factor_df: pd.DataFrame, lag: int = 1) -> float:
    """
    Average cross-sectional rank autocorrelation across all dates.

    A factor with high autocorrelation turns over slowly → lower transaction
    costs.  Momentum typically has AC > 0.95 at a 1-day lag.
    Volatility-based factors are even stickier (AC > 0.98).
    """
    acs = []
    dates = factor_df.index
    for i in range(lag, len(dates)):
        s_t   = factor_df.iloc[i]
        s_lag = factor_df.iloc[i - lag]
        valid = s_t.notna() & s_lag.notna()
        if valid.sum() < 10:
            continue
        ac, _ = stats.spearmanr(s_t[valid], s_lag[valid])
        acs.append(ac)
    return float(np.mean(acs)) if acs else np.nan


def factor_dispersion(factor_df: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional standard deviation of factor scores by date.

    Low dispersion means stocks are bunched together → weaker signal.
    After normalisation this should be ≈ 1.0 by construction.
    Significant deviation flags a data problem.
    """
    return factor_df.std(axis=1, ddof=1)
