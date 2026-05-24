"""
portfolio.py
------------
Phase 4a — Portfolio Construction

Translates factor scores into investable portfolio weights.

Responsibilities
----------------
1. Identify monthly rebalance dates from the trading calendar
2. Select the top-quintile stocks by composite z-score on each date
3. Apply equal-weight scheme
4. Enforce a minimum-stock floor (diversification guard)
5. Compute position changes (turnover) between rebalances
6. Apply one-way transaction cost drag to each rebalance

Design philosophy
-----------------
Portfolio construction is deliberately separated from backtesting.
`portfolio.py` answers "what do we hold and when?"
`backtest.py`  answers "what did that earn day-by-day?"
`performance.py` answers "how good was that?"

This mirrors the three-layer architecture used at quant funds:
  Alpha Research → Portfolio Construction → Execution / Risk

Look-ahead bias controls
------------------------
Every weight decision uses only information available *before* the period
it applies to.  Specifically:

  • Factor scores are computed from prices up to date t.
  • The portfolio formed at date t is applied starting on t+1.
  • The one-day lag is enforced in backtest.py via index shifting.

Author  : Quant Research Team
Phase   : 4 — Portfolio Construction & Backtesting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    """
    All tunable parameters in one place.

    Putting config in a dataclass (rather than module-level constants) lets
    you run sensitivity analyses cleanly:
        cfg_high_cost = PortfolioConfig(transaction_cost_bps=20)
        cfg_low_cost  = PortfolioConfig(transaction_cost_bps=5)
    """

    # Stock selection
    top_quantile: float = 0.20          # Top 20% by composite score
    min_stocks:   int   = 10            # Never hold fewer than N stocks

    # Rebalance schedule
    rebalance_freq: str = "BME"         # Business-Month-End (last trading day of month)

    # Transaction costs
    # 10 bps (0.10%) one-way is realistic for liquid large-cap US equities.
    # This covers bid-ask spread + market impact for ~$10M positions.
    # Retail brokers may charge 0; HF algos model impact curves instead.
    transaction_cost_bps: float = 10.0

    # Weighting scheme
    weighting: str = "equal"            # "equal" | future: "value", "min_var"

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))


# ---------------------------------------------------------------------------
# Step 1 — Build rebalance calendar
# ---------------------------------------------------------------------------

def get_rebalance_dates(
    factor_scores: pd.DataFrame,
    freq: str = "BME",
) -> pd.DatetimeIndex:
    """
    Identify the last trading day of each month in the factor panel.

    WHY MONTH-END REBALANCING?
    --------------------------
    Monthly is the standard for systematic equity strategies because:
    • It gives the momentum signal enough time to work (momentum reverts
      at horizons shorter than ~1 week).
    • It keeps turnover and transaction costs manageable.
    • It aligns with institutional reporting cycles.

    We use Business-Month-End (BME) to land on the actual last *trading*
    day of each month, not a Saturday or holiday.

    WHY NOT DAILY?
    --------------
    Daily rebalancing would cost ~252 × 10bps ≈ 25% per year in transaction
    costs alone — wiping out the entire factor alpha.

    Parameters
    ----------
    factor_scores : T × N DataFrame with DatetimeIndex
    freq          : pandas offset alias (default "BME" = business month end)

    Returns
    -------
    pd.DatetimeIndex of rebalance dates, filtered to dates with valid scores
    """
    # Resample to get the last trading day of each calendar month
    monthly = factor_scores.resample(freq).last()

    # Keep only months where we actually have factor scores for ≥ 10 stocks
    has_scores = monthly.notna().sum(axis=1) >= 10
    valid = monthly[has_scores].index

    log.info(f"Rebalance calendar: {len(valid)} dates | {valid[0].date()} → {valid[-1].date()}")
    return valid


# ---------------------------------------------------------------------------
# Step 2 — Select stocks for each rebalance date
# ---------------------------------------------------------------------------

def select_stocks(
    scores: pd.Series,
    top_quantile: float = 0.20,
    min_stocks:   int   = 10,
) -> list[str]:
    """
    Select the top-scoring stocks from one cross-section.

    WHY TOP QUINTILE (TOP 20%)?
    ---------------------------
    Academic momentum and low-vol studies typically form decile or quintile
    portfolios.  The top quintile balances:
    • Concentration of signal (top 5% is too small for diversification)
    • Sufficient diversification (we want ≥ 20 stocks to avoid name risk)

    With 105 S&P 100 stocks, top 20% ≈ 21 stocks — a well-diversified
    concentrated portfolio by institutional standards.

    Parameters
    ----------
    scores       : pd.Series of composite z-scores for one date
    top_quantile : fraction of universe to select (0.20 = top quintile)
    min_stocks   : minimum portfolio size regardless of quantile

    Returns
    -------
    list of ticker strings, sorted by score descending
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return []

    n_select = max(min_stocks, int(np.ceil(len(valid) * top_quantile)))
    n_select = min(n_select, len(valid))   # can't select more than universe

    selected = valid.nlargest(n_select).index.tolist()
    return selected


# ---------------------------------------------------------------------------
# Step 3 — Compute weights
# ---------------------------------------------------------------------------

def compute_weights(
    selected_tickers: list[str],
    all_tickers:      list[str],
    scheme:           str = "equal",
) -> pd.Series:
    """
    Translate a list of selected tickers into a full weight vector.

    The weight vector covers the *entire* universe (not just selected stocks)
    so that arithmetic in backtest.py is simply:
        portfolio_return = (weights × stock_returns).sum(axis=1)

    WHY EQUAL WEIGHT?
    -----------------
    Equal weighting is the max-entropy portfolio: given a set of stocks
    we believe are attractive, we assign equal confidence to each.

    Mean-variance optimisation (Markowitz) sounds better in theory but
    requires estimating a covariance matrix — which introduces estimation
    error that typically *increases* out-of-sample variance relative to
    equal weighting.  DeMiguel et al. (2007) showed that 1/N beats
    optimised portfolios across 14 datasets.

    For institutional portfolios, equal weight also avoids over-concentration
    in high-volatility names that tend to get large mean-variance weights.

    Parameters
    ----------
    selected_tickers : tickers to hold (non-zero weight)
    all_tickers      : full universe of tickers (for the weight vector)
    scheme           : "equal" (others reserved for future phases)

    Returns
    -------
    pd.Series indexed by all_tickers; selected = 1/N, rest = 0
    """
    weights = pd.Series(0.0, index=all_tickers)

    if not selected_tickers:
        return weights

    if scheme == "equal":
        w = 1.0 / len(selected_tickers)
        weights[selected_tickers] = w
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    return weights


# ---------------------------------------------------------------------------
# Step 4 — Build full weight schedule
# ---------------------------------------------------------------------------

def build_weight_schedule(
    factor_scores: pd.DataFrame,
    config:        PortfolioConfig,
) -> pd.DataFrame:
    """
    Construct the T × N weight matrix covering the full backtest period.

    The weight on date t is the portfolio *held* on date t.
    Weights are rebalanced at month-end; between rebalances, weights are
    held constant (we assume no intra-month drifting for simplicity —
    Phase 5 can add drift-adjusted weights).

    LOOK-AHEAD BIAS NOTE
    --------------------
    Weights on date t are computed from factor_scores on date t, which
    uses prices up to and including date t.  The *portfolio return*
    earned using those weights begins on date t+1.  The shift is enforced
    in backtest.py by using `.shift(1)` on the weight matrix.

    Parameters
    ----------
    factor_scores : T × N normalised composite z-score DataFrame
    config        : PortfolioConfig

    Returns
    -------
    weights_df : T × N DataFrame; weights sum to 1 on rebalance dates,
                 forward-filled between them
    holdings_df: T × list mapping each date → selected tickers
    """
    log.info("Building weight schedule …")

    rebal_dates = get_rebalance_dates(factor_scores, freq=config.rebalance_freq)
    all_tickers = factor_scores.columns.tolist()

    # Snap BME dates that fall after the last available factor date
    factor_index = factor_scores.index
    snapped = []
    for d in rebal_dates:
        if d in factor_index:
            snapped.append(d)
        else:
            prior = factor_index[factor_index <= d]
            if len(prior) > 0:
                snapped.append(prior[-1])
    rebal_dates = pd.DatetimeIndex(sorted(set(snapped)))

    rebal_weights: dict[pd.Timestamp, pd.Series] = {}
    rebal_holdings: dict[pd.Timestamp, list[str]] = {}

    for date in rebal_dates:
        scores   = factor_scores.loc[date]
        selected = select_stocks(scores, config.top_quantile, config.min_stocks)
        weights  = compute_weights(selected, all_tickers, config.weighting)

        rebal_weights[date]   = weights
        rebal_holdings[date]  = selected

    # DataFrame of weights on rebalance dates, NaN elsewhere
    weights_df = pd.DataFrame(rebal_weights).T
    weights_df.index.name = "date"

    # Forward-fill weights between rebalances (hold positions constant)
    # Reindex to full factor_scores date range, then ffill
    weights_df = (
        weights_df
        .reindex(factor_scores.index)
        .ffill()
        .fillna(0.0)          # before first rebalance, hold nothing (cash)
    )

    log.info(
        f"Weight schedule built | {len(rebal_dates)} rebalances | "
        f"avg holdings: {sum(len(v) for v in rebal_holdings.values()) / len(rebal_holdings):.1f} stocks"
    )
    return weights_df, rebal_holdings


# ---------------------------------------------------------------------------
# Step 5 — Turnover & transaction costs
# ---------------------------------------------------------------------------

def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """
    Compute one-way portfolio turnover on each rebalance date.

    Definition
    ----------
    Turnover = 0.5 × Σ |w_new(i) - w_old(i)|

    The 0.5 factor converts two-way (buy + sell) to one-way turnover,
    which is the convention for cost budgeting.  A turnover of 1.0 means
    100% of the portfolio was replaced; 0.30 means 30% was traded.

    WHY DOES TURNOVER MATTER?
    -------------------------
    At 10bps one-way cost, turnover of 30% costs 30bps per rebalance.
    At 12 rebalances per year that's 360bps = 3.6% annual drag — which
    can more than wipe out factor alpha.  Tracking turnover is essential
    for understanding the *net* (after-cost) Sharpe ratio.

    Parameters
    ----------
    weights_df : T × N weight matrix (full date range, ffilled)

    Returns
    -------
    pd.Series — one-way turnover on each day (0 on non-rebalance days)
    """
    # Day-over-day weight change
    delta = weights_df.diff().abs().sum(axis=1) * 0.5
    delta.iloc[0] = weights_df.iloc[0].sum() * 0.5   # initial buy-in
    return delta


def compute_cost_drag(
    turnover: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Daily P&L drag from transaction costs.

    cost_drag(t) = -turnover(t) × cost_bps / 10_000

    This is subtracted from gross daily returns in the backtest.

    Parameters
    ----------
    turnover : one-way turnover series
    cost_bps : one-way cost in basis points (1 bp = 0.01%)

    Returns
    -------
    pd.Series — negative daily return drag (e.g. -0.0010 = -10bps)
    """
    return -(turnover * cost_bps / 10_000)


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def build_portfolio(
    factor_scores: pd.DataFrame,
    config:        PortfolioConfig | None = None,
) -> dict:
    """
    Run the full portfolio construction pipeline.

    Returns
    -------
    dict with:
        weights      : T × N weight DataFrame (ffilled, pre-shift)
        holdings     : dict[date → list[ticker]]
        turnover     : pd.Series of one-way daily turnover
        cost_drag    : pd.Series of daily cost drag (negative returns)
        rebal_dates  : pd.DatetimeIndex
        config       : PortfolioConfig used
    """
    if config is None:
        config = PortfolioConfig()

    weights, holdings = build_weight_schedule(factor_scores, config)
    turnover  = compute_turnover(weights)
    cost_drag = compute_cost_drag(turnover, config.transaction_cost_bps)

    rebal_dates = get_rebalance_dates(factor_scores, config.rebalance_freq)

    log.info(
        f"Portfolio ready | avg monthly turnover: "
        f"{turnover[turnover > 0].mean():.1%} | "
        f"est. annual cost drag: {cost_drag.sum() / (len(cost_drag) / 252):.2%}"
    )

    return {
        "weights":     weights,
        "holdings":    holdings,
        "turnover":    turnover,
        "cost_drag":   cost_drag,
        "rebal_dates": rebal_dates,
        "config":      config,
    }
