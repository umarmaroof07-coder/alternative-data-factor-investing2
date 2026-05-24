"""
backtest.py
-----------
Phase 4b — Backtest Engine

Simulates the day-by-day NAV of a portfolio defined by portfolio.py.

Responsibilities
----------------
1. Apply the weight schedule to daily stock returns → gross daily P&L
2. Subtract transaction cost drag → net daily P&L
3. Compound daily returns into a NAV curve starting at 100
4. Build a benchmark NAV from equal-weighted universe returns
5. Run the same engine on a rolling basis for out-of-sample attribution

Key design decisions
--------------------
• We work with *returns*, not prices.  Chaining returns via (1+r).cumprod()
  is numerically stable and avoids currency effects.
• Weights are shifted forward by 1 day (weights[t] earns returns[t+1]).
  This is the single most important look-ahead bias control.
• The benchmark is the equal-weighted S&P 100 universe, giving a fair
  comparison that controls for the large-cap equity risk premium.

Author  : Quant Research Team
Phase   : 4 — Portfolio Construction & Backtesting
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core engine — vectorised NAV simulation
# ---------------------------------------------------------------------------

def simulate_nav(
    weights:    pd.DataFrame,
    returns:    pd.DataFrame,
    cost_drag:  pd.Series,
    initial_nav: float = 100.0,
) -> pd.DataFrame:
    """
    Simulate daily portfolio NAV from weights and stock returns.

    Maths
    -----
    gross_return(t) = Σ_i  w_i(t-1) × r_i(t)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      weights from *yesterday* applied to *today's* returns
                      → the core look-ahead-bias control

    net_return(t)   = gross_return(t) + cost_drag(t)
                      (cost_drag is negative on rebalance days, zero otherwise)

    NAV(t)          = NAV(t-1) × (1 + net_return(t))
    NAV(0)          = initial_nav   (e.g. 100.0 = $100 starting value)

    WHY SHIFT WEIGHTS BY 1 DAY?
    ---------------------------
    On rebalance day t we *decide* the new portfolio using factor scores
    through date t.  We cannot execute those trades and also earn today's
    return on the new portfolio — we execute at today's close and begin
    earning the new weights starting tomorrow.

    Failing to shift (using weights[t] with returns[t]) is a very common
    and hard-to-detect look-ahead bias.  It typically adds 20–50bps of
    spurious alpha per year in monthly-rebalanced strategies.

    Parameters
    ----------
    weights     : T × N weight DataFrame (from portfolio.py, not yet shifted)
    returns     : T × N simple daily return DataFrame
    cost_drag   : pd.Series of daily cost drag (0 on non-rebalance days)
    initial_nav : starting portfolio value

    Returns
    -------
    pd.DataFrame with columns:
        gross_return : daily portfolio return before costs
        cost_drag    : daily cost subtracted
        net_return   : gross - cost
        nav_gross    : cumulative NAV (no costs)
        nav_net      : cumulative NAV (after costs)
    """
    # Align on common dates (intersection of weights and returns)
    common = weights.index.intersection(returns.index)
    w = weights.loc[common]
    r = returns.loc[common]
    c = cost_drag.reindex(common).fillna(0.0)

    # *** THE KEY LOOK-AHEAD BIAS CONTROL ***
    # Shift weights forward by 1 trading day.
    # w.shift(1) at row t gives weights decided at t-1.
    # These weights earn the return at t.
    w_lagged = w.shift(1).fillna(0.0)

    # Gross daily portfolio return (vectorised dot product, row by row)
    gross_returns = (w_lagged * r).sum(axis=1)

    # Net daily return
    net_returns = gross_returns + c   # c is already negative

    # Compound into NAV curves
    nav_gross = initial_nav * (1 + gross_returns).cumprod()
    nav_net   = initial_nav * (1 + net_returns).cumprod()

    result = pd.DataFrame({
        "gross_return": gross_returns,
        "cost_drag":    c,
        "net_return":   net_returns,
        "nav_gross":    nav_gross,
        "nav_net":      nav_net,
    }, index=common)

    result.index.name = "date"

    log.info(
        f"NAV simulated | {len(result)} days | "
        f"final NAV gross={nav_gross.iloc[-1]:.2f} net={nav_net.iloc[-1]:.2f}"
    )
    return result


# ---------------------------------------------------------------------------
# Benchmark: equal-weighted universe
# ---------------------------------------------------------------------------

def build_benchmark(
    returns:     pd.DataFrame,
    name:        str = "EW Universe",
) -> pd.DataFrame:
    """
    Build an equal-weighted benchmark from the full stock universe.

    WHY EQUAL-WEIGHT BENCHMARK?
    ---------------------------
    The standard benchmark for a US large-cap strategy is SPY (S&P 500).
    Since we are working with synthetic data (not live Yahoo Finance data),
    we cannot download SPY.  Instead, we use the equal-weighted S&P 100
    universe as a benchmark.

    This is actually a *stricter* benchmark than SPY for our strategy because:
    1. It controls for the same universe (no style drift artifacts)
    2. Equal-weight benchmarks tend to have a small-cap and value tilt
       vs cap-weighted SPY, making it harder to look good on momentum alone
    3. It is purely data-driven — no vendor dependency

    In a live production system, replace this with your actual benchmark
    (SPY, Russell 1000, MSCI USA, etc.).

    Parameters
    ----------
    returns : T × N daily return DataFrame (full universe)
    name    : label for the benchmark in output tables

    Returns
    -------
    pd.DataFrame with columns:
        bm_return : daily benchmark return
        bm_nav    : cumulative benchmark NAV starting at 100
    """
    bm_return = returns.mean(axis=1)   # equal-weight = simple cross-sectional mean
    bm_nav    = 100.0 * (1 + bm_return).cumprod()

    result = pd.DataFrame({
        "bm_return": bm_return,
        "bm_nav":    bm_nav,
    })
    result.index.name = "date"

    log.info(
        f"Benchmark built ({name}) | "
        f"final NAV: {bm_nav.iloc[-1]:.2f}"
    )
    return result


# ---------------------------------------------------------------------------
# Rolling 12-month attribution windows
# ---------------------------------------------------------------------------

def rolling_annual_returns(
    nav: pd.Series,
    window_days: int = 252,
) -> pd.Series:
    """
    Rolling 12-month (252-day) annualised return of the NAV curve.

    Used for calendar-year analysis and "consistency of alpha" charts.

    A strategy with a good mean Sharpe but wildly variable annual returns
    would be rejected by most institutional allocators (career risk).
    Rolling annual return is the key metric for assessing consistency.

    Parameters
    ----------
    nav         : cumulative NAV series (starts at 100)
    window_days : look-back window (default 252 = 1 trading year)

    Returns
    -------
    pd.Series — annualised return for each rolling window
    """
    log_nav    = np.log(nav)
    rolling_lr = log_nav.diff(window_days)          # log return over window
    ann_return = np.exp(rolling_lr) - 1             # convert back to simple
    return ann_return


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    portfolio:   dict,
    returns:     pd.DataFrame,
    initial_nav: float = 100.0,
) -> dict:
    """
    Run the complete backtest and return all results in one bundle.

    Parameters
    ----------
    portfolio   : output dict from portfolio.build_portfolio()
    returns     : T × N simple daily return DataFrame (from Phase 2)
    initial_nav : starting portfolio value (default 100)

    Returns
    -------
    dict with keys:
        nav_df      : daily NAV DataFrame (gross + net + cost columns)
        benchmark   : daily benchmark NAV DataFrame
        rolling_ret : rolling 12-month annualised returns (net)
        portfolio   : the portfolio dict (pass-through for convenience)
    """
    log.info("Running backtest …")

    # Daily NAV simulation
    nav_df = simulate_nav(
        weights=portfolio["weights"],
        returns=returns,
        cost_drag=portfolio["cost_drag"],
        initial_nav=initial_nav,
    )

    # Benchmark
    benchmark = build_benchmark(returns)

    # Align benchmark to same dates as strategy
    common = nav_df.index.intersection(benchmark.index)
    benchmark = benchmark.loc[common]

    # Rolling annual returns (net)
    rolling_ret = rolling_annual_returns(nav_df["nav_net"])

    log.info("Backtest complete.")

    return {
        "nav_df":      nav_df,
        "benchmark":   benchmark,
        "rolling_ret": rolling_ret,
        "portfolio":   portfolio,
    }
