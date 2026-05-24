"""
robustness.py
-------------
Phase 6a — Robustness Testing Engine

PURPOSE
-------
A backtest result that cannot survive reasonable perturbations is not
evidence of a genuine strategy — it is evidence of over-fitting.

This module runs five classes of perturbation tests:

  Test 1 — Portfolio Concentration Sensitivity
  Test 2 — Rebalance Frequency Sensitivity
  Test 3 — Transaction Cost Stress Test
  Test 4 — Factor Weight Robustness
  Test 5 — Factor Ablation Study

WHY ROBUSTNESS TESTING MATTERS
-------------------------------
Every backtest has degrees of freedom — design choices that were made
after seeing the data (even unconsciously):

  • "I chose top 20% because it looked good" → concentration sensitivity
  • "I chose monthly because it balanced cost/signal" → freq sensitivity
  • "10bps seemed realistic" → cost stress test
  • "40/30/30 felt balanced" → weight robustness
  • "I included hiring AND sentiment" → ablation study

A robust strategy shows MONOTONE or SMOOTH degradation as you perturb
each parameter away from the chosen value.  Sharp non-linearities
(e.g., 20% is great but 21% collapses) are red flags for overfitting.

INSTITUTIONAL STANDARD
-----------------------
For publication in academic journals or institutional due diligence:
  - "The Sharpe ratio degrades gracefully from 1.2 at top 20% to 0.9 at
     top 30%, confirming the signal is not concentrated in a handful of stocks"
  - "The strategy survives 50bps transaction costs with positive net Sharpe"
  - "Removing either alt-data signal individually reduces Sharpe by ~15%,
     confirming each contributes independent value"

LOOK-AHEAD BIAS NOTE
--------------------
All tests use pre-computed factor scores (from Phase 3 and 5).  Those
scores were built with strict look-ahead controls in earlier phases.
Robustness testing does not re-compute scores, so it cannot introduce
new look-ahead bias — it only re-runs portfolio construction and backtest.

Author  : Quant Research Team
Phase   : 6 — Robustness Testing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio   import PortfolioConfig, build_portfolio
from backtest    import run_backtest, build_benchmark
from performance import (
    cagr, annualised_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, information_ratio,
)
from factor_utils import cross_sectional_zscore

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Lightweight container for a single backtest's key outputs.

    Storing only scalars (not DataFrames) keeps the results dict small
    enough to serialise to JSON for paper tables.
    """
    label:        str
    param_name:   str
    param_value:  Any

    # Performance metrics
    cagr_pct:     float = 0.0
    ann_vol_pct:  float = 0.0
    sharpe:       float = 0.0
    sortino:      float = 0.0
    max_dd_pct:   float = 0.0
    calmar:       float = 0.0
    ir:           float = 0.0    # vs EW benchmark

    # Activity metrics
    avg_turnover_pct: float = 0.0
    n_rebalances:     int   = 0
    ann_cost_drag_pct: float = 0.0

    # For time-series comparison charts
    nav_series:   pd.Series = field(default_factory=pd.Series)
    ret_series:   pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> dict:
        """Serialisable dict (excludes Series)."""
        return {
            "label":             self.label,
            "param_name":        self.param_name,
            "param_value":       str(self.param_value),
            "cagr_pct":          round(self.cagr_pct, 4),
            "ann_vol_pct":       round(self.ann_vol_pct, 4),
            "sharpe":            round(self.sharpe, 4),
            "sortino":           round(self.sortino, 4),
            "max_dd_pct":        round(self.max_dd_pct, 4),
            "calmar":            round(self.calmar, 4),
            "ir":                round(self.ir, 4),
            "avg_turnover_pct":  round(self.avg_turnover_pct, 4),
            "n_rebalances":      self.n_rebalances,
            "ann_cost_drag_pct": round(self.ann_cost_drag_pct, 4),
        }


# ---------------------------------------------------------------------------
# Core helper: run one backtest and extract all metrics
# ---------------------------------------------------------------------------

def _run_one(
    composite:  pd.DataFrame,
    returns:    pd.DataFrame,
    config:     PortfolioConfig,
    label:      str,
    param_name: str,
    param_val:  Any,
    benchmark_ret: pd.Series | None = None,
) -> BacktestResult:
    """
    Execute one (composite → portfolio → backtest → metrics) cycle.

    This helper is called by every test function below.  Centralising
    the boilerplate here ensures all tests use *identical* metric
    definitions, preventing subtle inconsistencies in comparison tables.
    """
    t0   = time.time()
    port = build_portfolio(composite, config)
    res  = run_backtest(port, returns)

    nav_net = res["nav_df"]["nav_net"]
    ret_net = res["nav_df"]["net_return"]

    # Turnover stats
    to_events = port["turnover"][port["turnover"] > 0.001]
    avg_to    = float(to_events.mean()) if len(to_events) else 0.0

    # Annual cost drag
    cost_drag_total = -res["nav_df"]["cost_drag"].sum()
    n_years         = len(ret_net) / 252
    ann_cost        = cost_drag_total / n_years if n_years > 0 else 0.0

    # Information ratio vs benchmark
    ir_val = np.nan
    if benchmark_ret is not None:
        common = ret_net.index.intersection(benchmark_ret.index)
        if len(common) > 20:
            ir_val = information_ratio(ret_net.loc[common], benchmark_ret.loc[common])

    result = BacktestResult(
        label=label,
        param_name=param_name,
        param_value=param_val,
        cagr_pct=cagr(nav_net),
        ann_vol_pct=annualised_volatility(ret_net),
        sharpe=sharpe_ratio(ret_net),
        sortino=sortino_ratio(ret_net),
        max_dd_pct=max_drawdown(nav_net),
        calmar=calmar_ratio(nav_net, ret_net),
        ir=float(ir_val) if not np.isnan(ir_val) else 0.0,
        avg_turnover_pct=avg_to,
        n_rebalances=int(len(to_events)),
        ann_cost_drag_pct=ann_cost,
        nav_series=nav_net,
        ret_series=ret_net,
    )

    elapsed = time.time() - t0
    log.info(
        f"  [{label:35s}] CAGR={result.cagr_pct:.2%}  "
        f"SR={result.sharpe:.3f}  MDD={result.max_dd_pct:.2%}  "
        f"TO={result.avg_turnover_pct:.1%}  [{elapsed:.2f}s]"
    )
    return result


# ===========================================================================
# Test 1 — Portfolio Concentration Sensitivity
# ===========================================================================

def test_concentration(
    composite:     pd.DataFrame,
    returns:       pd.DataFrame,
    quantiles:     list[float] | None = None,
    base_config:   PortfolioConfig | None = None,
    benchmark_ret: pd.Series | None = None,
) -> list[BacktestResult]:
    """
    Vary the top-N% selection threshold and observe performance.

    WHY THIS TEST?
    --------------
    If the strategy only works when exactly the top 20% is selected,
    that is strong evidence of in-sample over-fitting.  A genuine
    signal should show:
      - Monotone degradation from top 10% → top 30% (more dilution)
      - No cliff-edge non-linearities

    WHAT TO LOOK FOR
    ----------------
    Healthy pattern: Sharpe(10%) > Sharpe(20%) > Sharpe(30%)
    Red flag      : Sharpe(10%)  ≈ 0  but Sharpe(20%) >> 0
                    (factor has no alpha at the top but looks good at 20%
                    → the "good" stocks are actually the 11th-20th percentile,
                    not the top, which makes no economic sense)

    Note: top 10% ≈ 10–11 stocks.  Below 10 stocks, idiosyncratic risk
    dominates and Sharpe degrades for mechanical reasons (not signal
    quality) — we note this in the output.

    Parameters
    ----------
    quantiles   : fractions to test (default: [0.10, 0.20, 0.30])
    base_config : PortfolioConfig to clone with modified quantile
    """
    if quantiles is None:
        quantiles = [0.10, 0.20, 0.30]
    if base_config is None:
        base_config = PortfolioConfig()

    log.info("TEST 1: Portfolio Concentration Sensitivity")
    results = []

    for q in quantiles:
        n_stocks_approx = int(composite.shape[1] * q)
        label = f"Top {q:.0%} (~{n_stocks_approx} stocks)"
        cfg   = PortfolioConfig(
            top_quantile=q,
            transaction_cost_bps=base_config.transaction_cost_bps,
            rebalance_freq=base_config.rebalance_freq,
            weighting=base_config.weighting,
        )
        results.append(_run_one(composite, returns, cfg, label,
                                "top_quantile", q, benchmark_ret))

    return results


# ===========================================================================
# Test 2 — Rebalance Frequency Sensitivity
# ===========================================================================

def test_rebalance_frequency(
    composite:     pd.DataFrame,
    returns:       pd.DataFrame,
    frequencies:   list[tuple[str, str]] | None = None,
    base_config:   PortfolioConfig | None = None,
    benchmark_ret: pd.Series | None = None,
) -> list[BacktestResult]:
    """
    Vary the rebalancing cadence from weekly to quarterly.

    WHY THIS TEST?
    --------------
    Rebalance frequency creates a tension between:
      • Signal freshness (rebalance often → capture signal faster)
      • Transaction costs (rebalance less → pay less)

    A robust signal should show:
      - Quarterly: lower costs offset somewhat slower signal decay
      - Monthly: the sweet spot (our base case)
      - Weekly: higher costs erode alpha faster than the signal refreshes

    For a momentum factor (half-life ~3 months), monthly rebalancing
    is near-optimal in theory.  If weekly outperforms monthly *before*
    costs but underperforms after, that confirms the cost model is
    correctly calibrated.

    TURNOVER SCALING LAW
    --------------------
    Expected turnover scales roughly with rebalance frequency:
      Weekly   ≈ monthly_TO × sqrt(4)   [in practice ~10–15% per week]
      Monthly  ≈ baseline ~20–30%
      Quarterly ≈ monthly_TO × sqrt(1/3) [in practice ~35–50% per quarter]

    But the ANNUAL turnover is similar across frequencies because you
    rebalance more often but by smaller amounts.  This makes the annual
    cost comparable, but performance still differs due to signal timing.

    Parameters
    ----------
    frequencies : list of (pandas_offset_alias, human_label) pairs
    """
    if frequencies is None:
        frequencies = [
            ("W-FRI", "Weekly"),
            ("BME",   "Monthly"),
            ("BQE",   "Quarterly"),
        ]
    if base_config is None:
        base_config = PortfolioConfig()

    log.info("TEST 2: Rebalance Frequency Sensitivity")
    results = []

    for freq, label in frequencies:
        cfg = PortfolioConfig(
            top_quantile=base_config.top_quantile,
            transaction_cost_bps=base_config.transaction_cost_bps,
            rebalance_freq=freq,
            weighting=base_config.weighting,
        )
        results.append(_run_one(composite, returns, cfg, label,
                                "rebalance_freq", freq, benchmark_ret))

    return results


# ===========================================================================
# Test 3 — Transaction Cost Stress Test
# ===========================================================================

def test_transaction_costs(
    composite:     pd.DataFrame,
    returns:       pd.DataFrame,
    cost_bps_list: list[float] | None = None,
    base_config:   PortfolioConfig | None = None,
    benchmark_ret: pd.Series | None = None,
) -> list[BacktestResult]:
    """
    Stress test alpha survival under increasing transaction costs.

    WHY THIS TEST?
    --------------
    Transaction costs are one of the most frequently abused assumptions
    in academic backtests.  Many published "anomalies" disappear once
    realistic costs are applied (Novy-Marx & Velikov 2016 documented this
    for over 100 published factors).

    The institutional standard is to find the "cost breakeven" — the
    transaction cost level at which net Sharpe falls to zero.  A strategy
    with:
      • Breakeven < 5bps   → only viable at the largest HFT scale
      • Breakeven 10–20bps → viable for mid-size institutional funds
      • Breakeven > 50bps  → viable even for retail investors

    REALISTIC COST RANGES (one-way)
    --------------------------------
    0 bps   : Theoretical (zero-cost) — measures gross alpha
    5 bps   : Large institutional fund, passive execution
    10 bps  : Standard institutional (our base case)
    20 bps  : Smaller fund, some market impact
    50 bps  : Retail investor, wide spreads, market impact

    The crossover point (positive → negative Sharpe) is the key output.

    Parameters
    ----------
    cost_bps_list : one-way costs to test in basis points
    """
    if cost_bps_list is None:
        cost_bps_list = [0, 5, 10, 20, 50]
    if base_config is None:
        base_config = PortfolioConfig()

    log.info("TEST 3: Transaction Cost Stress Test")
    results = []

    for bps in cost_bps_list:
        label = f"{bps} bps one-way"
        cfg   = PortfolioConfig(
            top_quantile=base_config.top_quantile,
            transaction_cost_bps=float(bps),
            rebalance_freq=base_config.rebalance_freq,
            weighting=base_config.weighting,
        )
        results.append(_run_one(composite, returns, cfg, label,
                                "cost_bps", bps, benchmark_ret))

    return results


# ===========================================================================
# Test 4 — Factor Weight Robustness
# ===========================================================================

def _blend_composite(
    trad:     pd.DataFrame,
    hiring:   pd.DataFrame,
    sentiment: pd.DataFrame,
    w_t: float,
    w_h: float,
    w_s: float,
) -> pd.DataFrame:
    """
    Build a composite with specified weights (normalised to sum=1).

    Falls back to traditional score for stocks missing alt data.
    Identical logic to phase5_alt_factors.build_enhanced_composite()
    but extracted here so we can vary weights without re-running the
    expensive alt-data generation step.
    """
    total = w_t + w_h + w_s
    wt, wh, ws = w_t / total, w_h / total, w_s / total

    common_dates   = trad.index.intersection(hiring.index).intersection(sentiment.index)
    common_tickers = trad.columns.intersection(hiring.columns).intersection(sentiment.columns)

    T = trad.loc[common_dates, common_tickers]
    H = hiring.loc[common_dates, common_tickers]
    S = sentiment.loc[common_dates, common_tickers]

    blended   = wt * T + wh * H + ws * S
    all_valid = T.notna() & H.notna() & S.notna()
    blended   = blended.where(all_valid, other=T)

    blended = blended.apply(cross_sectional_zscore, axis=1)
    return blended


def test_factor_weights(
    trad_composite: pd.DataFrame,
    hiring_norm:    pd.DataFrame,
    sentiment_norm: pd.DataFrame,
    returns:        pd.DataFrame,
    weight_schemes: list[tuple[str, float, float, float]] | None = None,
    base_config:    PortfolioConfig | None = None,
    benchmark_ret:  pd.Series | None = None,
) -> list[BacktestResult]:
    """
    Test robustness to alternative factor weight allocations.

    WHY THIS TEST?
    --------------
    The 40/30/30 split was chosen by design (not data-mined), but we still
    need to verify the result is not sensitive to this choice.

    A robust multi-factor model should show:
      • Similar Sharpe across a range of weight schemes
      • Monotone improvement as alt data weight increases (if alt data has value)
      • Worst case: traditional-only should still show positive Sharpe

    WEIGHT SCHEMES TESTED
    ---------------------
    50/25/25 : Traditional-heavy (conservative allocation)
    40/30/30 : Our base case (balanced)
    60/20/20 : Even more traditional-heavy
    25/37.5/37.5 : Alt-data-heavy (aggressive allocation)
    33/33/33 : Equal weight (max entropy)

    Parameters
    ----------
    weight_schemes : list of (label, w_traditional, w_hiring, w_sentiment)
    """
    if weight_schemes is None:
        weight_schemes = [
            ("Trad 50 / Alt 50  (50/25/25)", 0.50, 0.25, 0.25),
            ("Trad 40 / Alt 60  (40/30/30)", 0.40, 0.30, 0.30),
            ("Trad 60 / Alt 40  (60/20/20)", 0.60, 0.20, 0.20),
            ("Trad 25 / Alt 75  (25/37/37)", 0.25, 0.375, 0.375),
            ("Equal weight      (33/33/33)", 1/3,  1/3,   1/3),
        ]
    if base_config is None:
        base_config = PortfolioConfig()

    log.info("TEST 4: Factor Weight Robustness")
    results = []

    for label, wt, wh, ws in weight_schemes:
        composite = _blend_composite(trad_composite, hiring_norm, sentiment_norm, wt, wh, ws)
        results.append(_run_one(composite, returns, base_config, label,
                                "weights", f"{wt:.0%}/{wh:.0%}/{ws:.0%}",
                                benchmark_ret))

    return results


# ===========================================================================
# Test 5 — Factor Ablation Study
# ===========================================================================

def test_ablation(
    trad_composite: pd.DataFrame,
    hiring_norm:    pd.DataFrame,
    sentiment_norm: pd.DataFrame,
    returns:        pd.DataFrame,
    base_config:    PortfolioConfig | None = None,
    benchmark_ret:  pd.Series | None = None,
) -> list[BacktestResult]:
    """
    Incrementally add each factor component and measure the marginal contribution.

    WHY ABLATION?
    -------------
    Ablation studies (removing one component at a time) are the standard
    way to measure the marginal contribution of each factor in machine
    learning and quantitative finance.

    The question being answered: "Does each signal add value independently,
    or is the performance driven by just one factor?"

    Expected pattern for a well-designed multi-factor model:
      Traditional only   → baseline Sharpe (e.g. 0.7)
      Trad + Hiring      → slightly better (e.g. 0.85)  [hiring adds ~0.15]
      Trad + Sentiment   → slightly better (e.g. 0.90)  [sentiment adds ~0.20]
      Full model         → best (e.g. 1.10)             [diversification bonus]

    If "Trad + Hiring" ≈ "Full model", then sentiment is redundant.
    If "Traditional only" ≈ "Full model", then alt data adds no value.

    INCREMENTAL ATTRIBUTION FORMULA
    --------------------------------
    Marginal contribution of signal X:
        ΔSharpe(X) = Sharpe(Full model) - Sharpe(Full model without X)

    We also report Δ vs traditional only, which shows the raw contribution
    of adding alt data at all.

    Ablation variants
    -----------------
    Model A: Traditional only   (Phase 3 composite)
    Model B: Trad + Hiring      (50/50 blend, no sentiment)
    Model C: Trad + Sentiment   (50/50 blend, no hiring)
    Model D: Full model         (40/30/30 baseline)
    """
    if base_config is None:
        base_config = PortfolioConfig()

    log.info("TEST 5: Factor Ablation Study")
    results = []

    # Model A: Traditional only
    results.append(_run_one(
        trad_composite, returns, base_config,
        "A: Traditional only", "ablation", "trad_only", benchmark_ret
    ))

    # Model B: Traditional + Hiring (50/50, no sentiment)
    b_composite = _blend_composite(trad_composite, hiring_norm,
                                   sentiment_norm, 0.50, 0.50, 0.00)
    results.append(_run_one(
        b_composite, returns, base_config,
        "B: Trad + Hiring (50/50)", "ablation", "trad_hiring", benchmark_ret
    ))

    # Model C: Traditional + Sentiment (50/50, no hiring)
    c_composite = _blend_composite(trad_composite, hiring_norm,
                                   sentiment_norm, 0.50, 0.00, 0.50)
    results.append(_run_one(
        c_composite, returns, base_config,
        "C: Trad + Sentiment (50/50)", "ablation", "trad_sentiment", benchmark_ret
    ))

    # Model D: Full model (40/30/30)
    d_composite = _blend_composite(trad_composite, hiring_norm,
                                   sentiment_norm, 0.40, 0.30, 0.30)
    results.append(_run_one(
        d_composite, returns, base_config,
        "D: Full model (40/30/30)", "ablation", "full_model", benchmark_ret
    ))

    return results


# ===========================================================================
# Summary helpers
# ===========================================================================

def results_to_df(results: list[BacktestResult]) -> pd.DataFrame:
    """Convert a list of BacktestResults to a summary DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def log_results_table(results: list[BacktestResult], title: str) -> None:
    """Pretty-print a results table to the log."""
    log.info(f"\n{'='*70}")
    log.info(f"  {title}")
    log.info(f"  {'Label':<38} {'CAGR':>7} {'SR':>7} {'MDD':>8} {'TO':>7}")
    log.info(f"  {'-'*70}")
    for r in results:
        log.info(
            f"  {r.label:<38} {r.cagr_pct:>6.2%}  {r.sharpe:>6.3f}  "
            f"{r.max_dd_pct:>7.2%}  {r.avg_turnover_pct:>6.1%}"
        )
    log.info(f"{'='*70}\n")
