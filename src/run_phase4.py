"""
run_phase4.py
-------------
Phase 4 Orchestrator — Portfolio Construction & Backtesting

Entry point: python src/run_phase4.py

Execution order
---------------
1. Load Phase-2 data  (prices, returns)
2. Load Phase-3 data  (composite factor scores)
3. portfolio.py       → weight schedule, turnover, cost drag
4. backtest.py        → daily NAV simulation, benchmark
5. performance.py     → statistics, figures, saved outputs

All intermediate results are returned and can be inspected interactively
in a notebook by importing and calling run_phase4() directly.

Author  : Quant Research Team
Phase   : 4 — Portfolio Construction & Backtesting
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio   import build_portfolio, PortfolioConfig
from backtest    import run_backtest
from performance import run_performance_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = ROOT / "data"
FIGURE_DIR = ROOT / "figures"


def run_phase4(
    top_quantile:         float = 0.20,
    transaction_cost_bps: float = 10.0,
    rebalance_freq:       str   = "BME",
    initial_nav:          float = 100.0,
) -> dict:
    """
    Run the complete Phase-4 pipeline.

    Parameters
    ----------
    top_quantile         : fraction of universe to select (default 0.20)
    transaction_cost_bps : one-way cost in basis points (default 10)
    rebalance_freq       : pandas offset alias for rebalancing (default "BME")
    initial_nav          : starting portfolio value (default 100)

    Returns
    -------
    dict with all intermediate and final results
    """
    log.info("=" * 62)
    log.info("PHASE 4 — PORTFOLIO CONSTRUCTION & BACKTESTING")
    log.info("=" * 62)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    log.info("Loading Phase-2 and Phase-3 data …")

    composite = pd.read_parquet(DATA_DIR / "factor_composite.parquet")
    returns   = pd.read_parquet(DATA_DIR / "returns_simple.parquet")
    prices    = pd.read_parquet(DATA_DIR / "prices_adjusted.parquet")

    log.info(
        f"  Composite factor: {composite.shape} | "
        f"Returns: {returns.shape} | Prices: {prices.shape}"
    )

    # -----------------------------------------------------------------------
    # 2. Portfolio construction
    # -----------------------------------------------------------------------
    config = PortfolioConfig(
        top_quantile=top_quantile,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq=rebalance_freq,
    )

    log.info(
        f"Config | top_quantile={top_quantile:.0%} | "
        f"cost={transaction_cost_bps}bps | freq={rebalance_freq}"
    )

    portfolio = build_portfolio(composite, config)

    # -----------------------------------------------------------------------
    # 3. Backtest
    # -----------------------------------------------------------------------
    backtest_results = run_backtest(portfolio, returns, initial_nav=initial_nav)

    # -----------------------------------------------------------------------
    # 4. Performance analysis
    # -----------------------------------------------------------------------
    perf = run_performance_analysis(
        backtest_results,
        data_dir=DATA_DIR,
        figure_dir=FIGURE_DIR,
    )

    log.info("=" * 62)
    log.info("PHASE 4 COMPLETE")
    log.info(f"  Data saved   → {DATA_DIR}")
    log.info(f"  Figures saved → {FIGURE_DIR}")
    log.info("=" * 62)

    return {
        "composite":         composite,
        "returns":           returns,
        "portfolio":         portfolio,
        "backtest_results":  backtest_results,
        "performance":       perf,
    }


if __name__ == "__main__":
    results = run_phase4()

    # Quick summary printout
    table = results["performance"]["table"]
    print("\n" + "=" * 55)
    print(f"{'Metric':<28} {'Strategy':>10} {'Benchmark':>10}")
    print("-" * 55)
    metrics = ["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar"]
    labels  = ["CAGR", "Ann. Vol", "Sharpe", "Max Drawdown", "Calmar"]
    for m, lbl in zip(metrics, labels):
        sv = table["strategy_net"][m]
        bv = table["benchmark"][m]
        fmt = ".2%" if m in ("cagr", "ann_vol", "max_drawdown") else ".3f"
        print(f"  {lbl:<26} {sv:{fmt}}{'':<4} {bv:{fmt}}")
    print("-" * 55)
    ir = table["relative"]["information_ratio"]
    alpha = table["relative"]["alpha_vs_benchmark"]
    avg_to = table["activity"]["avg_one_way_turnover"]
    print(f"  {'Alpha vs Benchmark':<26} {alpha:.2%}")
    print(f"  {'Information Ratio':<26} {ir:.3f}" if ir else "  IR: N/A")
    print(f"  {'Avg Monthly Turnover':<26} {avg_to:.1%}")
    print("=" * 55)
