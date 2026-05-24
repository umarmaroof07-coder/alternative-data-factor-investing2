"""
performance.py
--------------
Phase 4c — Performance Analytics & Visualisation

Computes institutional-grade performance statistics and generates
publication-quality figures for the research paper.

Metrics computed
----------------
Return metrics  : CAGR, total return, annualised return
Risk metrics    : Annualised vol, max drawdown, Calmar ratio
Risk-adj returns: Sharpe ratio, Sortino ratio, Information ratio
Activity metrics: Turnover, transaction costs, active share (approximated)

All statistics are computed for:
  • Strategy (gross of costs)
  • Strategy (net of costs)
  • Benchmark

The Sharpe ratio uses a 0% risk-free rate.  This is standard in
academic research.  For a production system, subtract the 3-month
T-bill rate.  With current Fed Funds rate ~4–5%, this matters a lot.

Author  : Quant Research Team
Phase   : 4 — Portfolio Construction & Backtesting
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style (matching Phase 3 conventions)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F7F7F7",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
})

COLORS = {
    "strategy_gross": "#4393C3",
    "strategy_net":   "#2166AC",
    "benchmark":      "#D6604D",
    "drawdown":       "#B2182B",
    "turnover":       "#4DAC26",
    "neutral":        "#636363",
}

ROOT       = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"
DATA_DIR   = ROOT / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 1 — Core performance statistics
# ===========================================================================

def cagr(nav: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    Formula: (NAV_end / NAV_start) ^ (252 / T) - 1

    We use 252 trading days per year (not calendar days) because our
    return series only contains trading days.  Using calendar days would
    understate CAGR by ~30% (365/252 ≈ 1.45x).

    WHY CAGR OVER ARITHMETIC MEAN?
    -------------------------------
    Arithmetic mean return of +10%, -10% over two periods implies you
    broke even — but (1.10)(0.90) = 0.99, so you actually lost 1%.
    CAGR captures this compounding effect and is the correct measure
    of wealth accumulation.
    """
    n_days = len(nav)
    if n_days < 2:
        return np.nan
    return (nav.iloc[-1] / nav.iloc[0]) ** (252 / n_days) - 1


def annualised_volatility(returns: pd.Series, ddof: int = 1) -> float:
    """
    Annualised standard deviation of daily returns.

    Formula: std(r) × sqrt(252)

    Using ddof=1 (Bessel correction) is the unbiased estimator for samples.
    For large T (>100 observations), ddof makes negligible difference.
    """
    return returns.std(ddof=ddof) * np.sqrt(252)


def sharpe_ratio(
    returns:  pd.Series,
    rf_daily: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Formula: sqrt(252) × mean(r - rf) / std(r)

    WHY ANNUALISE THIS WAY?
    -----------------------
    Daily Sharpe = mean(r)/std(r).  Annual Sharpe = Daily Sharpe × sqrt(252).
    This assumes returns are i.i.d. (independent and identically distributed).
    In reality equity returns have mild autocorrelation, so this slightly
    overstates the annual Sharpe — but the industry convention uses it anyway.

    A Sharpe > 1.0 is considered good.
    A Sharpe > 2.0 is exceptional (and often suspicious — check for errors).
    Most published factor strategies achieve 0.5–1.0 after costs.

    Parameters
    ----------
    rf_daily : daily risk-free rate.  0.0 for academic papers.
               Use (1 + fed_funds_rate) ** (1/252) - 1 for production.
    """
    excess = returns - rf_daily
    if excess.std() < 1e-10:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(
    returns:  pd.Series,
    rf_daily: float = 0.0,
    target:   float = 0.0,
) -> float:
    """
    Sortino ratio — like Sharpe but penalises only downside volatility.

    WHY SORTINO?
    ------------
    Sharpe treats upside and downside volatility equally.  A strategy with
    frequent small gains and rare large gains gets penalised by Sharpe.
    Sortino only penalises returns below the target (default: 0), which
    better represents an investor's actual loss aversion.

    Formula: sqrt(252) × mean(r - rf) / downside_std

    where downside_std = std of returns below target
    """
    excess = returns - rf_daily
    downside = returns[returns < target] - target
    if len(downside) < 2 or downside.std() < 1e-10:
        return np.nan
    return float(np.sqrt(252) * excess.mean() / downside.std(ddof=1))


def max_drawdown(nav: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in the NAV curve.

    Formula: min( NAV(t) / max(NAV[0:t]) - 1 )

    WHY MAX DRAWDOWN?
    -----------------
    Volatility measures the *frequency* of losses; max drawdown measures
    the *severity* of the worst loss.  Institutional allocators use max
    drawdown as a key risk constraint.  A fund with 30% max drawdown will
    struggle to raise institutional capital regardless of its Sharpe ratio
    (because the investor who allocated at the peak is down 30% and may
    redeem, creating a liquidity crisis).

    Returns
    -------
    float — negative number (e.g. -0.25 = 25% max drawdown)
    """
    running_max = nav.cummax()
    drawdown    = nav / running_max - 1
    return float(drawdown.min())


def calmar_ratio(nav: pd.Series, returns: pd.Series) -> float:
    """
    CAGR divided by absolute max drawdown.

    A Calmar > 0.5 is reasonable; > 1.0 is excellent.
    Used by CTAs and macro funds as the primary risk-adjusted return metric.
    """
    mdd = abs(max_drawdown(nav))
    if mdd < 1e-10:
        return np.nan
    return cagr(nav) / mdd


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information ratio = annualised active return / tracking error.

    Active return   = strategy_return - benchmark_return
    Tracking error  = std(active_return) × sqrt(252)

    WHY INFORMATION RATIO?
    ----------------------
    The IR measures how efficiently the manager converts active risk into
    active return.  An IR > 0.5 is considered good for a long-only strategy.
    An IR > 1.0 is exceptional.

    Unlike Sharpe, IR is benchmark-relative — it tells you how good the
    *stock selection* is, independent of market direction.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    active = strategy_returns.loc[common] - benchmark_returns.loc[common]
    te = active.std(ddof=1) * np.sqrt(252)
    if te < 1e-10:
        return np.nan
    return float(active.mean() * 252 / te)


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Return the full drawdown time series (for plotting)."""
    return nav / nav.cummax() - 1


# ===========================================================================
# 2 — Full performance table
# ===========================================================================

def compute_performance_table(
    nav_df:    pd.DataFrame,
    benchmark: pd.DataFrame,
    turnover:  pd.Series,
    cost_bps:  float = 10.0,
) -> dict:
    """
    Compute the complete performance summary for strategy and benchmark.

    Returns a nested dict:
        {
          "strategy_gross": { metric: value, ... },
          "strategy_net":   { metric: value, ... },
          "benchmark":      { metric: value, ... },
          "relative":       { metric: value, ... },
          "activity":       { metric: value, ... },
        }
    """
    # Align dates
    common = nav_df.index.intersection(benchmark.index)
    nav_g  = nav_df.loc[common, "nav_gross"]
    nav_n  = nav_df.loc[common, "nav_net"]
    r_g    = nav_df.loc[common, "gross_return"]
    r_n    = nav_df.loc[common, "net_return"]
    bm_nav = benchmark.loc[common, "bm_nav"]
    bm_ret = benchmark.loc[common, "bm_return"]

    def stats(nav, rets, label):
        return {
            "label":            label,
            "total_return":     round(nav.iloc[-1] / nav.iloc[0] - 1, 4),
            "cagr":             round(cagr(nav), 4),
            "ann_vol":          round(annualised_volatility(rets), 4),
            "sharpe":           round(sharpe_ratio(rets), 4),
            "sortino":          round(sortino_ratio(rets), 4),
            "max_drawdown":     round(max_drawdown(nav), 4),
            "calmar":           round(calmar_ratio(nav, rets), 4),
        }

    s_gross = stats(nav_g, r_g, "Strategy (Gross)")
    s_net   = stats(nav_n, r_n, "Strategy (Net)")
    bm      = stats(bm_nav, bm_ret, "EW Benchmark")

    # Relative / active stats
    ir   = information_ratio(r_n, bm_ret)
    ann_active = (r_n - bm_ret.reindex(r_n.index).fillna(0)).mean() * 252
    te         = (r_n - bm_ret.reindex(r_n.index).fillna(0)).std() * np.sqrt(252)

    # Activity stats
    rebal_turnover = turnover[turnover > 0]
    ann_cost       = -(nav_df.loc[common, "cost_drag"].sum()) / (len(common) / 252)

    table = {
        "strategy_gross": s_gross,
        "strategy_net":   s_net,
        "benchmark":      bm,
        "relative": {
            "active_return_ann":  round(ann_active, 4),
            "tracking_error_ann": round(te, 4),
            "information_ratio":  round(ir, 4) if not np.isnan(ir) else None,
            "alpha_vs_benchmark": round(s_net["cagr"] - bm["cagr"], 4),
        },
        "activity": {
            "n_rebalances":        int(len(rebal_turnover)),
            "avg_one_way_turnover": round(float(rebal_turnover.mean()), 4),
            "ann_turnover":        round(float(rebal_turnover.mean() * 12), 4),
            "est_ann_cost_drag":   round(float(ann_cost), 4),
            "cost_bps_one_way":    cost_bps,
        },
    }

    _log_table(table)
    return table


def _log_table(table: dict) -> None:
    """Pretty-print the performance table to the log."""
    log.info("=" * 62)
    log.info("PERFORMANCE SUMMARY")
    log.info("=" * 62)
    for section, metrics in table.items():
        log.info(f"  [{section.upper()}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                if abs(v) < 1 and k not in ("sharpe", "sortino", "calmar",
                                              "information_ratio", "alpha_vs_benchmark"):
                    log.info(f"    {k:<28}: {v:>8.2%}")
                else:
                    log.info(f"    {k:<28}: {v:>8.4f}")
            else:
                log.info(f"    {k:<28}: {v}")
    log.info("=" * 62)


# ===========================================================================
# 3 — Figures
# ===========================================================================

def plot_nav_curves(nav_df: pd.DataFrame, benchmark: pd.DataFrame) -> plt.Figure:
    """
    Panel 1 — Cumulative NAV: gross, net, benchmark
    Panel 2 — Drawdown of net strategy and benchmark

    The NAV chart is the primary deliverable for any backtest.
    Showing gross and net together immediately communicates how much
    of the alpha is consumed by transaction costs.
    """
    common  = nav_df.index.intersection(benchmark.index)
    nav_g   = nav_df.loc[common, "nav_gross"]
    nav_n   = nav_df.loc[common, "nav_net"]
    bm_nav  = benchmark.loc[common, "bm_nav"]
    dd_n    = drawdown_series(nav_n)
    dd_bm   = drawdown_series(bm_nav)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # — NAV panel —
    ax1.plot(common, nav_g,  color=COLORS["strategy_gross"], lw=1.2,
             linestyle="--", alpha=0.7, label="Strategy (Gross)")
    ax1.plot(common, nav_n,  color=COLORS["strategy_net"],   lw=1.5,
             label="Strategy (Net of 10bps)")
    ax1.plot(common, bm_nav, color=COLORS["benchmark"],      lw=1.2,
             label="EW Benchmark")

    ax1.set_ylabel("NAV (base = 100)")
    ax1.set_title("Multi-Factor Equity Strategy — Cumulative NAV", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Shade the outperformance region
    outperf = nav_n > bm_nav
    ax1.fill_between(common, nav_n, bm_nav,
                     where=outperf,  alpha=0.10, color=COLORS["strategy_net"],
                     label="Outperformance")
    ax1.fill_between(common, nav_n, bm_nav,
                     where=~outperf, alpha=0.10, color=COLORS["benchmark"])

    # — Drawdown panel —
    ax2.fill_between(dd_n.index,  dd_n,  0, color=COLORS["strategy_net"],
                     alpha=0.5, label="Strategy DD")
    ax2.fill_between(dd_bm.index, dd_bm, 0, color=COLORS["benchmark"],
                     alpha=0.4, label="Benchmark DD")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown", loc="left", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc="lower left", ncol=2)

    plt.tight_layout()
    return fig


def plot_rolling_metrics(
    nav_df:    pd.DataFrame,
    benchmark: pd.DataFrame,
) -> plt.Figure:
    """
    Rolling 12-month: Sharpe ratio, annualised return, active return.

    WHY ROLLING METRICS?
    --------------------
    A strategy's backtest Sharpe of 0.8 could mean consistently 0.8 every
    year, or alternating 2.0 / -0.4 years.  These have very different
    institutional implications.  Rolling metrics reveal regime dependence,
    show whether alpha is concentrated in one period, and expose structural
    breaks in the signal.
    """
    common   = nav_df.index.intersection(benchmark.index)
    r_n      = nav_df.loc[common, "net_return"]
    bm_ret   = benchmark.loc[common, "bm_return"]
    active   = r_n - bm_ret

    window = 252

    # Rolling annualised return
    roll_ret = r_n.rolling(window).apply(
        lambda x: (1 + x).prod() ** (252 / len(x)) - 1, raw=True
    )
    roll_bm = bm_ret.rolling(window).apply(
        lambda x: (1 + x).prod() ** (252 / len(x)) - 1, raw=True
    )

    # Rolling Sharpe
    roll_sharpe = r_n.rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0, raw=True
    )

    # Rolling active return
    roll_active = active.rolling(window).apply(
        lambda x: x.mean() * 252, raw=True
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Rolling 12-Month Performance Metrics", fontsize=11)

    # — Rolling return —
    ax = axes[0]
    ax.plot(roll_ret.index,  roll_ret,  color=COLORS["strategy_net"], lw=1.2,
            label="Strategy (Net)")
    ax.plot(roll_bm.index,   roll_bm,   color=COLORS["benchmark"],    lw=1.1,
            linestyle="--", label="Benchmark")
    ax.axhline(0, color="#888", lw=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Ann. Return")
    ax.set_title("Rolling 12-Month Annualised Return", loc="left", fontsize=9)
    ax.legend(ncol=2)

    # — Rolling Sharpe —
    ax = axes[1]
    ax.plot(roll_sharpe.index, roll_sharpe, color=COLORS["strategy_net"], lw=1.2)
    ax.axhline(1.0, color="#888", lw=0.8, linestyle="--", label="Sharpe = 1.0")
    ax.axhline(0.0, color="#888", lw=0.7, linestyle=":")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Rolling 12-Month Sharpe Ratio", loc="left", fontsize=9)
    ax.legend()

    # — Rolling active return —
    ax = axes[2]
    pos = roll_active > 0
    ax.fill_between(roll_active.index, roll_active, 0,
                    where=pos,  color=COLORS["strategy_net"], alpha=0.55,
                    label="Outperformance")
    ax.fill_between(roll_active.index, roll_active, 0,
                    where=~pos, color=COLORS["benchmark"],    alpha=0.55,
                    label="Underperformance")
    ax.axhline(0, color="#888", lw=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Active Return")
    ax.set_xlabel("Date")
    ax.set_title("Rolling 12-Month Active Return vs Benchmark", loc="left", fontsize=9)
    ax.legend(ncol=2)

    plt.tight_layout()
    return fig


def plot_turnover_and_costs(
    nav_df:   pd.DataFrame,
    turnover: pd.Series,
    cost_bps: float = 10.0,
) -> plt.Figure:
    """
    Two-panel: monthly turnover bar chart + cumulative cost drag.

    WHY SHOW COSTS SEPARATELY?
    --------------------------
    Many practitioners present gross-of-cost backtests and mention costs
    as a footnote.  Showing the cumulative cost drag line makes clear that
    transaction costs are a real, compounding drag — not an abstraction.
    At 10bps/rebalance × 12 rebalances = 120bps gross drag, but the
    compounding on reinvested savings makes the actual cost ~3–4% of
    terminal wealth over 5 years.
    """
    # Monthly turnover: aggregate to rebalance events
    rebal_turnover = turnover[turnover > 0.001]
    monthly_to     = rebal_turnover.resample("ME").sum()

    # Cumulative cost drag (convert to positive = money lost)
    cum_cost = (-nav_df["cost_drag"]).cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    fig.suptitle("Portfolio Activity: Turnover & Transaction Cost Drag", fontsize=11)

    # — Turnover bar chart —
    ax1.bar(monthly_to.index, monthly_to.values,
            color=COLORS["turnover"], alpha=0.75, width=20)
    avg_to = rebal_turnover.mean()
    ax1.axhline(avg_to, color="#333", lw=1.0, linestyle="--",
                label=f"Avg one-way TO = {avg_to:.1%}")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_ylabel("One-Way Turnover")
    ax1.set_title("Monthly Rebalance Turnover", loc="left", fontsize=9)
    ax1.legend()

    # — Cumulative cost drag —
    ax2.fill_between(cum_cost.index, cum_cost, 0,
                     color=COLORS["drawdown"], alpha=0.5)
    ax2.plot(cum_cost.index, cum_cost,
             color=COLORS["drawdown"], lw=1.2, label=f"Cum. cost ({cost_bps}bps/trade)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylabel("Cumulative Drag (% NAV)")
    ax2.set_xlabel("Date")
    ax2.set_title("Cumulative Transaction Cost Drag", loc="left", fontsize=9)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_annual_bar_chart(
    nav_df:    pd.DataFrame,
    benchmark: pd.DataFrame,
) -> plt.Figure:
    """
    Grouped bar chart of annual returns: strategy (net) vs benchmark.

    Institutional standard for presenting year-by-year performance.
    Red/green colouring of active return shows alpha consistency at a glance.
    """
    common = nav_df.index.intersection(benchmark.index)
    r_n    = nav_df.loc[common, "net_return"]
    bm_ret = benchmark.loc[common, "bm_return"]

    # Calendar year returns
    def annual_returns(ret_series):
        return ret_series.resample("YE").apply(lambda x: (1 + x).prod() - 1)

    ann_strat = annual_returns(r_n)
    ann_bm    = annual_returns(bm_ret)
    years     = [d.year for d in ann_strat.index]

    x    = np.arange(len(years))
    w    = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Annual Return Breakdown", fontsize=11)

    # — Grouped bar —
    bars_s = ax1.bar(x - w/2, ann_strat.values, w,
                     color=COLORS["strategy_net"], label="Strategy (Net)", alpha=0.85)
    bars_b = ax1.bar(x + w/2, ann_bm.values,    w,
                     color=COLORS["benchmark"],   label="Benchmark",       alpha=0.85)

    ax1.axhline(0, color="#444", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_ylabel("Annual Return")
    ax1.set_title("Annual Returns: Strategy vs Benchmark", loc="left", fontsize=9)
    ax1.legend()

    # — Active return —
    active  = ann_strat.values - ann_bm.reindex(ann_strat.index).fillna(0).values
    colours = [COLORS["strategy_net"] if a > 0 else COLORS["benchmark"] for a in active]
    ax2.bar(x, active, color=colours, alpha=0.8)
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylabel("Active Return")
    ax2.set_title("Annual Active Return (Strategy − Benchmark)", loc="left", fontsize=9)

    legend_elems = [
        Line2D([0], [0], color=COLORS["strategy_net"], lw=0, marker="s",
               markersize=8, label="Outperformance"),
        Line2D([0], [0], color=COLORS["benchmark"],    lw=0, marker="s",
               markersize=8, label="Underperformance"),
    ]
    ax2.legend(handles=legend_elems)

    plt.tight_layout()
    return fig


def plot_holdings_heatmap(
    holdings: dict,
    all_tickers: list[str],
    max_dates: int = 60,
) -> plt.Figure:
    """
    Binary heatmap: which stocks are held on each rebalance date.

    Rows = rebalance dates (most recent at top)
    Cols = tickers (sorted by total holding frequency)

    Shows factor persistence, sector clustering, and concentration risk
    at a glance.  A stock that appears in every single column is either
    a very strong factor stock or a data quality issue — worth investigating.
    """
    dates  = sorted(holdings.keys())[-max_dates:]
    matrix = pd.DataFrame(0, index=dates, columns=all_tickers)

    for date in dates:
        for ticker in holdings[date]:
            if ticker in matrix.columns:
                matrix.loc[date, ticker] = 1

    # Sort columns by holding frequency (most-held on left)
    freq_order = matrix.sum().sort_values(ascending=False)
    matrix = matrix[freq_order.index]
    # Keep only tickers that were held at least once
    matrix = matrix.loc[:, matrix.sum() > 0]

    fig, ax = plt.subplots(figsize=(14, max(4, len(dates) * 0.18)))
    im = ax.imshow(
        matrix.values, aspect="auto", cmap="Blues",
        vmin=0, vmax=1, interpolation="nearest",
    )

    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels([d.strftime("%Y-%m") for d in dates], fontsize=6)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_title(
        "Portfolio Holdings Heatmap\n(blue = held, white = not held; "
        "columns sorted by holding frequency)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


# ===========================================================================
# 4 — Save results
# ===========================================================================

def save_results(
    nav_df:    pd.DataFrame,
    benchmark: pd.DataFrame,
    table:     dict,
    data_dir:  Path = DATA_DIR,
) -> None:
    """Save NAV curves and performance table to data/."""
    data_dir.mkdir(parents=True, exist_ok=True)
    nav_df.to_parquet(data_dir / "backtest_nav.parquet")
    benchmark.to_parquet(data_dir / "backtest_benchmark.parquet")
    (data_dir / "performance_table.json").write_text(
        json.dumps(table, indent=2, default=str), encoding="utf-8"
    )
    log.info(f"Results saved to {data_dir}")


def save_figures(figures: dict[str, plt.Figure], fig_dir: Path = FIGURE_DIR) -> None:
    """Save all figures as PNG."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved figure → {path.name}")


# ===========================================================================
# 5 — Full performance pipeline
# ===========================================================================

def run_performance_analysis(
    backtest_results: dict,
    data_dir:   Path = DATA_DIR,
    figure_dir: Path = FIGURE_DIR,
) -> dict:
    """
    Orchestrate the full performance analysis and output generation.

    Parameters
    ----------
    backtest_results : output dict from backtest.run_backtest()

    Returns
    -------
    dict with performance table and figure paths
    """
    nav_df    = backtest_results["nav_df"]
    benchmark = backtest_results["benchmark"]
    portfolio = backtest_results["portfolio"]
    turnover  = portfolio["turnover"]
    cost_bps  = portfolio["config"].transaction_cost_bps

    log.info("Computing performance statistics …")
    table = compute_performance_table(nav_df, benchmark, turnover, cost_bps)

    log.info("Generating figures …")
    figs = {
        "backtest_nav_curves":       plot_nav_curves(nav_df, benchmark),
        "backtest_rolling_metrics":  plot_rolling_metrics(nav_df, benchmark),
        "backtest_turnover_costs":   plot_turnover_and_costs(nav_df, turnover, cost_bps),
        "backtest_annual_returns":   plot_annual_bar_chart(nav_df, benchmark),
        "backtest_holdings_heatmap": plot_holdings_heatmap(
            portfolio["holdings"],
            list(nav_df.columns) if hasattr(nav_df, "columns") else
            list(portfolio["weights"].columns),
        ),
    }
    save_figures(figs, figure_dir)
    save_results(nav_df, benchmark, table, data_dir)

    return {"table": table, "figures": list(figs.keys())}
