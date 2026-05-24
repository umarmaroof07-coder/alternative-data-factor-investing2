"""
phase6_robustness.py
--------------------
Phase 6 Orchestrator — Robustness Testing & Publication Output

Runs all six robustness tests and produces a complete publication-ready
analysis package:

  Test 1 — Portfolio Concentration Sensitivity   (3 variants)
  Test 2 — Rebalance Frequency Sensitivity       (3 variants)
  Test 3 — Transaction Cost Stress Test          (5 variants)
  Test 4 — Factor Weight Robustness              (5 variants)
  Test 5 — Market Regime Analysis                (4 regimes × 3 strategies)
  Test 6 — Factor Ablation Study                 (4 variants)

Total: 23 backtests + regime analysis

Output files
------------
  data/
    phase6_robustness_summary.json      — all scalar results (machine-readable)
    phase6_regime_analysis.json         — regime breakdown tables

  figures/
    phase6_concentration.png
    phase6_frequency.png
    phase6_cost_stress.png
    phase6_weights.png
    phase6_regime_bars.png
    phase6_regime_nav.png
    phase6_ablation.png
    phase6_summary_heatmap.png

Author  : Quant Research Team
Phase   : 6 — Robustness Testing
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robustness     import (
    BacktestResult,
    test_concentration, test_rebalance_frequency,
    test_transaction_costs, test_factor_weights,
    test_ablation, results_to_df, log_results_table,
)
from regime_analysis import (
    run_regime_analysis, RegimeLabels,
    regime_stats_to_df, classify_regimes,
)
from portfolio   import PortfolioConfig
from backtest    import run_backtest, build_benchmark, build_benchmark
from performance import (
    cagr, sharpe_ratio, max_drawdown, annualised_volatility, save_figures,
    drawdown_series,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

# ---------------------------------------------------------------------------
# Shared plot style
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

METRIC_LABELS = {
    "cagr_pct":    "CAGR",
    "sharpe":      "Sharpe Ratio",
    "max_dd_pct":  "Max Drawdown",
    "ann_vol_pct": "Ann. Volatility",
    "calmar":      "Calmar Ratio",
}

COLORS_MAIN  = ["#2166AC", "#4DAC26", "#D6604D", "#E08214", "#8073AC"]
COLOR_TRAD   = "#2166AC"
COLOR_ENH    = "#4DAC26"
COLOR_BM     = "#D6604D"


# ===========================================================================
# Figure generators
# ===========================================================================

def _bar_metric_grid(
    results:    list[BacktestResult],
    metrics:    list[str],
    title:      str,
    x_label:    str,
    colors:     list[str] | None = None,
) -> plt.Figure:
    """
    Generic 2×2 grid of bar charts: one panel per metric.

    Used for concentration, frequency, cost, weight, and ablation tests.
    All share the same layout for visual consistency across the paper.
    """
    if colors is None:
        colors = COLORS_MAIN[: len(results)]

    labels = [r.label for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, metric in zip(axes.flat, metrics):
        values = [getattr(r, metric) for r in results]
        bars   = ax.bar(range(len(labels)), values, color=colors[: len(labels)],
                        alpha=0.82, edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + abs(bar.get_height()) * 0.02
            if bar.get_height() < 0:
                y_pos = bar.get_height() - abs(bar.get_height()) * 0.12
            fmt = f"{val:.1%}" if metric in ("cagr_pct","max_dd_pct","ann_vol_pct","avg_turnover_pct") else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    fmt, ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left", fontsize=9)
        ax.axhline(0, color="#555", lw=0.7, linestyle=":")

        if metric in ("cagr_pct","max_dd_pct","ann_vol_pct","avg_turnover_pct"):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    return fig


def _nav_comparison(
    results: list[BacktestResult],
    title:   str,
    colors:  list[str] | None = None,
) -> plt.Figure:
    """NAV + drawdown comparison for any set of BacktestResults."""
    if colors is None:
        colors = COLORS_MAIN[: len(results)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for r, c in zip(results, colors):
        if r.nav_series.empty:
            continue
        ax1.plot(r.nav_series.index, r.nav_series, color=c, lw=1.3, label=r.label)
        dd = drawdown_series(r.nav_series)
        ax2.fill_between(dd.index, dd, 0, color=c, alpha=0.35)
        ax2.plot(dd.index, dd, color=c, lw=0.8)

    ax1.set_ylabel("NAV (base = 100)")
    ax1.legend(loc="upper left", ncol=2, fontsize=7)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 1 — Concentration
# ---------------------------------------------------------------------------

def plot_concentration(results: list[BacktestResult]) -> plt.Figure:
    colors = ["#2166AC", "#4393C3", "#92C5DE"]
    return _bar_metric_grid(
        results,
        metrics=["cagr_pct","sharpe","max_dd_pct","avg_turnover_pct"],
        title="Test 1: Portfolio Concentration Sensitivity\n"
              "(equal cost, monthly rebalance; all else equal)",
        x_label="Top-N% Selection",
        colors=colors,
    )


# ---------------------------------------------------------------------------
# Fig 2 — Rebalance Frequency
# ---------------------------------------------------------------------------

def plot_frequency(results: list[BacktestResult]) -> plt.Figure:
    colors = ["#D6604D", "#F4A582", "#FDDBC7"]
    return _bar_metric_grid(
        results,
        metrics=["cagr_pct","sharpe","max_dd_pct","avg_turnover_pct"],
        title="Test 2: Rebalance Frequency Sensitivity\n"
              "(top 20%, 10bps cost; all else equal)",
        x_label="Rebalance Frequency",
        colors=colors,
    )


# ---------------------------------------------------------------------------
# Fig 3 — Transaction Cost Stress Test
# ---------------------------------------------------------------------------

def plot_cost_stress(results: list[BacktestResult]) -> plt.Figure:
    """
    Special layout: line chart of Sharpe vs cost (shows breakeven clearly)
    plus a 3-panel bar for CAGR, vol, max DD.
    """
    costs  = [r.param_value for r in results]
    sharpes= [r.sharpe      for r in results]
    cagrs  = [r.cagr_pct    for r in results]
    vols   = [r.ann_vol_pct for r in results]
    dds    = [r.max_dd_pct  for r in results]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "Test 3: Transaction Cost Stress Test\n"
        "(top 20%, monthly rebalance; only cost varies)",
        fontsize=11, fontweight="bold",
    )

    # Top: Sharpe vs cost line chart (main result)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(costs, sharpes, "o-", color="#2166AC", lw=2, ms=7)
    ax_main.axhline(0, color="#B2182B", lw=1.2, linestyle="--", label="Zero Sharpe (breakeven)")
    ax_main.axhline(0.5, color="#4DAC26", lw=0.8, linestyle=":", label="Sharpe = 0.5 (good)")

    # Mark breakeven via linear interpolation
    for i in range(len(sharpes) - 1):
        if (sharpes[i] >= 0) != (sharpes[i+1] >= 0):
            x_be = costs[i] + (0 - sharpes[i]) * (costs[i+1]-costs[i]) / (sharpes[i+1]-sharpes[i])
            ax_main.axvline(x_be, color="#B2182B", lw=1.0, linestyle=":", alpha=0.7)
            ax_main.text(x_be + 0.5, ax_main.get_ylim()[0]*0.9,
                         f"Breakeven\n~{x_be:.0f}bps",
                         ha="left", fontsize=8, color="#B2182B")

    for x, y in zip(costs, sharpes):
        ax_main.annotate(f"{y:.3f}", (x, y),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, fontweight="bold")

    ax_main.set_xlabel("One-Way Transaction Cost (bps)")
    ax_main.set_ylabel("Sharpe Ratio")
    ax_main.set_title("Sharpe Ratio vs Transaction Cost", loc="left", fontsize=9)
    ax_main.set_xticks(costs)
    ax_main.legend()

    # Bottom: CAGR, Vol, Max DD
    for ax, vals, lbl, pct in zip(
        [fig.add_subplot(gs[1, i]) for i in range(3)],
        [cagrs, vols, dds],
        ["CAGR", "Annualised Volatility", "Max Drawdown"],
        [True, True, True],
    ):
        clrs = ["#4DAC26" if v > 0 else "#D6604D" for v in vals]
        ax.bar(range(len(costs)), vals, color=clrs, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(costs)))
        ax.set_xticklabels([f"{c}bps" for c in costs], fontsize=7)
        ax.set_ylabel(lbl)
        ax.set_title(lbl, loc="left", fontsize=9)
        ax.axhline(0, color="#555", lw=0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Factor Weights
# ---------------------------------------------------------------------------

def plot_weights(results: list[BacktestResult]) -> plt.Figure:
    colors = ["#8073AC","#2166AC","#4DAC26","#E08214","#D6604D"]
    return _bar_metric_grid(
        results,
        metrics=["cagr_pct","sharpe","max_dd_pct","calmar"],
        title="Test 4: Factor Weight Robustness\n"
              "(top 20%, monthly, 10bps; weight allocation varies)",
        x_label="Factor Weight Scheme",
        colors=colors,
    )


# ---------------------------------------------------------------------------
# Fig 5 — Regime Analysis: Bar Charts
# ---------------------------------------------------------------------------

def plot_regime_bars(regime_results: dict) -> plt.Figure:
    """
    Grouped bar chart: annualised return and Sharpe by regime.

    Each regime gets three bars: Traditional, Enhanced, Benchmark.
    This is the most compact way to show regime attribution.
    """
    regimes   = RegimeLabels.ORDERED
    stats_map = {
        "Traditional": regime_results["stats_traditional"],
        "Enhanced":    regime_results["stats_enhanced"],
        "Benchmark":   regime_results["stats_benchmark"],
    }

    # Build DataFrames indexed by regime
    ret_data   = {}
    sr_data    = {}
    hit_data   = {}
    mdd_data   = {}

    for strat_label, stats_list in stats_map.items():
        stats_by_regime = {s.regime: s for s in stats_list}
        ret_data[strat_label]  = [stats_by_regime.get(r, None) for r in regimes]
        sr_data[strat_label]   = ret_data[strat_label]
        hit_data[strat_label]  = ret_data[strat_label]
        mdd_data[strat_label]  = ret_data[strat_label]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Test 5: Market Regime Analysis\n"
        "(Traditional vs Enhanced vs EW Benchmark)",
        fontsize=12, fontweight="bold",
    )

    metric_getters = [
        ("ann_return",  "Annualised Return",    True,  axes[0,0]),
        ("sharpe",      "Sharpe Ratio",         False, axes[0,1]),
        ("hit_rate",    "Hit Rate (% days +)",  True,  axes[1,0]),
        ("max_dd",      "Max Drawdown",         True,  axes[1,1]),
    ]

    strat_colors = [COLOR_TRAD, COLOR_ENH, COLOR_BM]
    strat_labels = list(stats_map.keys())
    n_strats     = len(strat_labels)
    bar_width    = 0.25
    x            = np.arange(len(regimes))

    for metric, ylabel, pct_fmt, ax in metric_getters:
        for i, (strat_lbl, stats_list) in enumerate(stats_map.items()):
            stats_by_regime = {s.regime: s for s in stats_list}
            vals = []
            for r in regimes:
                s = stats_by_regime.get(r)
                v = getattr(s, metric) if s else np.nan
                vals.append(v if not np.isnan(v) else 0.0)

            offset = (i - n_strats / 2 + 0.5) * bar_width
            bars   = ax.bar(x + offset, vals, bar_width,
                            color=strat_colors[i], alpha=0.80,
                            label=strat_lbl, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, loc="left", fontsize=9)
        ax.axhline(0, color="#555", lw=0.7, linestyle=":")
        if pct_fmt:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        if ax == axes[0,0]:
            ax.legend(fontsize=8)

        # Colour x-axis labels by regime
        for tick, regime in zip(ax.get_xticklabels(), regimes):
            tick.set_color(RegimeLabels.COLORS.get(regime, "#333333"))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6 — Regime Overlay NAV
# ---------------------------------------------------------------------------

def plot_regime_nav(
    regime_results: dict,
    trad_nav:       pd.Series,
    enh_nav:        pd.Series,
    bm_nav:         pd.Series,
) -> plt.Figure:
    """
    NAV curves with regime shading.

    Background colour shows which regime is active on each day.
    This allows visual inspection of *when* each strategy outperforms.
    """
    regimes = regime_results["regimes"]
    common  = trad_nav.index.intersection(enh_nav.index).intersection(regimes.index)
    t_nav   = trad_nav.loc[common]
    e_nav   = enh_nav.loc[common]
    b_nav   = bm_nav.loc[common]
    reg     = regimes.loc[common]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.suptitle("Market Regime Overlay — NAV Curves", fontsize=11, fontweight="bold")

    # Regime background shading
    regime_alpha = 0.12
    for ax in [ax1, ax2]:
        prev_r  = None
        start_i = 0
        for i, (date, r) in enumerate(reg.items()):
            if r != prev_r:
                if prev_r is not None:
                    end_date = common[i-1]
                    c = RegimeLabels.COLORS.get(prev_r, "#cccccc")
                    ax.axvspan(common[start_i], end_date, alpha=regime_alpha,
                               color=c, linewidth=0)
                prev_r  = r
                start_i = i
        if prev_r is not None:
            c = RegimeLabels.COLORS.get(prev_r, "#cccccc")
            ax.axvspan(common[start_i], common[-1], alpha=regime_alpha,
                       color=c, linewidth=0)

    ax1.plot(common, t_nav, color=COLOR_TRAD, lw=1.4, linestyle="--", label="Traditional")
    ax1.plot(common, e_nav, color=COLOR_ENH,  lw=1.8, label="Enhanced (Phase 5)")
    ax1.plot(common, b_nav, color=COLOR_BM,   lw=1.0, alpha=0.6, label="EW Benchmark")
    ax1.set_ylabel("NAV (base = 100)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Legend including regime patches
    regime_patches = [
        Patch(facecolor=c, alpha=0.4, label=r)
        for r, c in RegimeLabels.COLORS.items()
        if r in reg.values
    ]
    ax1.legend(handles=[
        Line2D([0],[0], color=COLOR_TRAD, lw=1.4, ls="--", label="Traditional"),
        Line2D([0],[0], color=COLOR_ENH,  lw=1.8, label="Enhanced"),
        Line2D([0],[0], color=COLOR_BM,   lw=1.0, label="Benchmark"),
    ] + regime_patches, loc="upper left", ncol=4, fontsize=7)

    # Active return
    active = e_nav / t_nav * 100 - 100
    pos = active > 0
    ax2.fill_between(common, active, 0, where=pos,  color=COLOR_ENH,  alpha=0.5)
    ax2.fill_between(common, active, 0, where=~pos, color=COLOR_TRAD, alpha=0.4)
    ax2.axhline(0, color="#444", lw=0.7)
    ax2.set_ylabel("Enh vs Trad (%)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 7 — Ablation Study
# ---------------------------------------------------------------------------

def plot_ablation(results: list[BacktestResult]) -> plt.Figure:
    """
    Waterfall-style incremental attribution + NAV comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Test 6: Factor Ablation Study\n"
        "(incremental contribution of each signal)",
        fontsize=11, fontweight="bold",
    )

    colors = ["#888888", "#8073AC", "#E08214", "#2166AC"]
    labels = [r.label.split(":")[1].strip() if ":" in r.label else r.label for r in results]

    # Panel 1: Sharpe bar
    sharpes = [r.sharpe for r in results]
    bars = axes[0].bar(range(len(results)), sharpes, color=colors, alpha=0.82,
                       edgecolor="white")
    for bar, val in zip(bars, sharpes):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Sharpe Ratio by Model", loc="left", fontsize=9)
    axes[0].axhline(sharpes[0], color="#888", lw=0.8, linestyle="--",
                    label=f"Baseline = {sharpes[0]:.3f}")
    axes[0].legend(fontsize=7)

    # Panel 2: CAGR bar
    cagrs = [r.cagr_pct for r in results]
    colors2 = ["#4DAC26" if v > 0 else "#D6604D" for v in cagrs]
    axes[1].bar(range(len(results)), cagrs, color=colors2, alpha=0.82, edgecolor="white")
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[1].set_ylabel("CAGR")
    axes[1].set_title("CAGR by Model", loc="left", fontsize=9)
    axes[1].axhline(0, color="#555", lw=0.7)

    # Panel 3: NAV curves
    for r, c in zip(results, colors):
        if not r.nav_series.empty:
            axes[2].plot(r.nav_series.index, r.nav_series, color=c, lw=1.3, label=r.label.split(":")[0])
    axes[2].set_ylabel("NAV (base = 100)")
    axes[2].set_title("NAV Comparison", loc="left", fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 8 — Summary Heatmap (all tests in one table)
# ---------------------------------------------------------------------------

def plot_summary_heatmap(all_results: dict[str, list[BacktestResult]]) -> plt.Figure:
    """
    Publication-style heatmap: rows = variants, cols = metrics.

    All 23 backtest results in one visual.  Cell colour = percentile rank
    among all variants for that metric (green = top quartile, red = bottom).

    This is the most powerful single figure for demonstrating robustness:
    a robust strategy shows consistent green across rows even as parameters change.
    """
    all_rows = []
    for test_name, results in all_results.items():
        for r in results:
            row = r.to_dict()
            row["test"] = test_name
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    metrics    = ["cagr_pct","sharpe","sortino","max_dd_pct","calmar","avg_turnover_pct"]
    met_labels = ["CAGR","Sharpe","Sortino","Max DD","Calmar","Avg TO"]

    # Rank-normalise each column (0=worst, 1=best)
    # For max_dd and turnover, lower is better → invert
    scored = df[metrics].copy()
    for m in metrics:
        if m in ("max_dd_pct","avg_turnover_pct"):
            scored[m] = (-scored[m]).rank(pct=True)   # lower is better
        else:
            scored[m] = scored[m].rank(pct=True)

    row_labels = [f"[{r['test'][:4].upper()}] {r['label'][:38]}" for _, r in df.iterrows()]
    mat        = scored.values

    fig, ax = plt.subplots(figsize=(13, max(8, len(row_labels) * 0.38)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(met_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Cell text: actual values
    fmt_map = {
        "cagr_pct": ".1%", "sharpe": ".2f", "sortino": ".2f",
        "max_dd_pct": ".1%", "calmar": ".2f", "avg_turnover_pct": ".0%",
    }
    for i in range(len(row_labels)):
        for j, m in enumerate(metrics):
            val  = df[m].iloc[i]
            text = format(val, fmt_map[m])
            clr  = "white" if mat[i, j] < 0.25 or mat[i, j] > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=clr)

    plt.colorbar(im, ax=ax, label="Percentile rank (green = better)", shrink=0.4)
    ax.set_title(
        "Robustness Heatmap — All Tests\n"
        "(rows = strategy variants; green = top quartile for that metric)",
        fontsize=10, pad=20,
    )
    plt.tight_layout()
    return fig


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_phase6() -> dict:
    """
    Run the complete Phase-6 robustness testing suite.

    Returns
    -------
    dict with all results for notebook inspection.
    """
    log.info("=" * 66)
    log.info("PHASE 6 — ROBUSTNESS TESTING & REGIME ANALYSIS")
    log.info("=" * 66)

    # --- Load data ---
    log.info("Loading data …")
    trad_composite = pd.read_parquet(DATA_DIR / "factor_composite.parquet")
    enh_composite  = pd.read_parquet(DATA_DIR / "factor_enhanced_composite.parquet")
    hiring_norm    = pd.read_parquet(DATA_DIR / "factor_hiring_norm.parquet")
    sentiment_norm = pd.read_parquet(DATA_DIR / "factor_sentiment_norm.parquet")
    returns        = pd.read_parquet(DATA_DIR / "returns_simple.parquet")

    # Benchmark returns (EW universe)
    bm_df      = build_benchmark(returns)
    bm_ret     = bm_df["bm_return"]
    bm_nav     = bm_df["bm_nav"]

    # Base configuration (our Phase 4/5 settings)
    base_cfg = PortfolioConfig(
        top_quantile=0.20,
        transaction_cost_bps=10.0,
        rebalance_freq="BME",
    )

    # ------------------------------------------------------------------
    log.info("\n--- TEST 1: Concentration ---")
    conc_results = test_concentration(
        enh_composite, returns,
        quantiles=[0.10, 0.20, 0.30],
        base_config=base_cfg, benchmark_ret=bm_ret,
    )
    log_results_table(conc_results, "Test 1: Portfolio Concentration")

    # ------------------------------------------------------------------
    log.info("\n--- TEST 2: Rebalance Frequency ---")
    freq_results = test_rebalance_frequency(
        enh_composite, returns,
        base_config=base_cfg, benchmark_ret=bm_ret,
    )
    log_results_table(freq_results, "Test 2: Rebalance Frequency")

    # ------------------------------------------------------------------
    log.info("\n--- TEST 3: Transaction Cost Stress ---")
    cost_results = test_transaction_costs(
        enh_composite, returns,
        cost_bps_list=[0, 5, 10, 20, 50],
        base_config=base_cfg, benchmark_ret=bm_ret,
    )
    log_results_table(cost_results, "Test 3: Transaction Cost Stress")

    # ------------------------------------------------------------------
    log.info("\n--- TEST 4: Factor Weight Robustness ---")
    weight_results = test_factor_weights(
        trad_composite, hiring_norm, sentiment_norm, returns,
        base_config=base_cfg, benchmark_ret=bm_ret,
    )
    log_results_table(weight_results, "Test 4: Factor Weights")

    # ------------------------------------------------------------------
    log.info("\n--- TEST 5: Market Regime Analysis ---")
    # Get NAV series for traditional and enhanced strategies
    from portfolio import build_portfolio
    from backtest  import run_backtest

    trad_port = build_portfolio(trad_composite, base_cfg)
    trad_bt   = run_backtest(trad_port, returns)
    enh_port  = build_portfolio(enh_composite, base_cfg)
    enh_bt    = run_backtest(enh_port, returns)

    trad_nav  = trad_bt["nav_df"]["nav_net"]
    enh_nav   = enh_bt["nav_df"]["nav_net"]
    trad_ret  = trad_bt["nav_df"]["net_return"]
    enh_ret   = enh_bt["nav_df"]["net_return"]

    regime_results = run_regime_analysis(
        strategy_ret_trad=trad_ret,
        strategy_ret_enh=enh_ret,
        benchmark_ret=bm_ret,
        market_returns=bm_ret,          # use EW universe as market proxy
    )

    # ------------------------------------------------------------------
    log.info("\n--- TEST 6: Factor Ablation ---")
    ablation_results = test_ablation(
        trad_composite, hiring_norm, sentiment_norm, returns,
        base_config=base_cfg, benchmark_ret=bm_ret,
    )
    log_results_table(ablation_results, "Test 6: Factor Ablation")

    # ------------------------------------------------------------------
    log.info("\nGenerating figures …")
    all_results = {
        "conc":    conc_results,
        "freq":    freq_results,
        "cost":    cost_results,
        "weights": weight_results,
        "ablat":   ablation_results,
    }

    figs = {
        "phase6_concentration":   plot_concentration(conc_results),
        "phase6_frequency":       plot_frequency(freq_results),
        "phase6_cost_stress":     plot_cost_stress(cost_results),
        "phase6_weights":         plot_weights(weight_results),
        "phase6_regime_bars":     plot_regime_bars(regime_results),
        "phase6_regime_nav":      plot_regime_nav(regime_results, trad_nav, enh_nav, bm_nav),
        "phase6_ablation":        plot_ablation(ablation_results),
        "phase6_summary_heatmap": plot_summary_heatmap(all_results),
    }
    save_figures(figs, FIGURE_DIR)

    # ------------------------------------------------------------------
    log.info("Saving data outputs …")
    DATA_DIR.mkdir(exist_ok=True)

    # Summary JSON
    summary = {}
    for test_name, results in all_results.items():
        summary[test_name] = [r.to_dict() for r in results]

    (DATA_DIR / "phase6_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Regime JSON
    regime_json = {}
    for key in ("stats_traditional", "stats_enhanced", "stats_benchmark"):
        regime_json[key] = regime_stats_to_df(regime_results[key]).to_dict(orient="records")
    (DATA_DIR / "phase6_regime_analysis.json").write_text(
        json.dumps(regime_json, indent=2, default=str), encoding="utf-8"
    )

    log.info("=" * 66)
    log.info("PHASE 6 COMPLETE")
    log.info(f"  Data    → {DATA_DIR}")
    log.info(f"  Figures → {FIGURE_DIR}")
    log.info("=" * 66)

    # Terminal summary
    print("\n" + "=" * 66)
    print("PHASE 6 ROBUSTNESS SUMMARY")
    print("=" * 66)
    for test_name, results in all_results.items():
        print(f"\n  [{test_name.upper()}]")
        print(f"  {'Label':<38} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8}")
        print(f"  {'-'*62}")
        for r in results:
            print(f"  {r.label:<38} {r.cagr_pct:>6.2%}  {r.sharpe:>6.3f}  {r.max_dd_pct:>7.2%}")
    print("=" * 66)

    return {
        "conc_results":    conc_results,
        "freq_results":    freq_results,
        "cost_results":    cost_results,
        "weight_results":  weight_results,
        "ablation_results":ablation_results,
        "regime_results":  regime_results,
        "trad_nav":        trad_nav,
        "enh_nav":         enh_nav,
        "bm_nav":          bm_nav,
    }


if __name__ == "__main__":
    results = run_phase6()
