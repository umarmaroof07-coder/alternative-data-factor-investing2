"""
phase5_alt_factors.py
---------------------
Phase 5 Orchestrator — Alternative Data Alpha Layer

This module:
  1. Builds hiring-trend and news-sentiment alt-data factors
  2. Constructs the enhanced composite:
       40% traditional factor composite (Phase 3)
       30% hiring trends score
       30% news sentiment score
  3. Runs IDENTICAL Phase-4 backtest on enhanced composite
  4. Produces a head-to-head comparison:
       - Traditional strategy (Phase 4 baseline)
       - Alt-data enhanced strategy (Phase 5)
       - Equal-weighted benchmark
  5. Saves all outputs to data/ and figures/

WHY 40/30/30 WEIGHTING?
------------------------
  • Traditional composite (40%): battle-tested momentum + low-vol signals
    with 30+ years of academic support.  Gets the largest weight.
  • Hiring (30%): strong economic rationale, monthly cadence, lower
    noise than daily signals.  Equal weight with sentiment.
  • Sentiment (30%): complementary to momentum (captures short-term
    attention effects); daily cadence gives fresher signal.

  Why not 33/33/33?
  We bias toward the traditional composite (40%) because:
  (a) It has longer out-of-sample track record in the literature
  (b) Alt data signals are more regime-dependent
  (c) This mirrors how most quant allocators incorporate alt data:
      as a *supplement* to, not a *replacement* for, traditional factors.

COMPARISON METHODOLOGY
----------------------
Both strategies use IDENTICAL:
  • Universe (S&P 100)
  • Rebalancing schedule (month-end)
  • Position count (top 20%)
  • Weighting scheme (equal weight)
  • Transaction costs (10bps one-way)

The only difference is the composite score that drives stock selection.
This is the cleanest possible controlled experiment for evaluating whether
alt data adds incremental value.

Author  : Quant Research Team
Phase   : 5 — Alternative Data Alpha Layer
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment_factor import (
    build_hiring_factor,
    build_sentiment_factor,
    build_alt_composite,
    alt_factor_diagnostics,
)
from factor_utils     import normalise_factor, cross_sectional_zscore
from portfolio        import build_portfolio, PortfolioConfig
from backtest         import run_backtest, build_benchmark
from performance      import (
    cagr, annualised_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, information_ratio,
    drawdown_series, save_figures,
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
# Colour palette (extend Phase 4 palette)
# ---------------------------------------------------------------------------
COLORS = {
    "traditional":  "#2166AC",    # blue  — Phase-4 traditional strategy
    "enhanced":     "#4DAC26",    # green — Phase-5 enhanced strategy
    "benchmark":    "#D6604D",    # red   — equal-weight benchmark
    "hiring":       "#8073AC",    # purple
    "sentiment":    "#E08214",    # orange
    "neutral":      "#636363",
}


# ===========================================================================
# Step 1 — Build the enhanced composite score
# ===========================================================================

def build_enhanced_composite(
    trad_composite: pd.DataFrame,
    daily_returns:  pd.DataFrame,
    tickers:        list[str],
    start_date:     str | pd.Timestamp,
    end_date:       str | pd.Timestamp,
    w_traditional:  float = 0.40,
    w_hiring:       float = 0.30,
    w_sentiment:    float = 0.30,
    hiring_seed:    int   = 101,
    sentiment_seed: int   = 202,
) -> dict:
    """
    Build the full enhanced composite and the constituent alt-data factors.

    COMBINATION FORMULA
    -------------------
    For each stock i at daily date t:

      enhanced(i,t) = w_trad × trad_z(i,t)
                    + w_hire × hiring_z(i,t)
                    + w_sent × sentiment_z(i,t)

    All three inputs are normalised to ≈ N(0,1) before combination,
    so the weights are truly interpretable as "fraction of statistical
    influence" rather than "fraction of raw signal magnitude."

    After combination, the composite is re-z-scored cross-sectionally
    so that portfolio construction in Phase 4 sees a consistent signal.

    Parameters
    ----------
    trad_composite  : T_daily × N traditional factor z-scores (from Phase 3)
    daily_returns   : T_daily × N simple daily returns (from Phase 2)
    tickers         : universe ticker list
    start_date/end  : date range
    w_*             : combination weights (automatically normalised to sum=1)

    Returns
    -------
    dict with:
        'enhanced_composite' : T_daily × N enhanced z-scores
        'hiring_norm'        : T_daily × N hiring z-scores
        'sentiment_norm'     : T_daily × N sentiment z-scores
        'alt_composite'      : T_daily × N combined alt-only z-scores
        'alt_diagnostics'    : dict of factor quality metrics
    """
    log.info("=" * 62)
    log.info("BUILDING ENHANCED COMPOSITE")
    log.info("=" * 62)

    # --- Hiring factor ---
    hiring_out = build_hiring_factor(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        daily_returns=daily_returns,
        seed=hiring_seed,
        alpha_corr=0.09,
    )
    hiring_norm = hiring_out["normalised"]

    # --- Sentiment factor ---
    sent_out = build_sentiment_factor(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        daily_returns=daily_returns,
        seed=sentiment_seed,
        alpha_corr=0.07,
    )
    sentiment_norm = sent_out["normalised"]

    # --- Alt composite (hiring + sentiment) ---
    alt_composite = build_alt_composite(
        hiring_norm=hiring_norm,
        sentiment_norm=sentiment_norm,
    )

    # --- Diagnostics ---
    log.info("Computing alt-data factor diagnostics …")
    diag = alt_factor_diagnostics(
        factors={
            "hiring_trends": hiring_norm,
            "news_sentiment": sentiment_norm,
            "alt_composite": alt_composite,
        },
        trad_composite=trad_composite,
    )

    # --- Final enhanced composite ---
    # Normalise weights
    total_w = w_traditional + w_hiring + w_sentiment
    wt = w_traditional / total_w
    wh = w_hiring      / total_w
    ws = w_sentiment   / total_w

    log.info(
        f"Blending composites | "
        f"traditional={wt:.0%}, hiring={wh:.0%}, sentiment={ws:.0%}"
    )

    # Align all three on common dates and tickers
    common_dates   = (
        trad_composite.index
        .intersection(hiring_norm.index)
        .intersection(sentiment_norm.index)
    )
    common_tickers = (
        trad_composite.columns
        .intersection(hiring_norm.columns)
        .intersection(sentiment_norm.columns)
    )

    trad = trad_composite.loc[common_dates, common_tickers]
    hire = hiring_norm.loc[common_dates, common_tickers]
    sent = sentiment_norm.loc[common_dates, common_tickers]

    # Weighted combination — stock only gets a score if all three are valid
    enhanced = wt * trad + wh * hire + ws * sent
    all_valid = trad.notna() & hire.notna() & sent.notna()

    # For stocks missing alt data, fall back to traditional composite only
    # (important for real-world robustness — alt data often has gaps)
    trad_only = trad_composite.loc[common_dates, common_tickers]
    enhanced  = enhanced.where(all_valid, other=trad_only)

    # Re-z-score cross-sectionally
    enhanced = enhanced.apply(cross_sectional_zscore, axis=1)

    log.info(
        f"Enhanced composite built | shape: {enhanced.shape} | "
        f"NaN rate: {enhanced.isna().mean().mean():.1%}"
    )

    return {
        "enhanced_composite": enhanced,
        "hiring_norm":        hiring_norm,
        "sentiment_norm":     sentiment_norm,
        "alt_composite":      alt_composite,
        "alt_diagnostics":    diag,
        "weights":            {"traditional": wt, "hiring": wh, "sentiment": ws},
    }


# ===========================================================================
# Step 2 — Run dual backtest and collect results
# ===========================================================================

def run_dual_backtest(
    trad_composite:    pd.DataFrame,
    enhanced_composite: pd.DataFrame,
    daily_returns:     pd.DataFrame,
    config:            PortfolioConfig | None = None,
) -> dict:
    """
    Run IDENTICAL backtests on traditional and enhanced composites.

    Everything is held equal except the composite score driving selection.
    This isolation makes the comparison causally valid: any performance
    difference is attributable solely to the alternative data signals.

    Parameters
    ----------
    trad_composite     : Phase-3 composite (T × N)
    enhanced_composite : Phase-5 enhanced composite (T × N)
    daily_returns      : Phase-2 simple returns (T × N)
    config             : PortfolioConfig (shared between both strategies)

    Returns
    -------
    dict with:
        'traditional' : backtest result dict (from backtest.run_backtest)
        'enhanced'    : backtest result dict
        'benchmark'   : equal-weight benchmark DataFrame
        'config'      : PortfolioConfig used
    """
    if config is None:
        config = PortfolioConfig(
            top_quantile=0.20,
            transaction_cost_bps=10.0,
            rebalance_freq="BME",
        )

    log.info("Running dual backtest …")

    # --- Traditional strategy ---
    log.info("  [1/2] Traditional composite …")
    trad_port    = build_portfolio(trad_composite, config)
    trad_results = run_backtest(trad_port, daily_returns)

    # --- Enhanced strategy ---
    log.info("  [2/2] Enhanced composite …")
    enh_port    = build_portfolio(enhanced_composite, config)
    enh_results = run_backtest(enh_port, daily_returns)

    # --- Benchmark (shared) ---
    benchmark = build_benchmark(daily_returns)

    # Align benchmark to common dates
    common = (
        trad_results["nav_df"].index
        .intersection(enh_results["nav_df"].index)
        .intersection(benchmark.index)
    )
    benchmark = benchmark.loc[common]

    log.info("Dual backtest complete.")

    return {
        "traditional": trad_results,
        "enhanced":    enh_results,
        "benchmark":   benchmark,
        "config":      config,
    }


# ===========================================================================
# Step 3 — Compute comparison performance table
# ===========================================================================

def build_comparison_table(
    dual_results: dict,
) -> dict:
    """
    Side-by-side performance metrics for traditional, enhanced, and benchmark.

    All three are evaluated on the SAME date range (intersection) so no
    strategy gets an unfair advantage from a different start/end date.
    """
    trad_nav  = dual_results["traditional"]["nav_df"]
    enh_nav   = dual_results["enhanced"]["nav_df"]
    benchmark = dual_results["benchmark"]

    # Common date range
    common = (
        trad_nav.index
        .intersection(enh_nav.index)
        .intersection(benchmark.index)
    )
    trad_nav  = trad_nav.loc[common]
    enh_nav   = enh_nav.loc[common]
    bm        = benchmark.loc[common]

    def stats(nav_col, ret_col, label):
        return {
            "label":        label,
            "cagr":         round(cagr(nav_col),                    4),
            "ann_vol":      round(annualised_volatility(ret_col),    4),
            "sharpe":       round(sharpe_ratio(ret_col),             4),
            "sortino":      round(sortino_ratio(ret_col),            4),
            "max_drawdown": round(max_drawdown(nav_col),             4),
            "calmar":       round(calmar_ratio(nav_col, ret_col),    4),
        }

    trad_s = stats(trad_nav["nav_net"], trad_nav["net_return"],  "Traditional (Net)")
    enh_s  = stats(enh_nav["nav_net"],  enh_nav["net_return"],   "Enhanced (Net)")
    bm_s   = stats(bm["bm_nav"],        bm["bm_return"],          "EW Benchmark")

    # Relative metrics
    def rel(strat_ret, bm_ret, label):
        ir = information_ratio(strat_ret, bm_ret)
        return {
            "label":                label,
            "alpha_vs_benchmark":   round(cagr(strat_ret.add(1).cumprod()) - bm_s["cagr"], 4)
                                    if False else round(
                                        (strat_ret.mean() - bm_ret.mean()) * 252, 4),
            "information_ratio":    round(ir, 4) if not np.isnan(ir) else None,
            "tracking_error":       round(
                (strat_ret - bm_ret).std() * np.sqrt(252), 4),
        }

    trad_r = rel(trad_nav["net_return"], bm["bm_return"], "Traditional vs BM")
    enh_r  = rel(enh_nav["net_return"],  bm["bm_return"], "Enhanced vs BM")

    # Alt data incremental
    incremental_sharpe = enh_s["sharpe"] - trad_s["sharpe"]
    incremental_cagr   = enh_s["cagr"]   - trad_s["cagr"]

    table = {
        "traditional":  trad_s,
        "enhanced":     enh_s,
        "benchmark":    bm_s,
        "trad_relative":  trad_r,
        "enh_relative":   enh_r,
        "alt_data_impact": {
            "incremental_cagr":   round(incremental_cagr,   4),
            "incremental_sharpe": round(incremental_sharpe, 4),
            "incremental_calmar": round(enh_s["calmar"] - trad_s["calmar"], 4),
        },
    }

    # Log it
    log.info("=" * 62)
    log.info("PHASE 5 COMPARISON SUMMARY")
    log.info("=" * 62)
    metrics = ["cagr","ann_vol","sharpe","sortino","max_drawdown","calmar"]
    labels  = ["CAGR","Ann Vol","Sharpe","Sortino","Max DD","Calmar"]
    log.info(f"  {'Metric':<20} {'Traditional':>14} {'Enhanced':>14} {'Benchmark':>14}")
    log.info(f"  {'-'*62}")
    for m, lbl in zip(metrics, labels):
        tv = table["traditional"][m]
        ev = table["enhanced"][m]
        bv = table["benchmark"][m]
        fmt = ".2%" if m in ("cagr","ann_vol","max_drawdown") else ".3f"
        log.info(f"  {lbl:<20} {tv:{fmt}}{'':<8} {ev:{fmt}}{'':<8} {bv:{fmt}}")
    log.info(f"  {'-'*62}")
    ic = table["alt_data_impact"]
    log.info(f"  Alt-data Δ CAGR   : {ic['incremental_cagr']:>+.2%}")
    log.info(f"  Alt-data Δ Sharpe : {ic['incremental_sharpe']:>+.4f}")
    log.info("=" * 62)

    return table


# ===========================================================================
# Step 4 — Visualisations
# ===========================================================================

def plot_three_way_nav(dual_results: dict) -> plt.Figure:
    """
    Three-way NAV comparison: traditional, enhanced, benchmark.
    Bottom panel: drawdown comparison.

    The shaded band between enhanced and traditional shows the incremental
    contribution of alternative data — positive when alt data helps,
    negative when it hurts.
    """
    trad = dual_results["traditional"]["nav_df"]
    enh  = dual_results["enhanced"]["nav_df"]
    bm   = dual_results["benchmark"]

    common = trad.index.intersection(enh.index).intersection(bm.index)
    trad_nav = trad.loc[common, "nav_net"]
    enh_nav  = enh.loc[common,  "nav_net"]
    bm_nav   = bm.loc[common,   "bm_nav"]

    dd_trad = drawdown_series(trad_nav)
    dd_enh  = drawdown_series(enh_nav)
    dd_bm   = drawdown_series(bm_nav)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    fig.suptitle(
        "Phase 5: Multi-Factor Strategy vs Alt-Data Enhanced Strategy",
        fontsize=12, fontweight="bold",
    )

    # — NAV curves —
    ax1.plot(common, trad_nav, color=COLORS["traditional"], lw=1.4,
             linestyle="--", label="Traditional (Phase 4)")
    ax1.plot(common, enh_nav,  color=COLORS["enhanced"],    lw=1.8,
             label="Alt-Data Enhanced (Phase 5)")
    ax1.plot(common, bm_nav,   color=COLORS["benchmark"],   lw=1.1,
             alpha=0.75, label="EW Benchmark")

    # Shade alt-data contribution band
    alt_ahead = enh_nav > trad_nav
    ax1.fill_between(common, enh_nav, trad_nav,
                     where=alt_ahead,  alpha=0.13, color=COLORS["enhanced"],
                     label="Alt-data advantage")
    ax1.fill_between(common, enh_nav, trad_nav,
                     where=~alt_ahead, alpha=0.13, color=COLORS["traditional"],
                     label="Alt-data drag")

    ax1.set_ylabel("NAV (base = 100)")
    ax1.legend(loc="upper left", ncol=2, fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # — Drawdowns —
    ax2.fill_between(common, dd_trad, 0,
                     color=COLORS["traditional"], alpha=0.45, label="Traditional DD")
    ax2.fill_between(common, dd_enh, 0,
                     color=COLORS["enhanced"],    alpha=0.45, label="Enhanced DD")
    ax2.plot(common, dd_bm, color=COLORS["benchmark"], lw=0.8,
             linestyle=":", label="Benchmark DD")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(loc="lower left", ncol=3, fontsize=7)

    plt.tight_layout()
    return fig


def plot_alt_factor_signals(
    hiring_norm:    pd.DataFrame,
    sentiment_norm: pd.DataFrame,
    tickers_to_show: list[str] | None = None,
) -> plt.Figure:
    """
    Time-series of alt-data factor scores for a sample of tickers.

    Shows the cross-sectional variation in alt-data signals over time,
    demonstrating that the factors have genuine differentiation across
    the universe (not just market-level noise).
    """
    if tickers_to_show is None:
        # Pick 6 tickers with the widest average score dispersion
        tickers_to_show = (
            hiring_norm.std()
            .nlargest(6).index.tolist()
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "Alternative Data Factor Scores — Sample Tickers",
        fontsize=11,
    )

    cmap  = plt.cm.tab10
    color_cycle = [cmap(i) for i in range(len(tickers_to_show))]

    for ax, (factor_df, title) in zip(
        axes,
        [(hiring_norm,    "Hiring Trends Factor (z-score)"),
         (sentiment_norm, "News Sentiment Factor (z-score)")],
    ):
        for i, ticker in enumerate(tickers_to_show):
            if ticker not in factor_df.columns:
                continue
            series = factor_df[ticker].rolling(21).mean()  # smooth for readability
            ax.plot(series.index, series, lw=1.0,
                    color=color_cycle[i], label=ticker, alpha=0.85)

        ax.axhline(0, color="#888", lw=0.7, linestyle=":")
        ax.set_ylabel("z-score")
        ax.set_title(title, loc="left", fontsize=9)
        ax.legend(ncol=len(tickers_to_show), fontsize=7, loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


def plot_composite_composition(
    trad_composite:    pd.DataFrame,
    enhanced_composite: pd.DataFrame,
    hiring_norm:       pd.DataFrame,
    sentiment_norm:    pd.DataFrame,
    weights:           dict,
) -> plt.Figure:
    """
    Four-panel: scatter plots of each input vs the enhanced composite.

    Shows how much each component contributes and how they diversify.
    A correlation near +1 means the two factors carry the same information.
    A correlation near 0 means they are complementary — the ideal case.
    """
    common_dates   = (
        trad_composite.index
        .intersection(hiring_norm.index)
        .intersection(sentiment_norm.index)
        .intersection(enhanced_composite.index)
    )
    common_tickers = (
        trad_composite.columns
        .intersection(hiring_norm.columns)
        .intersection(sentiment_norm.columns)
        .intersection(enhanced_composite.columns)
    )

    trad = trad_composite.loc[common_dates, common_tickers].stack().dropna()
    hire = hiring_norm.loc[common_dates, common_tickers].stack().dropna()
    sent = sentiment_norm.loc[common_dates, common_tickers].stack().dropna()
    enh  = enhanced_composite.loc[common_dates, common_tickers].stack().dropna()

    # Sample 1500 obs for legibility
    rng = np.random.default_rng(0)
    idx = rng.choice(len(enh), min(1500, len(enh)), replace=False)
    enh_s = enh.iloc[idx]

    from scipy.stats import spearmanr

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Enhanced Composite Decomposition\n"
        "(cross-sectional z-scores; sample of 1,500 obs)",
        fontsize=11,
    )

    pairs = [
        (trad, "Traditional Composite", COLORS["traditional"],
         f"w = {weights['traditional']:.0%}"),
        (hire, "Hiring Trends",         COLORS["hiring"],
         f"w = {weights['hiring']:.0%}"),
        (sent, "News Sentiment",        COLORS["sentiment"],
         f"w = {weights['sentiment']:.0%}"),
    ]

    for ax, (factor, label, color, weight_lbl) in zip(axes, pairs):
        common_obs = factor.index.intersection(enh_s.index)
        if len(common_obs) < 10:
            continue
        x = factor.loc[common_obs]
        y = enh_s.loc[common_obs]
        rho, _ = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.25, s=8, color=color)
        ax.set_xlabel(f"{label} z-score")
        ax.set_ylabel("Enhanced Composite z-score")
        ax.set_title(
            f"{label}\n{weight_lbl} | ρ = {rho:.3f}",
            fontsize=9,
        )
        # Fit line
        z = np.polyfit(x, y, 1)
        x_range = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_range, np.polyval(z, x_range), "k--", lw=1.0)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)

    plt.tight_layout()
    return fig


def plot_rolling_ir_comparison(dual_results: dict) -> plt.Figure:
    """
    Rolling 6-month information ratio: traditional vs enhanced.

    The IR is the primary metric for evaluating incremental factor value.
    A consistently higher IR for the enhanced strategy (even by a small
    margin) indicates that alt data adds genuine value beyond noise.
    """
    trad = dual_results["traditional"]["nav_df"]
    enh  = dual_results["enhanced"]["nav_df"]
    bm   = dual_results["benchmark"]

    common = trad.index.intersection(enh.index).intersection(bm.index)
    bm_ret = bm.loc[common, "bm_return"]
    t_ret  = trad.loc[common, "net_return"]
    e_ret  = enh.loc[common,  "net_return"]

    window = 126   # 6 months

    def rolling_ir(strat, bench, w):
        active = strat - bench
        ir = active.rolling(w).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252)
            if x.std() > 1e-8 else 0,
            raw=True,
        )
        return ir

    t_ir = rolling_ir(t_ret, bm_ret, window)
    e_ir = rolling_ir(e_ret, bm_ret, window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Rolling 6-Month Information Ratio\n"
        "Traditional vs Alt-Data Enhanced Strategy",
        fontsize=11,
    )

    for ax, (ir, label, color) in zip(
        [ax1, ax2],
        [(t_ir, "Traditional",         COLORS["traditional"]),
         (e_ir, "Enhanced (Alt-Data)", COLORS["enhanced"])],
    ):
        pos = ir > 0
        ax.fill_between(ir.index, ir, 0,
                        where=pos,  color=color, alpha=0.55)
        ax.fill_between(ir.index, ir, 0,
                        where=~pos, color=COLORS["benchmark"], alpha=0.40)
        ax.plot(ir.index, ir, color=color, lw=1.1, label=label)
        ax.axhline(0,   color="#555", lw=0.8, linestyle=":")
        ax.axhline(0.5, color=color,  lw=0.7, linestyle="--", alpha=0.5,
                   label="IR = 0.5 (good)")
        ax.set_ylabel("Information Ratio")
        ax.set_title(label, loc="left", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(-3, 3)

    ax2.set_xlabel("Date")
    plt.tight_layout()
    return fig


def plot_factor_correlation_matrix(
    trad_composite:    pd.DataFrame,
    hiring_norm:       pd.DataFrame,
    sentiment_norm:    pd.DataFrame,
    enhanced_composite: pd.DataFrame,
) -> plt.Figure:
    """
    Spearman correlation heatmap between all four factor layers.

    LOW inter-factor correlation is desirable: it means each factor
    contributes independent information.  High correlation means
    they're measuring the same thing and the combination adds noise
    but no new signal.

    Institutional rule of thumb: factors with ρ > 0.7 between them
    should not both be included without careful analysis.
    """
    from scipy.stats import spearmanr

    names = ["Traditional", "Hiring", "Sentiment", "Enhanced"]
    dfs   = [trad_composite, hiring_norm, sentiment_norm, enhanced_composite]

    # Flatten to vectors
    common_dates   = dfs[0].index
    for d in dfs[1:]:
        common_dates = common_dates.intersection(d.index)
    common_tickers = dfs[0].columns
    for d in dfs[1:]:
        common_tickers = common_tickers.intersection(d.columns)

    vectors = []
    for df in dfs:
        v = df.loc[common_dates, common_tickers].stack().dropna()
        vectors.append(v)

    # Sample for speed
    rng   = np.random.default_rng(1)
    n_obs = min(len(v) for v in vectors)
    idx   = rng.choice(n_obs, min(5000, n_obs), replace=False)

    corr_matrix = np.eye(4)
    for i in range(4):
        for j in range(i+1, 4):
            vi = vectors[i].iloc[idx]
            vj = vectors[j].iloc[idx]
            common_idx = vi.index.intersection(vj.index)
            if len(common_idx) > 50:
                rho, _ = spearmanr(vi.loc[common_idx], vj.loc[common_idx])
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=9)

    for i in range(4):
        for j in range(4):
            val = corr_matrix[i, j]
            txt_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")

    ax.set_title(
        "Factor Cross-Correlation Matrix\n"
        "(Spearman ρ — low values = diversification value)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


# ===========================================================================
# Step 5 — Save all Phase 5 outputs
# ===========================================================================

def save_phase5_outputs(
    enhanced_composite: pd.DataFrame,
    hiring_norm:        pd.DataFrame,
    sentiment_norm:     pd.DataFrame,
    alt_composite:      pd.DataFrame,
    comparison_table:   dict,
    alt_diagnostics:    dict,
    dual_results:       dict,
    data_dir:           Path = DATA_DIR,
) -> None:
    """Persist all Phase-5 DataFrames and metadata."""
    data_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "factor_enhanced_composite": enhanced_composite,
        "factor_hiring_norm":        hiring_norm,
        "factor_sentiment_norm":     sentiment_norm,
        "factor_alt_composite":      alt_composite,
        "backtest_nav_enhanced":     dual_results["enhanced"]["nav_df"],
        "backtest_nav_traditional":  dual_results["traditional"]["nav_df"],
    }

    for name, df in file_map.items():
        path = data_dir / f"{name}.parquet"
        df.to_parquet(path)
        log.info(f"  Saved {path.name}")

    (data_dir / "phase5_comparison_table.json").write_text(
        json.dumps(comparison_table, indent=2, default=str), encoding="utf-8"
    )
    (data_dir / "phase5_alt_diagnostics.json").write_text(
        json.dumps(alt_diagnostics, indent=2, default=str), encoding="utf-8"
    )
    log.info(f"Phase 5 data saved to {data_dir}")


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_phase5(
    w_traditional: float = 0.40,
    w_hiring:      float = 0.30,
    w_sentiment:   float = 0.30,
) -> dict:
    """
    Run the complete Phase-5 pipeline end-to-end.

    Returns
    -------
    dict with all intermediate results for notebook inspection.
    """
    log.info("=" * 62)
    log.info("PHASE 5 — ALTERNATIVE DATA ALPHA LAYER")
    log.info("=" * 62)

    # --- Load Phase-2 and Phase-3 data ---
    log.info("Loading Phase-2 / Phase-3 data …")
    trad_composite = pd.read_parquet(DATA_DIR / "factor_composite.parquet")
    daily_returns  = pd.read_parquet(DATA_DIR / "returns_simple.parquet")

    tickers    = list(trad_composite.columns)
    start_date = daily_returns.index[0]
    end_date   = daily_returns.index[-1]

    log.info(
        f"  Tickers: {len(tickers)} | "
        f"Date range: {start_date.date()} → {end_date.date()}"
    )

    # --- Build enhanced composite ---
    alt_out = build_enhanced_composite(
        trad_composite=trad_composite,
        daily_returns=daily_returns,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        w_traditional=w_traditional,
        w_hiring=w_hiring,
        w_sentiment=w_sentiment,
    )
    enhanced_composite = alt_out["enhanced_composite"]
    hiring_norm        = alt_out["hiring_norm"]
    sentiment_norm     = alt_out["sentiment_norm"]
    alt_composite      = alt_out["alt_composite"]
    weights            = alt_out["weights"]

    # --- Dual backtest ---
    dual_results = run_dual_backtest(
        trad_composite=trad_composite,
        enhanced_composite=enhanced_composite,
        daily_returns=daily_returns,
    )

    # --- Comparison table ---
    comparison_table = build_comparison_table(dual_results)

    # --- Figures ---
    log.info("Generating Phase-5 figures …")
    figs = {
        "phase5_three_way_nav":       plot_three_way_nav(dual_results),
        "phase5_alt_signals":         plot_alt_factor_signals(hiring_norm, sentiment_norm),
        "phase5_composite_decomp":    plot_composite_composition(
            trad_composite, enhanced_composite, hiring_norm, sentiment_norm, weights
        ),
        "phase5_rolling_ir":          plot_rolling_ir_comparison(dual_results),
        "phase5_correlation_matrix":  plot_factor_correlation_matrix(
            trad_composite, hiring_norm, sentiment_norm, enhanced_composite
        ),
    }
    save_figures(figs, FIGURE_DIR)

    # --- Save data ---
    save_phase5_outputs(
        enhanced_composite=enhanced_composite,
        hiring_norm=hiring_norm,
        sentiment_norm=sentiment_norm,
        alt_composite=alt_composite,
        comparison_table=comparison_table,
        alt_diagnostics=alt_out["alt_diagnostics"],
        dual_results=dual_results,
    )

    # --- Terminal summary ---
    tbl = comparison_table
    ic  = tbl["alt_data_impact"]
    print("\n" + "=" * 60)
    print(f"{'Metric':<22} {'Traditional':>12} {'Enhanced':>12} {'Benchmark':>12}")
    print("-" * 60)
    for metric, lbl in [("cagr","CAGR"),("ann_vol","Ann Vol"),
                         ("sharpe","Sharpe"),("max_drawdown","Max DD"),
                         ("calmar","Calmar")]:
        tv = tbl["traditional"][metric]
        ev = tbl["enhanced"][metric]
        bv = tbl["benchmark"][metric]
        fmt = ".2%" if metric in ("cagr","ann_vol","max_drawdown") else ".3f"
        print(f"  {lbl:<20} {tv:{fmt}}{'':<6} {ev:{fmt}}{'':<6} {bv:{fmt}}")
    print("-" * 60)
    print(f"  {'Alt-data Δ CAGR':<20} {ic['incremental_cagr']:>+.2%}")
    print(f"  {'Alt-data Δ Sharpe':<20} {ic['incremental_sharpe']:>+.4f}")
    print(f"  {'Alt-data Δ Calmar':<20} {ic['incremental_calmar']:>+.4f}")
    print("=" * 60)

    log.info("=" * 62)
    log.info("PHASE 5 COMPLETE")
    log.info(f"  Data    → {DATA_DIR}")
    log.info(f"  Figures → {FIGURE_DIR}")
    log.info("=" * 62)

    return {
        "trad_composite":    trad_composite,
        "enhanced_composite": enhanced_composite,
        "hiring_norm":       hiring_norm,
        "sentiment_norm":    sentiment_norm,
        "alt_composite":     alt_composite,
        "weights":           weights,
        "dual_results":      dual_results,
        "comparison_table":  comparison_table,
        "alt_diagnostics":   alt_out["alt_diagnostics"],
    }


if __name__ == "__main__":
    results = run_phase5()
