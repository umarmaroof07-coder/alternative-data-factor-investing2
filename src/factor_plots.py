"""
factor_plots.py
---------------
Publication-quality diagnostic plots for Phase 3 factor panel.

All figures follow institutional research conventions:
  - Neutral grey palette (avoids colour-blind issues in printed PDFs)
  - All axes labelled with units
  - No chartjunk (Tufte-style)
  - Figures saved as PNG at 150 dpi (screen) and PDF (paper-ready)

Author : Quant Research Team
Phase  : 3 — Factor Construction
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F7F7F7",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
})

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "momentum":  "#2166AC",
    "low_vol":   "#D6604D",
    "composite": "#4DAC26",
    "neutral":   "#636363",
}


# ---------------------------------------------------------------------------
# Figure 1 — Factor score distributions (cross-section snapshots)
# ---------------------------------------------------------------------------

def plot_score_distributions(
    factors: dict[str, pd.DataFrame],
    n_snapshots: int = 3,
) -> plt.Figure:
    """
    Histogram of cross-sectional z-scores at evenly spaced dates.

    What to look for:
    - Should be approximately N(0,1) after normalisation
    - Bimodal → two distinct regimes in the cross-section
    - Very fat tails → winsorisation percentiles may need tightening
    """
    factor_names = ["momentum", "low_vol", "composite"]
    fig, axes = plt.subplots(
        n_snapshots, len(factor_names),
        figsize=(12, 3 * n_snapshots),
        sharey=False,
    )

    fig.suptitle(
        "Factor Score Cross-Sectional Distributions\n"
        "(each panel = one date's cross-section of z-scores)",
        fontsize=11, y=1.01,
    )

    # Pick dates evenly spaced from the valid period
    ref_factor = factors["composite"].dropna(how="all")
    snapshot_dates = ref_factor.index[
        np.linspace(0, len(ref_factor) - 1, n_snapshots, dtype=int)
    ]

    x_grid = np.linspace(-3.5, 3.5, 200)
    norm_y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid**2)

    for col_idx, fname in enumerate(factor_names):
        df = factors[fname]
        for row_idx, date in enumerate(snapshot_dates):
            ax = axes[row_idx, col_idx]
            scores = df.loc[date].dropna()

            ax.hist(
                scores, bins=20, density=True,
                color=COLORS[fname], alpha=0.7, edgecolor="white", linewidth=0.5,
            )
            ax.plot(x_grid, norm_y, "k--", linewidth=1.0, label="N(0,1)")
            ax.set_xlim(-4, 4)
            ax.set_title(f"{fname.replace('_',' ').title()}\n{date.date()}", pad=4)
            ax.set_xlabel("z-score")
            if col_idx == 0:
                ax.set_ylabel("Density")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Factor scores over time (top vs bottom quintile)
# ---------------------------------------------------------------------------

def plot_factor_quintile_spread(
    factors: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Time-series of top-quintile vs bottom-quintile mean factor score.

    A well-constructed factor should show:
    - Stable spread between Q1 and Q5 over time
    - Neither quintile drifting to extreme values (data quality check)
    - Consistent coverage (not dropping to zero in certain periods)
    """
    factor_names = ["momentum", "low_vol", "composite"]
    fig, axes = plt.subplots(len(factor_names), 1, figsize=(13, 9), sharex=True)

    fig.suptitle(
        "Factor Scores: Top vs Bottom Quintile Mean (21-day rolling)",
        fontsize=11,
    )

    for ax, fname in zip(axes, factor_names):
        df = factors[fname].dropna(how="all")

        # Compute quintile means for each date
        q1_means, q5_means = [], []
        for date, row in df.iterrows():
            row = row.dropna()
            if len(row) < 10:
                q1_means.append(np.nan)
                q5_means.append(np.nan)
                continue
            q1_means.append(row[row >= row.quantile(0.80)].mean())  # top 20%
            q5_means.append(row[row <= row.quantile(0.20)].mean())  # bottom 20%

        dates  = df.index
        q1_s   = pd.Series(q1_means, index=dates).rolling(21).mean()
        q5_s   = pd.Series(q5_means, index=dates).rolling(21).mean()
        spread = q1_s - q5_s

        ax.plot(dates, q1_s,   color=COLORS[fname], linewidth=1.2, label="Top 20%")
        ax.plot(dates, q5_s,   color=COLORS[fname], linewidth=1.2,
                linestyle="--", alpha=0.7, label="Bottom 20%")
        ax.fill_between(dates, q1_s, q5_s, alpha=0.12, color=COLORS[fname])
        ax.axhline(0, color="#666666", linewidth=0.7, linestyle=":")

        ax.set_ylabel("Mean z-score")
        ax.set_title(f"{fname.replace('_',' ').title()}", loc="left", pad=3)
        ax.legend(loc="upper right", ncol=2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Factor correlation heatmap over time
# ---------------------------------------------------------------------------

def plot_factor_correlations(
    factors: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Rolling 63-day cross-sectional correlation between factors.

    What to look for:
    - Momentum and low-vol should have LOW correlation (good diversification)
    - If correlation > 0.7, the two factors are essentially the same signal
      and combining them adds no value
    - Correlation spikes during market stress → factor crowding
    """
    mom = factors["momentum"]
    vol = factors["low_vol"]

    # Align
    common = mom.index.intersection(vol.index)
    mom = mom.loc[common]
    vol = vol.loc[common]

    # Rolling cross-sectional Spearman correlation
    window = 63
    rolling_corr = []
    dates = common[window:]

    for i in range(window, len(common)):
        m_slice = mom.iloc[i - window : i]
        v_slice = vol.iloc[i - window : i]
        # Stack all observations, compute rank correlation
        m_vals = m_slice.values.ravel()
        v_vals = v_slice.values.ravel()
        mask = ~(np.isnan(m_vals) | np.isnan(v_vals))
        if mask.sum() < 50:
            rolling_corr.append(np.nan)
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(m_vals[mask], v_vals[mask])
        rolling_corr.append(rho)

    corr_series = pd.Series(rolling_corr, index=dates)

    # Static full-sample correlation matrix
    full_mom = mom.stack().rename("momentum")
    full_vol = vol.stack().rename("low_vol")
    combined = pd.concat([full_mom, full_vol], axis=1).dropna()
    from scipy.stats import spearmanr
    full_corr, _ = spearmanr(combined["momentum"], combined["low_vol"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: rolling correlation
    ax = axes[0]
    ax.plot(corr_series.index, corr_series, color=COLORS["neutral"], linewidth=1.1)
    ax.axhline(full_corr, color="#D6604D", linewidth=1.0, linestyle="--",
               label=f"Full-sample ρ = {full_corr:.3f}")
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle=":")
    ax.fill_between(corr_series.index, corr_series, 0,
                    where=corr_series > 0, alpha=0.15, color="#D6604D")
    ax.fill_between(corr_series.index, corr_series, 0,
                    where=corr_series < 0, alpha=0.15, color="#2166AC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Rolling {window}-day Factor Correlation\n(Momentum vs Low Volatility)")
    ax.legend()
    ax.set_ylim(-1, 1)

    # Right: static scatter (sample 500 obs for legibility)
    ax = axes[1]
    sample = combined.sample(min(500, len(combined)), random_state=42)
    ax.scatter(sample["momentum"], sample["low_vol"],
               alpha=0.3, s=8, color=COLORS["neutral"])
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.axvline(0, color="#888888", linewidth=0.7)
    ax.set_xlabel("Momentum z-score")
    ax.set_ylabel("Low-vol z-score")
    ax.set_title(f"Cross-sectional Scatter\n(full sample ρ = {full_corr:.3f})")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Factor coverage & dispersion
# ---------------------------------------------------------------------------

def plot_factor_quality(factors: dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Two panels:
    Left  — fraction of universe with valid scores each day
    Right — cross-sectional dispersion (std of z-scores each day; should ≈ 1)
    """
    from factor_utils import factor_coverage, factor_dispersion

    factor_names = ["momentum", "low_vol", "composite"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax_cov, ax_disp = axes

    for fname in factor_names:
        df  = factors[fname]
        cov = factor_coverage(df).rolling(21).mean()
        dsp = factor_dispersion(df).rolling(21).mean()
        c   = COLORS[fname]
        lbl = fname.replace("_", " ").title()
        ax_cov.plot(cov.index,  cov,  color=c, linewidth=1.1, label=lbl)
        ax_disp.plot(dsp.index, dsp,  color=c, linewidth=1.1, label=lbl)

    ax_cov.set_ylabel("Coverage (fraction of universe)")
    ax_cov.set_xlabel("Date")
    ax_cov.set_title("Factor Universe Coverage\n(21-day rolling mean)")
    ax_cov.set_ylim(0, 1.05)
    ax_cov.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_cov.legend()

    ax_disp.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", label="Target = 1")
    ax_disp.set_ylabel("Cross-sectional Std of z-scores")
    ax_disp.set_xlabel("Date")
    ax_disp.set_title("Factor Dispersion\n(should be ≈ 1.0 after normalisation)")
    ax_disp.legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(data_dir: Path = DATA_DIR) -> None:
    print("Loading factor data …")
    factors = {
        "momentum":  pd.read_parquet(data_dir / "factor_momentum.parquet"),
        "low_vol":   pd.read_parquet(data_dir / "factor_low_vol.parquet"),
        "composite": pd.read_parquet(data_dir / "factor_composite.parquet"),
    }

    plots = [
        ("factor_distributions",    plot_score_distributions,    (factors,)),
        ("factor_quintile_spread",  plot_factor_quintile_spread, (factors,)),
        ("factor_correlations",     plot_factor_correlations,    (factors,)),
        ("factor_quality",          plot_factor_quality,         (factors,)),
    ]

    for name, fn, args in plots:
        print(f"  Generating {name} …")
        fig = fn(*args)
        path = FIGURE_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    generate_all_figures()
