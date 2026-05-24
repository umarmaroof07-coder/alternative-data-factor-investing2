"""
alternative_data.py
-------------------
Phase 5a — Synthetic Alternative Data Generation Engine

PURPOSE
-------
Alternative data refers to any non-traditional data source used to gain
an informational edge in equity investing.  Real-world examples include:

  • Job postings / hiring trends    (LinkedIn, Glassdoor, Burning Glass)
  • News & social media sentiment   (RavenPack, Bloomberg NLP, Refinitiv)
  • Satellite imagery               (SpaceKnow, Orbital Insight)
  • Credit card transactions        (Second Measure, Earnest Research)
  • Web traffic / app downloads     (SimilarWeb, Apptopia)
  • Supply chain signals            (TruEra, Altana)

WHY SIMULATE RATHER THAN USE REAL DATA?
----------------------------------------
1. Real alt-data vendors charge $50K–$500K/year per dataset.
2. Simulation lets us prove the pipeline works correctly before paying.
3. We can inject *known* alpha signals into synthetic data, then verify
   the pipeline recovers them — a critical validation step.
4. The simulation is architecturally identical to a live vendor feed:
   same DataFrame shapes, same date conventions, same look-ahead controls.

SIMULATION METHODOLOGY
-----------------------
We generate data with three layers of structure:

  Layer 1 — Market factor:
    All companies share a common macro signal (e.g., broad economy hiring
    trends up/down together).

  Layer 2 — Sector factor:
    Companies in the same industry co-move (tech hiring booms/busts
    together; energy hiring tracks oil prices).

  Layer 3 — Idiosyncratic signal:
    Each company has its own noise process PLUS a persistent "quality"
    effect that creates mild cross-sectional alpha.

  Layer 4 — Known alpha injection:
    We deliberately build in a positive correlation between the
    alt-data signal and future 1-month returns.  This lets us verify
    that the backtest correctly recovers this injected alpha.

LOOK-AHEAD BIAS CONTROLS
-------------------------
Alternative data has particularly severe look-ahead bias risks:

  Problem 1 — Publication lag:
    A company's Q3 earnings call happens Oct 22.  Job posting data
    collected on Oct 22 *reflects* the call but was available *before*.
    However, if we measure "Q3 job postings" we must use postings from
    July–September, not October.  We enforce this with explicit lags.

  Problem 2 — Backfill bias:
    Some vendors retroactively improve historical data quality.  A data
    point that "would have been" 85 gets revised to 92 after the fact.
    Simulation avoids this by generating data forward-in-time only.

  Problem 3 — Survivorship in alt data:
    Only companies that existed and were covered by the vendor appear.
    We maintain full coverage of all tickers for the full date range.

Author  : Quant Research Team
Phase   : 5 — Alternative Data Alpha Layer
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector definitions (drives cross-sectional correlation structure)
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, list[str]] = {
    "Technology":    ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
                      "IBM",  "QCOM", "TXN",  "ADBE","CRM",  "INTU", "ACN"],
    "Communication": ["GOOG", "GOOGL","META", "NFLX","CHTR", "CMCSA","T",
                      "VZ",   "TMUS", "DIS"],
    "ConsDiscret":   ["AMZN", "TSLA", "HD",   "LOW", "NKE",  "MCD",  "SBUX",
                      "TGT",  "BKNG", "F",    "GM"],
    "ConsStaples":   ["WMT",  "COST", "PG",   "KO",  "PEP",  "MO",   "PM",
                      "CL",   "MDLZ", "KHC",  "WBA"],
    "Healthcare":    ["JNJ",  "UNH",  "PFE",  "ABBV","MRK",  "ABT",  "TMO",
                      "MDT",  "AMGN", "GILD", "BMY", "LLY",  "DHR",  "CVS"],
    "Financials":    ["JPM",  "BAC",  "WFC",  "GS",  "MS",   "C",    "BLK",
                      "AXP",  "USB",  "BK",   "COF", "MET",  "AIG",  "SCHW",
                      "MA",   "V",    "PYPL", "BRK-B"],
    "Industrials":   ["GE",   "HON",  "BA",   "CAT", "LMT",  "UPS",  "UNP",
                      "RTX",  "GD",   "EMR",  "DD",  "FDX",  "MMM"],
    "Energy":        ["XOM",  "CVX",  "COP",  "HAL", "KMI",  "OXY"],
    "Utilities":     ["NEE",  "DUK",  "SO",   "EXC"],
    "RealEstate":    ["AMT",  "SPG"],
    "Materials":     ["LIN",  "DOW"],
}

# Reverse map: ticker -> sector
TICKER_TO_SECTOR: dict[str, str] = {
    ticker: sector
    for sector, tickers in SECTOR_MAP.items()
    for ticker in tickers
}


def _get_sector(ticker: str) -> str:
    """Return sector label for a ticker, defaulting to 'Other'."""
    return TICKER_TO_SECTOR.get(ticker, "Other")


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

class AltDataSimulator:
    """
    Generates realistic synthetic alternative data panels.

    The simulator produces monthly observations (matching the alt-data
    vendor delivery cadence) with realistic:
      - Autocorrelation (alt signals are sticky; don't jump overnight)
      - Cross-sectional dispersion (some companies always score higher)
      - Sector clustering (tech and healthcare co-move)
      - Injected forward-looking alpha (recoverable by the backtest)
      - Measurement noise (real alt data is never clean)

    Parameters
    ----------
    tickers     : list of ticker symbols (defines the cross-section)
    start_date  : first date of the panel
    end_date    : last date of the panel
    signal_name : label used in logs and output column names
    seed        : numpy RNG seed (for reproducibility)
    alpha_corr  : target rank-correlation between signal and next-month return
                  0.05 = weak but economically meaningful alpha
                  0.15 = strong alpha (suspicious in real data)
    ar_coef     : AR(1) coefficient for time-series persistence
                  0.80 = moderately sticky (monthly hiring data)
                  0.60 = faster-moving (daily sentiment)
    """

    def __init__(
        self,
        tickers:     list[str],
        start_date:  str | pd.Timestamp,
        end_date:    str | pd.Timestamp,
        signal_name: str,
        seed:        int   = 42,
        alpha_corr:  float = 0.08,
        ar_coef:     float = 0.78,
    ) -> None:
        self.tickers     = tickers
        self.start_date  = pd.Timestamp(start_date)
        self.end_date    = pd.Timestamp(end_date)
        self.signal_name = signal_name
        self.rng         = np.random.default_rng(seed)
        self.alpha_corr  = alpha_corr
        self.ar_coef     = ar_coef

        # Build monthly business-month-end date index
        self.monthly_dates = pd.bdate_range(
            start=self.start_date, end=self.end_date, freq="BME"
        )
        self.T = len(self.monthly_dates)
        self.N = len(tickers)

        log.debug(
            f"[{signal_name}] Simulator initialised | "
            f"{self.T} months × {self.N} tickers | alpha_corr={alpha_corr:.2f}"
        )

    def _simulate_ar1_panel(self) -> np.ndarray:
        """
        Simulate a T × N AR(1) panel with sector-level correlation.

        AR(1): x_t = φ·x_{t-1} + ε_t    where ε ~ N(0, σ²)

        WHY AR(1)?
        ----------
        Real alternative signals are persistent (autocorrelated):
        - A company hiring aggressively in January continues in February.
        - Sentiment shifts gradually unless a shock occurs.

        AR(1) is the simplest stationary model that captures this persistence.
        The AR coefficient φ controls the half-life:
            half_life = log(0.5) / log(φ)
        At φ=0.78: half-life ≈ 2.8 months (signal fades over ~3 months)
        At φ=0.60: half-life ≈ 1.4 months
        """
        phi   = self.ar_coef
        sigma = np.sqrt(1 - phi**2)   # innovations scaled so stationary var = 1

        # --- Market factor (shared across all tickers) ---
        mkt_factor = np.zeros(self.T)
        mkt_factor[0] = self.rng.standard_normal()
        for t in range(1, self.T):
            mkt_factor[t] = phi * mkt_factor[t-1] + sigma * self.rng.standard_normal()

        # --- Sector factors (one per sector) ---
        sector_labels  = [_get_sector(t) for t in self.tickers]
        unique_sectors = list(set(sector_labels))
        sector_factors = {}
        for s in unique_sectors:
            sf = np.zeros(self.T)
            sf[0] = self.rng.standard_normal()
            for t in range(1, self.T):
                sf[t] = phi * sf[t-1] + sigma * self.rng.standard_normal()
            sector_factors[s] = sf

        # --- Company-level persistent quality scores ---
        # Each ticker has a stable "quality" that makes it persistently high/low
        company_quality = self.rng.normal(0, 0.5, size=self.N)

        # --- Idiosyncratic AR(1) for each ticker ---
        idio = np.zeros((self.T, self.N))
        idio[0] = self.rng.standard_normal(self.N)
        for t in range(1, self.T):
            idio[t] = phi * idio[t-1] + sigma * self.rng.standard_normal(self.N)

        # --- Combine: market (30%) + sector (25%) + quality (15%) + idio (30%) ---
        panel = np.zeros((self.T, self.N))
        for i, (ticker, sector) in enumerate(zip(self.tickers, sector_labels)):
            panel[:, i] = (
                0.30 * mkt_factor
                + 0.25 * sector_factors[sector]
                + 0.15 * company_quality[i]
                + 0.30 * idio[:, i]
            )

        return panel

    def _inject_forward_alpha(
        self,
        panel:        np.ndarray,
        future_ranks: np.ndarray | None,
    ) -> np.ndarray:
        """
        Blend in a forward-looking signal to create injected alpha.

        WHY DO THIS?
        ------------
        In real markets, good alt data has genuine predictive power.
        We inject this so the backtest can validate that the pipeline
        correctly recovers the known alpha.

        Method: for each month t, blend the raw signal with the
        *next month's* cross-sectional return rank.  The alpha_corr
        parameter controls the blend weight.

        This is a purely internal simulation tool — in production, the
        predictive power comes from the data itself, not from blending.

        Parameters
        ----------
        panel        : T × N raw AR(1) panel
        future_ranks : T × N matrix of next-month return ranks (0 to 1)
                       If None, no alpha is injected.
        """
        if future_ranks is None or self.alpha_corr <= 0:
            return panel

        # Blend: signal = (1-α)·raw + α·future_rank_zscore
        # Convert ranks to z-scores first so scales match
        rank_zscores = (future_ranks - 0.5) / 0.3   # rough standardisation

        blended = (1 - self.alpha_corr) * panel + self.alpha_corr * rank_zscores
        return blended

    def generate(
        self,
        future_returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate the full synthetic alt-data panel.

        Parameters
        ----------
        future_returns : T × N daily return DataFrame.  If provided, used to
                         compute next-month return ranks for alpha injection.
                         CRITICAL: must be the *same* returns used in the
                         backtest.  The injection uses FUTURE returns only
                         during generation — the output signal itself is
                         used with a 1-month LAG in the backtest, preserving
                         look-ahead safety.

        Returns
        -------
        pd.DataFrame  : T_monthly × N panel of alt-data scores
                        Standardised to approx N(0, 1) cross-sectionally.
                        DatetimeIndex = business-month-end dates.
        """
        log.info(
            f"[{self.signal_name}] Generating {self.T} months × "
            f"{self.N} tickers …"
        )

        raw_panel = self._simulate_ar1_panel()

        # --- Compute next-month return ranks for alpha injection ---
        future_ranks = None
        if future_returns is not None and self.alpha_corr > 0:
            future_ranks = self._compute_monthly_fwd_ranks(future_returns)

        panel = self._inject_forward_alpha(raw_panel, future_ranks)

        # --- Add measurement noise (real alt data is never clean) ---
        noise_scale = 0.15
        panel += noise_scale * self.rng.standard_normal(panel.shape)

        # --- Inject occasional missing data (vendor outages, coverage gaps) ---
        missing_mask = self.rng.random(panel.shape) < 0.02   # 2% missingness
        panel[missing_mask] = np.nan

        # --- Pack into DataFrame ---
        df = pd.DataFrame(
            panel,
            index=self.monthly_dates,
            columns=self.tickers,
        )
        df.index.name = "date"
        df.columns.name = "ticker"

        log.info(
            f"[{self.signal_name}] Generated | "
            f"NaN rate: {df.isna().mean().mean():.1%} | "
            f"score range: [{df.stack().min():.2f}, {df.stack().max():.2f}]"
        )
        return df

    def _compute_monthly_fwd_ranks(
        self,
        daily_returns: pd.DataFrame,
    ) -> np.ndarray | None:
        """
        Compute forward 1-month return ranks aligned to the monthly alt-data dates.

        For each monthly alt-data observation on date t, the 'forward return'
        is the cumulative return over the *next* calendar month.

        LOOK-AHEAD SAFETY NOTE
        -----------------------
        This method uses future returns ONLY during simulation to inject
        known predictive structure into the synthetic data.  The resulting
        signal DataFrame is then consumed by the backtest with a 1-month
        lag (portfolio formed at t earns return at t+1), so no look-ahead
        bias enters the actual backtest.

        In production, you would NEVER have access to future returns —
        the predictive power would come from the actual information content
        of the data (e.g., a company posting more jobs genuinely predicts
        growth).
        """
        T, N = len(self.monthly_dates), self.N
        fwd_ranks = np.full((T, N), np.nan)

        # Align tickers
        common_tickers = [t for t in self.tickers if t in daily_returns.columns]
        if not common_tickers:
            return None

        for i, date in enumerate(self.monthly_dates):
            if i + 1 >= T:
                break
            next_date = self.monthly_dates[i + 1]

            # Cumulative return over next month
            month_ret = (
                daily_returns
                .loc[date:next_date, common_tickers]
                .apply(lambda col: (1 + col).prod() - 1, axis=0)
            )
            if month_ret.isna().all():
                continue

            # Cross-sectional rank → [0, 1]
            ranks = month_ret.rank(pct=True)

            for j, ticker in enumerate(self.tickers):
                if ticker in ranks.index:
                    fwd_ranks[i, j] = ranks[ticker]

        return fwd_ranks
