"""
universe.py
-----------
Defines the S&P 100 investment universe.

Design principle (institutional practice):
    The universe is the single source of truth for all downstream modules.
    Hard-coding the ticker list — rather than scraping it live — ensures
    reproducibility: tomorrow's index membership may differ from today's,
    which would silently break backtests run months later.
"""

# S&P 100 constituents as of May 2025
# Source: CBOE OEX index constituent list
SP100_TICKERS: list[str] = [
    "AAPL", "ABBV", "ABT",  "ACN",  "ADBE", "AIG",  "AMD",  "AMGN",
    "AMT",  "AMZN", "AXP",  "BA",   "BAC",  "BK",   "BKNG", "BLK",
    "BMY",  "BRK-B","C",    "CAT",  "CHTR", "CL",   "CMCSA","COF",
    "COP",  "COST", "CRM",  "CSCO", "CVS",  "CVX",  "DD",   "DHR",
    "DIS",  "DOW",  "DUK",  "EMR",  "EXC",  "F",    "FDX",  "GD",
    "GE",   "GILD", "GM",   "GOOG", "GOOGL","GS",   "HAL",  "HD",
    "HON",  "IBM",  "INTC", "INTU", "JNJ",  "JPM",  "KHC",  "KMI",
    "KO",   "LIN",  "LLY",  "LMT",  "LOW",  "MA",   "MCD",  "MDLZ",
    "MDT",  "MET",  "META", "MMM",  "MO",   "MRK",  "MS",   "MSFT",
    "NEE",  "NFLX", "NKE",  "NVDA", "ORCL", "OXY",  "PEP",  "PFE",
    "PG",   "PM",   "PYPL", "QCOM", "RTX",  "SBUX", "SCHW", "SO",
    "SPG",  "T",    "TGT",  "TMO",  "TMUS", "TSLA", "TXN",  "UNH",
    "UNP",  "UPS",  "USB",  "V",    "VZ",   "WBA",  "WFC",  "WMT",
    "XOM",
]

# De-duplicate while preserving order (safety net)
SP100_TICKERS = list(dict.fromkeys(SP100_TICKERS))


def get_universe(exclude: list[str] | None = None) -> list[str]:
    """
    Return the ticker universe, optionally excluding specific symbols.

    Parameters
    ----------
    exclude : list[str] | None
        Tickers to drop (e.g. stocks with data quality issues discovered
        during pipeline validation).

    Returns
    -------
    list[str]
        Sorted list of uppercase ticker strings.
    """
    tickers = SP100_TICKERS.copy()
    if exclude:
        exclude_upper = {t.upper() for t in exclude}
        tickers = [t for t in tickers if t not in exclude_upper]
    return sorted(tickers)


if __name__ == "__main__":
    universe = get_universe()
    print(f"Universe size: {len(universe)} tickers")
    print(universe)
