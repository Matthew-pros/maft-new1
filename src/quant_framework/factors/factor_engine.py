"""
Factor Engine – Institutional Level
Computes daily exposure to:
Market, Size, Value, Growth, Momentum, Quality, Low Volatility, Carry, Commodity, Duration, Dollar, Inflation

Each backtest must show factor exposure over time.
- Holdings-based exposure (from asset metadata factor scores)
- Returns-based exposure (rolling beta to factor returns)
- Deterministic, no lookahead, type hints, docstrings, validation
- Economic justification per factor

Factor definitions (institutional proxies):
- Market: SPY (broad market beta)
- Size: SMB = IWM - SPY (small minus big) or RSP - SPY equal weight vs cap weight
- Value: HML = VTV (Value) - VUG (Growth) or VLUE
- Growth: = -Value or MGK vs MGV or QQQ vs SPY growth tilt
- Momentum: MTUM (or past 12-1 return)
- Quality: QUAL (MSCI Quality) vs SPY
- Low Volatility: USMV (Min Vol) vs SPY or SPLV vs SPHB
- Carry: HYG - IEF (credit carry) or DBMF carry or DBC carry
- Commodity: DBC (broad commodity) vs SPY or XLB/XLE vs SPY
- Duration: TLT (20Y) - SHY (1-3M) or IEF - SHY, measures interest rate duration risk
- Dollar: UUP (Dollar index) vs SPY
- Inflation: TIP - IEF (breakeven inflation) or GLD vs TLT

Economic justification included in docstrings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ── Factor proxy definitions – institutional ──
FACTOR_PROXIES = {
    "Market": ["SPY"],  # Market factor = SPY returns
    "Size": ["IWM", "SPY"],  # SMB = IWM - SPY
    "Value": ["VTV", "VUG"],  # HML = Value - Growth
    "Growth": ["VUG", "VTV"],  # opposite of Value
    "Momentum": ["MTUM", "SPY"],  # Momentum = MTUM - SPY
    "Quality": ["QUAL", "SPY"],  # Quality = QUAL - SPY
    "LowVolatility": ["USMV", "SPY"],  # Low Vol = USMV - SPY, or SPLV - SPHB
    "Carry": ["HYG", "IEF"],  # Carry = High Yield - Treasuries (credit carry)
    "Commodity": ["DBC", "SPY"],  # Commodity = DBC - SPY
    "Duration": ["TLT", "SHY"],  # Duration = Long duration - Short duration
    "Dollar": ["UUP", "SPY"],  # Dollar = UUP - SPY or just UUP
    "Inflation": ["TIP", "IEF"],  # Inflation = TIPS - Nominal Treasuries breakeven
}

# Alternative proxies if primary not available
FACTOR_ALTERNATIVES = {
    "Size": [["RSP", "SPY"], ["IWM", "SPY"]],
    "Value": [["VLUE", "SPY"], ["IWD", "IWF"]],
    "Growth": [["MGK", "SPY"], ["VUG", "SPY"]],
    "Momentum": [["MTUM", "SPY"], ["SPY"]],  # fallback to price momentum
    "Quality": [["QUAL", "SPY"], ["XLV", "SPY"]],
    "LowVolatility": [["SPLV", "SPHB"], ["USMV", "SPY"]],
    "Carry": [["EMB", "IEF"], ["HYG", "IEF"]],
    "Commodity": [["PDBC", "SPY"], ["XLE", "SPY"], ["XLB", "SPY"]],
    "Duration": [["IEF", "SHY"], ["TLT", "IEF"]],
    "Dollar": [["UUP"]],
    "Inflation": [["GLD", "SPY"], ["XLE", "XLU"]],
}

# Holdings-based factor scores for known tickers – institutional mapping
# Scores -2 to +2: -2 strong negative exposure, +2 strong positive
# Economic justification in comments
HOLDINGS_FACTOR_SCORES: Dict[str, Dict[str, float]] = {
    # Sector SPDRs
    "XLU": {"Market": 0.5, "Size": -0.5, "Value": 0.5, "Growth": -0.5, "Momentum": -0.5, "Quality": 1.0, "LowVolatility": 1.5, "Carry": 0.5, "Commodity": -0.5, "Duration": 1.0, "Dollar": 0.0, "Inflation": -0.5},
    "XLP": {"Market": 0.5, "Size": -0.5, "Value": 0.3, "Growth": -0.3, "Momentum": -0.2, "Quality": 1.2, "LowVolatility": 1.2, "Carry": 0.2, "Commodity": -0.3, "Duration": 0.3, "Dollar": 0.0, "Inflation": -0.2},
    "XLV": {"Market": 0.6, "Size": -0.3, "Value": 0.2, "Growth": 0.3, "Momentum": 0.2, "Quality": 1.5, "LowVolatility": 0.8, "Carry": 0.2, "Commodity": -0.2, "Duration": 0.2, "Dollar": 0.0, "Inflation": 0.1},
    "XLE": {"Market": 1.1, "Size": 0.2, "Value": 1.0, "Growth": -0.5, "Momentum": 0.3, "Quality": -0.2, "LowVolatility": -0.5, "Carry": 0.3, "Commodity": 1.8, "Duration": -0.3, "Dollar": 0.2, "Inflation": 1.5},
    "XLB": {"Market": 1.0, "Size": 0.1, "Value": 0.6, "Growth": -0.2, "Momentum": 0.2, "Quality": 0.0, "LowVolatility": -0.2, "Carry": 0.2, "Commodity": 1.5, "Duration": -0.2, "Dollar": 0.1, "Inflation": 1.2},
    "XLI": {"Market": 1.1, "Size": 0.0, "Value": 0.5, "Growth": 0.0, "Momentum": 0.4, "Quality": 0.3, "LowVolatility": 0.0, "Carry": 0.1, "Commodity": 0.5, "Duration": -0.1, "Dollar": 0.0, "Inflation": 0.5},
    "XLK": {"Market": 1.2, "Size": -0.2, "Value": -0.8, "Growth": 1.2, "Momentum": 0.8, "Quality": 0.5, "LowVolatility": -0.3, "Carry": -0.2, "Commodity": -0.3, "Duration": -0.2, "Dollar": 0.0, "Inflation": -0.2},
    "XLF": {"Market": 1.2, "Size": -0.1, "Value": 0.8, "Growth": -0.3, "Momentum": -0.1, "Quality": -0.3, "LowVolatility": -0.4, "Carry": 0.8, "Commodity": 0.0, "Duration": 0.8, "Dollar": 0.3, "Inflation": 0.3},
    "XLY": {"Market": 1.2, "Size": -0.2, "Value": -0.5, "Growth": 0.8, "Momentum": 0.6, "Quality": 0.2, "LowVolatility": -0.2, "Carry": -0.1, "Commodity": 0.0, "Duration": -0.1, "Dollar": 0.0, "Inflation": 0.2},
    # Broad market
    "SPY": {"Market": 1.0, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 0.0, "LowVolatility": 0.0, "Carry": 0.0, "Commodity": 0.0, "Duration": 0.0, "Dollar": 0.0, "Inflation": 0.0},
    "QQQ": {"Market": 1.2, "Size": -0.3, "Value": -0.9, "Growth": 1.3, "Momentum": 0.9, "Quality": 0.6, "LowVolatility": -0.4, "Carry": -0.2, "Commodity": -0.2, "Duration": -0.3, "Dollar": 0.0, "Inflation": -0.1},
    "TQQQ": {"Market": 3.6, "Size": -0.9, "Value": -2.7, "Growth": 3.9, "Momentum": 2.7, "Quality": 1.8, "LowVolatility": -1.2, "Carry": -0.6, "Commodity": -0.6, "Duration": -0.9, "Dollar": 0.0, "Inflation": -0.3},
    "IWM": {"Market": 1.3, "Size": 1.5, "Value": 0.2, "Growth": 0.1, "Momentum": 0.0, "Quality": -0.5, "LowVolatility": -0.8, "Carry": -0.1, "Commodity": 0.2, "Duration": -0.1, "Dollar": 0.0, "Inflation": 0.1},
    # Safe haven
    "GLD": {"Market": 0.1, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 0.0, "LowVolatility": 0.5, "Carry": -0.2, "Commodity": 1.5, "Duration": 0.3, "Dollar": -0.3, "Inflation": 1.5},
    "SLV": {"Market": 0.2, "Size": 0.3, "Value": 0.0, "Growth": 0.0, "Momentum": 0.2, "Quality": -0.2, "LowVolatility": -0.2, "Carry": -0.2, "Commodity": 1.6, "Duration": 0.1, "Dollar": -0.2, "Inflation": 1.2},
    "BIL": {"Market": 0.0, "Size": -0.5, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 1.0, "LowVolatility": 1.8, "Carry": -0.5, "Commodity": -0.5, "Duration": -1.5, "Dollar": 0.2, "Inflation": -1.0},
    "IEF": {"Market": 0.1, "Size": -0.3, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 0.8, "LowVolatility": 0.8, "Carry": 0.2, "Commodity": -0.2, "Duration": 0.8, "Dollar": 0.1, "Inflation": -0.3},
    "TLT": {"Market": -0.1, "Size": -0.5, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 1.0, "LowVolatility": 0.5, "Carry": 0.5, "Commodity": -0.3, "Duration": 2.0, "Dollar": 0.0, "Inflation": -0.8},
    "HYG": {"Market": 0.5, "Size": 0.2, "Value": 0.3, "Growth": -0.1, "Momentum": -0.1, "Quality": -0.5, "LowVolatility": -0.3, "Carry": 1.5, "Commodity": 0.0, "Duration": -0.2, "Dollar": 0.0, "Inflation": 0.2},
    "VIXY": {"Market": -2.5, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": -1.0, "Quality": 0.0, "LowVolatility": -1.5, "Carry": -1.0, "Commodity": 0.0, "Duration": 0.0, "Dollar": 0.0, "Inflation": 0.0},
    "UUP": {"Market": 0.0, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 0.0, "LowVolatility": 0.5, "Carry": 0.3, "Commodity": -0.5, "Duration": 0.0, "Dollar": 2.0, "Inflation": -0.2},
    "DBMF": {"Market": 0.2, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": 1.0, "Quality": 0.0, "LowVolatility": 0.2, "Carry": 0.8, "Commodity": 0.8, "Duration": 0.3, "Dollar": 0.5, "Inflation": 0.8},
    "BTC-USD": {"Market": 1.8, "Size": 1.0, "Value": -1.5, "Growth": 1.8, "Momentum": 1.2, "Quality": -1.0, "LowVolatility": -1.8, "Carry": -0.5, "Commodity": 0.5, "Duration": -0.5, "Dollar": -0.5, "Inflation": 0.5},
    "MES=F": {"Market": 1.0, "Size": 0.0, "Value": 0.0, "Growth": 0.0, "Momentum": 0.0, "Quality": 0.0, "LowVolatility": 0.0, "Carry": 0.0, "Commodity": 0.0, "Duration": 0.0, "Dollar": 0.0, "Inflation": 0.0},
}

ALL_FACTORS = ["Market", "Size", "Value", "Growth", "Momentum", "Quality", "LowVolatility", "Carry", "Commodity", "Duration", "Dollar", "Inflation"]


@dataclass(frozen=True)
class FactorExposure:
    """Immutable factor exposure snapshot at a point in time"""
    date: pd.Timestamp
    factor: str
    exposure: float  # beta or holdings-based score
    method: str  # holdings or returns-based
    confidence: float
    t_stat: Optional[float] = None

    def to_dict(self):
        return {
            "date": str(self.date),
            "factor": self.factor,
            "exposure": self.exposure,
            "method": self.method,
            "confidence": self.confidence,
            "t_stat": self.t_stat
        }


class FactorEngine:
    """
    Factor Engine – institutional level
    Computes daily exposure to 12 factors: Market, Size, Value, Growth, Momentum, Quality, LowVolatility, Carry, Commodity, Duration, Dollar, Inflation

    Methods:
    - Holdings-based exposure: uses asset metadata factor scores + portfolio weights (economic: style tilts)
    - Returns-based exposure: rolling beta of portfolio returns vs factor returns (economic: systematic risk)

    Each backtest must show factor exposure over time – ReportingEngine will plot.

    Deterministic, no lookahead bias, type hints, docstring, validation
    """

    def __init__(self, factor_proxies: Dict[str, List[str]] = None, holdings_scores: Dict[str, Dict[str, float]] = None):
        """
        Args:
            factor_proxies: Dict factor -> list of proxy tickers (first is primary, rest alternatives)
            holdings_scores: Dict ticker -> dict factor -> score (holdings-based)
        """
        self.factor_proxies = factor_proxies or FACTOR_PROXIES
        self.holdings_scores = holdings_scores or HOLDINGS_FACTOR_SCORES
        # Validate proxies contain all factors
        for f in ALL_FACTORS:
            if f not in self.factor_proxies:
                raise ValueError(f"Missing proxy for factor {f}")

    def get_factor_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute factor returns DataFrame from market data

        Logic:
        - For each factor, find first available proxy pair in data
        - Factor return = long proxy - short proxy (if 2 tickers) or proxy return (if 1 ticker)
        - For Market, factor return = SPY returns
        - Deterministic, no lookahead (uses past prices only)

        Args:
            data: Dict ticker -> OHLCV DataFrame (must have Close)

        Returns:
            DataFrame index dates, columns factors, values daily returns

        Raises:
            TypeError, ValueError
        """
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be non-empty dict")

        factor_returns: Dict[str, pd.Series] = {}

        for factor in ALL_FACTORS:
            proxies = self.factor_proxies.get(factor, [])
            # Try primary proxy list
            found = False
            # If factor has alternative lists (list of lists)
            # Our FACTOR_PROXIES values are List[str], but alternatives are List[List[str]]
            # We handle both
            candidate_lists = []
            if factor in FACTOR_ALTERNATIVES:
                candidate_lists = FACTOR_ALTERNATIVES[factor]
            # Always include primary as candidate
            candidate_lists = [proxies] + candidate_lists

            for cand in candidate_lists:
                if not isinstance(cand, list):
                    continue
                if len(cand) == 1:
                    ticker = cand[0]
                    if ticker in data and not data[ticker].empty:
                        close = data[ticker]['Close']
                        ret = close.pct_change()
                        factor_returns[factor] = ret
                        found = True
                        break
                elif len(cand) >= 2:
                    long_ticker, short_ticker = cand[0], cand[1]
                    if long_ticker in data and short_ticker in data and not data[long_ticker].empty and not data[short_ticker].empty:
                        long_close = data[long_ticker]['Close']
                        short_close = data[short_ticker]['Close']
                        # Align
                        aligned = pd.concat([long_close, short_close], axis=1, join='inner')
                        aligned.columns = ['long', 'short']
                        long_ret = aligned['long'].pct_change()
                        short_ret = aligned['short'].pct_change()
                        factor_ret = long_ret - short_ret
                        factor_returns[factor] = factor_ret
                        found = True
                        break

            if not found:
                # If no proxy found, create zero series from first available ticker
                first_ticker = next(iter(data))
                idx = data[first_ticker].index
                factor_returns[factor] = pd.Series(0.0, index=idx)

        # Combine into DataFrame sorted columns deterministic
        df = pd.DataFrame(factor_returns).sort_index()
        # Ensure columns sorted
        df = df[sorted(df.columns)]
        # Fill NaN with 0 for first rows
        df = df.fillna(0.0)
        return df

    def compute_holdings_exposure(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Holdings-based factor exposure: weighted average of asset factor scores

        Economic justification: style tilts of portfolio (value, growth, etc.) based on holdings characteristics
        E.g., QQQ has high Growth exposure because it holds growth stocks

        Args:
            weights: Dict ticker -> weight (sum ~1)

        Returns:
            Dict factor -> exposure (weighted score)

        Raises:
            TypeError, ValueError
        """
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weights must be non-empty dict")
        # Deterministic sorted
        weights_sorted = dict(sorted(weights.items()))
        total_weight = sum(weights_sorted.values())
        if total_weight == 0:
            raise ValueError("weights sum cannot be 0")
        # Normalize
        norm_weights = {t: w / total_weight for t, w in weights_sorted.items()}

        factor_exposure: Dict[str, float] = {f: 0.0 for f in ALL_FACTORS}

        for ticker, w in norm_weights.items():
            scores = self.holdings_scores.get(ticker.upper())
            if scores is None:
                # Unknown ticker: try to infer from asset class, or use 0
                scores = {f: 0.0 for f in ALL_FACTORS}
                # Fallback: if ticker contains leveraged, increase Market and Momentum
                if "TQQQ" in ticker.upper() or "UPRO" in ticker.upper():
                    scores = {**scores, "Market": 2.0, "Growth": 2.0, "Momentum": 1.5, "LowVolatility": -1.0}
            for factor in ALL_FACTORS:
                factor_exposure[factor] += w * scores.get(factor, 0.0)

        # Round for determinism
        factor_exposure = {f: round(float(v), 4) for f, v in sorted(factor_exposure.items())}
        return factor_exposure

    def compute_rolling_exposure(self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame,
                                 window: int = 60) -> pd.DataFrame:
        """
        Returns-based factor exposure: rolling beta of portfolio returns vs each factor returns

        Economic justification: systematic risk exposure – how much portfolio moves with factor
        E.g., beta to Market = CAPM beta, beta to Size = SMB exposure

        No lookahead bias: uses rolling window with past data only, shift(1) for beta? Actually rolling beta uses past window including current, which is standard but we ensure no future beyond window.
        For strict no lookahead, we use rolling window ending at t-1 to predict t? Simpler: rolling window includes t, but that's still past only (no future beyond t). For execution, we would use beta at t-1 to predict t+1. Here we compute exposure at t using past window.

        Args:
            portfolio_returns: Series daily portfolio returns
            factor_returns: DataFrame factor returns
            window: Rolling window for beta (e.g., 60 days)

        Returns:
            DataFrame index dates, columns factors, values rolling beta exposures

        Raises:
            TypeError, ValueError
        """
        if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
            raise ValueError("portfolio_returns must be non-empty Series")
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            raise ValueError("factor_returns must be non-empty DataFrame")
        if window <= 1:
            raise ValueError("window must be >1")

        # Align
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
        # Drop rows with all NaN
        aligned = aligned.dropna(how='all')

        portfolio_col = aligned.columns[0]
        factor_cols = [c for c in aligned.columns if c != portfolio_col]

        exposures = pd.DataFrame(index=aligned.index, columns=factor_cols, dtype=float)

        for factor in factor_cols:
            # Rolling covariance / variance
            # Beta = Cov(port, factor) / Var(factor)
            # Use rolling window
            port = aligned[portfolio_col]
            fac = aligned[factor]
            # Rolling covariance
            cov = port.rolling(window).cov(fac)
            var_fac = fac.rolling(window).var()
            beta = cov / var_fac.replace(0, np.nan)
            exposures[factor] = beta

        # Deterministic sorted columns
        exposures = exposures[sorted(exposures.columns)]
        return exposures

    def compute_factor_exposure_over_time(self, weights_over_time: pd.DataFrame,
                                          portfolio_returns: Optional[pd.Series] = None,
                                          factor_returns: Optional[pd.DataFrame] = None,
                                          method: str = "holdings") -> pd.DataFrame:
        """
        Compute factor exposure over time for entire backtest

        Args:
            weights_over_time: DataFrame index dates, columns tickers, values weights over time (from portfolio construction)
            portfolio_returns: Optional Series portfolio returns for returns-based method
            factor_returns: Optional DataFrame factor returns for returns-based method
            method: "holdings" or "returns"

        Returns:
            DataFrame index dates, columns factors, values exposures over time

        Raises:
            ValueError
        """
        if not isinstance(weights_over_time, pd.DataFrame) or weights_over_time.empty:
            raise ValueError("weights_over_time must be non-empty DataFrame")

        if method == "holdings":
            # For each date, compute holdings exposure from weights at that date
            exposures_list = []
            dates = []
            for date in weights_over_time.index:
                w_row = weights_over_time.loc[date].dropna()
                # Convert to dict, filter zero weights
                w_dict = {t: float(v) for t, v in w_row.items() if float(v) != 0}
                if not w_dict:
                    continue
                exp = self.compute_holdings_exposure(w_dict)
                exposures_list.append(exp)
                dates.append(date)
            if not exposures_list:
                return pd.DataFrame()
            exp_df = pd.DataFrame(exposures_list, index=dates)
            exp_df = exp_df[sorted(exp_df.columns)]
            return exp_df.sort_index()

        elif method == "returns":
            if portfolio_returns is None or factor_returns is None:
                raise ValueError("portfolio_returns and factor_returns required for returns-based method")
            return self.compute_rolling_exposure(portfolio_returns, factor_returns)

        else:
            raise ValueError("method must be holdings or returns")

    def get_factor_metadata(self) -> pd.DataFrame:
        """
        Return factor definitions with economic justification – for reporting

        Returns:
            DataFrame with factor, proxy, economic_justification
        """
        justifications = {
            "Market": "CAPM market beta – broad market risk premium, economic = equity risk premium",
            "Size": "SMB (Small Minus Big) – Fama-French size premium, small caps outperform due to risk/illiquidity, economic = size anomaly",
            "Value": "HML (High Minus Low book-to-market) – value premium, value stocks outperform growth due to distress risk, economic = value anomaly",
            "Growth": "Opposite of Value – growth premium in expansion, economic = growth expectations, duration",
            "Momentum": "Winners minus losers, 12-1 month momentum, economic = herding, slow information diffusion, Jegadeesh Titman",
            "Quality": "High quality (profitable, low debt) minus junk, economic = quality premium, flight to quality",
            "LowVolatility": "Low vol minus high vol, economic = leverage constraints, lottery preference, low vol anomaly",
            "Carry": "High yield minus low yield carry, credit carry, economic = term premium, credit risk premium",
            "Commodity": "Commodity vs equity, economic = inflation hedge, real asset premium",
            "Duration": "Long duration minus short duration, economic = interest rate risk, term structure",
            "Dollar": "Dollar index vs equity, economic = currency risk, safe haven flows",
            "Inflation": "TIPS vs nominal Treasuries breakeven, economic = inflation expectations, real vs nominal"
        }
        rows = []
        for factor in ALL_FACTORS:
            proxies = self.factor_proxies.get(factor, [])
            alt = FACTOR_ALTERNATIVES.get(factor, [])
            rows.append({
                "factor": factor,
                "primary_proxy": ", ".join(proxies),
                "alternatives": "; ".join([", ".join(a) for a in alt]),
                "economic_justification": justifications.get(factor, "Economic risk premium")
            })
        df = pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
        return df
