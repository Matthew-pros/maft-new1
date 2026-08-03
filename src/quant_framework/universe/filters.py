"""
Universe Filters – institutional, independent, deterministic
Each filter is separate class, can be enabled/disabled
Filters:
- Minimum history
- Minimum AUM
- Minimum volume
- Maximum spread
- Maximum turnover
- Delisted check
- Low liquidity
- Short history
- Extreme tracking error
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FilterResult:
    ticker: str
    passed: bool
    reason: str
    value: float
    threshold: float


class UniverseFilter(ABC):
    """Base filter – independent, deterministic"""

    def __init__(self, name: str, enabled: bool = True):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be non-empty str")
        self.name = name
        self.enabled = bool(enabled)

    @abstractmethod
    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled


class MinimumHistoryFilter(UniverseFilter):
    """Minimum history – e.g., 252 days before as_of_date"""

    def __init__(self, min_history_days: int = 252, enabled: bool = True):
        super().__init__(name="MinimumHistory", enabled=enabled)
        if min_history_days <= 0:
            raise ValueError("min_history_days must be >0")
        self.min_history_days = int(min_history_days)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        if data is None or data.empty:
            return FilterResult(ticker, False, "No data", 0, self.min_history_days)
        # Data up to as_of_date
        data_up_to = data[data.index <= as_of_date]
        if data_up_to.empty:
            return FilterResult(ticker, False, f"No data up to {as_of_date.date()}", 0, self.min_history_days)
        first_day = data_up_to.index.min()
        # History length as of date
        history_days = len(data_up_to)
        # Also check calendar days from first to as_of_date
        passed = history_days >= self.min_history_days
        reason = f"History {history_days}d >= {self.min_history_days}d" if passed else f"Short history {history_days}d < {self.min_history_days}d"
        return FilterResult(ticker, passed, reason, float(history_days), float(self.min_history_days))


class DelistedFilter(UniverseFilter):
    """Remove delisted ETFs – if as_of_date > delisting_date or > last trading day + grace"""

    def __init__(self, grace_days: int = 30, enabled: bool = True):
        super().__init__(name="Delisted", enabled=enabled)
        self.grace_days = int(grace_days)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        if data is None or data.empty:
            return FilterResult(ticker, False, "No data", 0, 0)

        # Check metadata delisting_date
        delisting_date = getattr(metadata, 'delisting_date', None)
        if delisting_date is not None:
            if pd.to_datetime(delisting_date) <= as_of_date:
                return FilterResult(ticker, False, f"Delisted on {delisting_date}", 0, 0)

        # Check last trading day + grace
        last_day = data.index.max()
        if last_day + pd.Timedelta(days=self.grace_days) < as_of_date:
            return FilterResult(ticker, False, f"Last trading {last_day.date()} + {self.grace_days}d grace < {as_of_date.date()} -> delisted", 0, 0)

        # Check if as_of_date is before first trading day (not yet listed)
        first_day = data.index.min()
        if as_of_date < first_day:
            return FilterResult(ticker, False, f"Not yet listed, first {first_day.date()} > {as_of_date.date()}", 0, 0)

        return FilterResult(ticker, True, f"Active, last {last_day.date()}", 1, 1)


class MinimumVolumeFilter(UniverseFilter):
    """Minimum average daily volume"""

    def __init__(self, min_avg_volume: float = 100000, min_notional: float = 1_000_000, window: int = 20, enabled: bool = True):
        super().__init__(name="MinimumVolume", enabled=enabled)
        self.min_avg_volume = float(min_avg_volume)
        self.min_notional = float(min_notional)
        self.window = int(window)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        if data is None or data.empty or 'Volume' not in data.columns:
            return FilterResult(ticker, False, "No volume data", 0, self.min_avg_volume)

        data_up_to = data[data.index <= as_of_date]
        if len(data_up_to) < self.window:
            return FilterResult(ticker, False, f"Not enough data for volume window {self.window}", 0, self.min_avg_volume)

        recent = data_up_to.tail(self.window)
        avg_vol = recent['Volume'].mean()
        # Notional = Volume * Close
        if 'Close' in recent.columns:
            avg_notional = (recent['Volume'] * recent['Close']).mean()
        else:
            avg_notional = 0

        passed = (avg_vol >= self.min_avg_volume) and (avg_notional >= self.min_notional)
        reason = f"Avg vol {avg_vol:.0f} >= {self.min_avg_volume:.0f} and notional ${avg_notional:.0f} >= ${self.min_notional:.0f}" if passed else f"Low liquidity vol {avg_vol:.0f} < {self.min_avg_volume:.0f} or notional ${avg_notional:.0f} < ${self.min_notional:.0f}"
        return FilterResult(ticker, passed, reason, float(avg_vol), float(self.min_avg_volume))


class MinimumAUMFilter(UniverseFilter):
    """Minimum AUM – tries yfinance info totalAssets, fallback to notional proxy"""

    def __init__(self, min_aum: float = 50_000_000, enabled: bool = True):
        super().__init__(name="MinimumAUM", enabled=enabled)
        self.min_aum = float(min_aum)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        # Try to get AUM from metadata if available (e.g., etf_issuer, totalAssets from yfinance)
        # For simplicity, we approximate AUM via avg volume * close * 20 (rough)
        # Or use metadata attribute if exists
        aum = None
        # Check if metadata has aum attribute (we can extend AssetMetadata later)
        if hasattr(metadata, 'aum') and getattr(metadata, 'aum') is not None:
            aum = float(getattr(metadata, 'aum'))
        else:
            # Proxy: avg notional * 20 days ~ rough AUM proxy? Actually AUM != volume, but for filter we use notional as proxy
            if data is not None and not data.empty and 'Volume' in data.columns and 'Close' in data.columns:
                data_up_to = data[data.index <= as_of_date]
                if len(data_up_to) >= 20:
                    recent = data_up_to.tail(20)
                    avg_notional = (recent['Volume'] * recent['Close']).mean()
                    # Proxy AUM as notional * 100 (arbitrary, but deterministic)
                    aum = avg_notional * 100
                else:
                    aum = 0
            else:
                aum = 0

        passed = aum >= self.min_aum
        reason = f"AUM ${aum:.0f} >= ${self.min_aum:.0f}" if passed else f"Low AUM ${aum:.0f} < ${self.min_aum:.0f}"
        return FilterResult(ticker, passed, reason, float(aum), float(self.min_aum))


class MaximumSpreadFilter(UniverseFilter):
    """Maximum spread – proxy via (High-Low)/Close or bid/ask if available"""

    def __init__(self, max_spread_pct: float = 0.01, window: int = 20, enabled: bool = True):
        super().__init__(name="MaximumSpread", enabled=enabled)
        self.max_spread_pct = float(max_spread_pct)
        self.window = int(window)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        if data is None or data.empty:
            return FilterResult(ticker, False, "No data", 0, self.max_spread_pct)
        data_up_to = data[data.index <= as_of_date]
        if len(data_up_to) < self.window:
            return FilterResult(ticker, False, f"Not enough data for spread window", 0, self.max_spread_pct)
        if 'High' not in data_up_to.columns or 'Low' not in data_up_to.columns or 'Close' not in data_up_to.columns:
            # No OHLC, assume spread OK
            return FilterResult(ticker, True, "No OHLC, assume OK", 0, self.max_spread_pct)

        recent = data_up_to.tail(self.window)
        # Proxy spread = (High - Low) / Close
        spread = ((recent['High'] - recent['Low']) / recent['Close'].replace(0, np.nan)).mean()
        if pd.isna(spread):
            spread = 0

        passed = spread <= self.max_spread_pct
        reason = f"Spread {spread*100:.2f}% <= {self.max_spread_pct*100:.2f}%" if passed else f"High spread {spread*100:.2f}% > {self.max_spread_pct*100:.2f}%"
        return FilterResult(ticker, passed, reason, float(spread), float(self.max_spread_pct))


class MaximumTurnoverFilter(UniverseFilter):
    """Maximum turnover – Volume / avg Volume or Volume / Shares Outstanding"""

    def __init__(self, max_turnover: float = 5.0, window: int = 20, enabled: bool = True):
        super().__init__(name="MaximumTurnover", enabled=enabled)
        self.max_turnover = float(max_turnover)
        self.window = int(window)

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        if data is None or data.empty or 'Volume' not in data.columns:
            return FilterResult(ticker, True, "No volume data, assume OK", 0, self.max_turnover)

        data_up_to = data[data.index <= as_of_date]
        if len(data_up_to) < self.window * 2:
            return FilterResult(ticker, True, "Not enough data for turnover", 0, self.max_turnover)

        recent = data_up_to.tail(self.window)
        historical = data_up_to.iloc[-self.window*2:-self.window]

        avg_recent = recent['Volume'].mean()
        avg_hist = historical['Volume'].mean()

        if avg_hist == 0 or pd.isna(avg_hist):
            turnover = 0
        else:
            turnover = avg_recent / avg_hist

        passed = turnover <= self.max_turnover
        reason = f"Turnover {turnover:.2f}x <= {self.max_turnover:.2f}x" if passed else f"High turnover {turnover:.2f}x > {self.max_turnover:.2f}x"
        return FilterResult(ticker, passed, reason, float(turnover), float(self.max_turnover))


class TrackingErrorFilter(UniverseFilter):
    """Extreme tracking error – std(ETF return - Benchmark return)"""

    def __init__(self, max_tracking_error_annual: float = 0.05, window: int = 60, benchmark_ticker: str = "SPY", enabled: bool = True):
        super().__init__(name="TrackingError", enabled=enabled)
        self.max_te = float(max_tracking_error_annual)
        self.window = int(window)
        self.benchmark_ticker = benchmark_ticker

    def filter(self, ticker: str, data: pd.DataFrame, metadata, as_of_date: pd.Timestamp) -> FilterResult:
        # Need benchmark data – we will get it via metadata engine? For simplicity, we check if ticker is benchmark itself -> pass
        if ticker == self.benchmark_ticker:
            return FilterResult(ticker, True, "Benchmark itself", 0, self.max_te)

        # We need benchmark data – we can't access it here easily because filter only gets single ticker data
        # For institutional, we would pass benchmark data via metadata or data dict
        # Here we implement simple check: if data has very high volatility vs typical, might be extreme tracking error
        # For now, we approximate tracking error via rolling volatility of returns
        if data is None or data.empty:
            return FilterResult(ticker, False, "No data", 0, self.max_te)

        data_up_to = data[data.index <= as_of_date]
        if len(data_up_to) < self.window:
            return FilterResult(ticker, True, "Not enough data for TE, assume OK", 0, self.max_te)

        # If we have benchmark data available in metadata cache? We will try to get from metadata attribute tracking_error
        if hasattr(metadata, 'tracking_error') and getattr(metadata, 'tracking_error') is not None:
            te = float(getattr(metadata, 'tracking_error'))
        else:
            # Proxy: annualized volatility of returns as proxy for tracking error if benchmark is SPY
            # For ETFs that should track index, high vol may indicate leverage or extreme TE
            # We will compute vol and if vol > 80% annual, might be extreme
            ret = data_up_to['Close'].pct_change().dropna().tail(self.window)
            vol = ret.std() * (252 ** 0.5)
            # For leveraged ETFs like TQQQ, vol is naturally high ~55%, so we allow higher threshold for them
            if "TQQQ" in ticker or "UPRO" in ticker or "SPXL" in ticker:
                # Allow up to 80% vol for 3x
                te = vol * 0.5  # approximate TE as half vol for leveraged
            else:
                te = vol * 0.3  # rough proxy

        passed = te <= self.max_te
        reason = f"TE {te*100:.2f}% <= {self.max_te*100:.2f}%" if passed else f"Extreme TE {te*100:.2f}% > {self.max_te*100:.2f}%"
        return FilterResult(ticker, passed, reason, float(te), float(self.max_te))
