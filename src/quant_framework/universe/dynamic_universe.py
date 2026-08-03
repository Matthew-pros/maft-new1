"""
Dynamic Universe Engine – institutional, time-varying, survivorship bias protected

- Universe changes over time
- Each day automatically filters delisted, low liquidity, short history, extreme tracking error
- Filters: Minimum history, Minimum AUM, Minimum volume, Maximum spread, Maximum turnover, Tracking error
- Each ticker contains IPO date, Delisting date, Historical availability
- Backtest automatically filters by date
- Reports: Universe size over time, Survivorship bias score
- No lookahead bias, deterministic, reproducible
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from ..config.settings import UniverseConfig
from ..data.asset_metadata import AssetMetadataEngine
from ..data.data_manager import DataManager
from .filters import (
    UniverseFilter, MinimumHistoryFilter, DelistedFilter,
    MinimumVolumeFilter, MinimumAUMFilter, MaximumSpreadFilter,
    MaximumTurnoverFilter, TrackingErrorFilter, FilterResult
)


@dataclass(frozen=True)
class UniverseSnapshot:
    date: pd.Timestamp
    tickers: List[str]
    filtered_out: Dict[str, List[str]]  # filter_name -> list of tickers filtered out with reason
    size: int

    def to_dict(self):
        return {
            "date": str(self.date),
            "tickers": self.tickers,
            "size": self.size,
            "filtered_out": self.filtered_out
        }


class DynamicUniverseManager:
    """
    Dynamic Universe Manager – universe changes over time, survivorship bias protected
    - Each ticker has IPO date, delisting date, historical availability
    - Backtest filters by date automatically
    - Reports universe size over time and survivorship bias score
    """

    def __init__(self,
                 config: UniverseConfig,
                 metadata_engine: AssetMetadataEngine,
                 data_manager: DataManager,
                 filters: Optional[List[UniverseFilter]] = None):
        if not isinstance(config, UniverseConfig):
            raise TypeError("config must be UniverseConfig")
        if not isinstance(metadata_engine, AssetMetadataEngine):
            raise TypeError("metadata_engine must be AssetMetadataEngine")
        if not isinstance(data_manager, DataManager):
            raise TypeError("data_manager must be DataManager")

        self.config = config
        self.metadata_engine = metadata_engine
        self.data_manager = data_manager

        # Default filters – institutional
        if filters is None:
            self.filters: List[UniverseFilter] = [
                DelistedFilter(grace_days=30, enabled=True),
                MinimumHistoryFilter(min_history_days=252, enabled=True),
                MinimumVolumeFilter(min_avg_volume=100000, min_notional=1_000_000, window=20, enabled=True),
                MinimumAUMFilter(min_aum=50_000_000, enabled=True),
                MaximumSpreadFilter(max_spread_pct=0.01, window=20, enabled=True),
                MaximumTurnoverFilter(max_turnover=5.0, window=20, enabled=True),
                TrackingErrorFilter(max_tracking_error_annual=0.05, window=60, benchmark_ticker=config.benchmark, enabled=True),
            ]
        else:
            if not isinstance(filters, list) or not all(isinstance(f, UniverseFilter) for f in filters):
                raise TypeError("filters must be list of UniverseFilter")
            self.filters = sorted(filters, key=lambda x: x.name)

        # Cache for performance
        self._universe_cache: Dict[pd.Timestamp, UniverseSnapshot] = {}

    def get_universe_at_date(self, as_of_date: str, verbose: bool = False) -> UniverseSnapshot:
        """
        Get universe filtered as of specific date – point-in-time, no lookahead bias

        Logic:
        - Only include tickers where as_of_date >= IPO date and (delisting_date is None or as_of_date <= delisting_date)
        - Apply all enabled filters sequentially
        - Each filter uses only data up to as_of_date (no future)

        Args:
            as_of_date: Date string YYYY-MM-DD or pd.Timestamp
            verbose: Whether to print filtered reasons

        Returns:
            UniverseSnapshot with tickers that pass all filters as of date

        Raises:
            TypeError, ValueError
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)
        if not isinstance(as_of_date, pd.Timestamp):
            raise TypeError("as_of_date must be str or Timestamp")

        # Normalize to UTC and midnight for determinism
        as_of_date = as_of_date.tz_localize(None) if as_of_date.tzinfo is None else as_of_date.tz_convert(None)
        as_of_date = pd.to_datetime(as_of_date.date())

        # Check cache
        cache_key = pd.to_datetime(as_of_date.date())
        if cache_key in self._universe_cache:
            return self._universe_cache[cache_key]

        # Get all tickers from config (static superset)
        all_tickers = self.config.tickers

        # Get metadata for all tickers
        all_metadata = self.metadata_engine.get_all_metadata(all_tickers)

        # Get data for all tickers (should already be downloaded via DataManager)
        # For efficiency, we use data_manager._data_store which has data
        data_store = self.data_manager._data_store

        filtered_tickers = []
        filtered_out: Dict[str, List[str]] = {f.name: [] for f in self.filters}

        for ticker in sorted(all_tickers):
            meta = all_metadata.get(ticker)
            if meta is None:
                # No metadata -> skip
                filtered_out.setdefault("NoMetadata", []).append(ticker)
                continue

            # Get data for ticker
            df = data_store.get(ticker)
            if df is None or df.empty:
                # No data as of now, might be not yet listed or delisted and no cache
                # Try to check if as_of_date is before first available data in future? We skip
                filtered_out.setdefault("NoData", []).append(ticker)
                continue

            # Check historical availability – data up to as_of_date must exist
            # IPO date check: first trading day
            first_day = meta.listing_date or df.index.min()
            if isinstance(first_day, str):
                first_day = pd.to_datetime(first_day)
            if first_day.tzinfo is not None:
                first_day = first_day.tz_localize(None)
            # Delisting date check
            delisting_date = getattr(meta, 'delisting_date', None)
            if delisting_date:
                if isinstance(delisting_date, str):
                    delisting_date = pd.to_datetime(delisting_date)
                if delisting_date.tzinfo is not None:
                    delisting_date = delisting_date.tz_localize(None)

            # Historical availability: as_of_date must be >= first_day and (delisting_date is None or as_of_date <= delisting_date)
            if as_of_date < pd.to_datetime(first_day).tz_localize(None) if pd.to_datetime(first_day).tzinfo is None else first_day.tz_localize(None):
                # Before IPO
                filtered_out.setdefault("BeforeIPO", []).append(ticker)
                continue

            if delisting_date and as_of_date > pd.to_datetime(delisting_date):
                filtered_out.setdefault("Delisted", []).append(ticker)
                continue

            # Apply each enabled filter sequentially, no lookahead (filter uses data up to as_of_date)
            passed_all = True
            for filt in self.filters:
                if not filt.is_enabled():
                    continue
                try:
                    result: FilterResult = filt.filter(ticker, df, meta, as_of_date)
                    if not result.passed:
                        filtered_out[filt.name].append(f"{ticker} ({result.reason})")
                        passed_all = False
                        if verbose:
                            print(f"{as_of_date.date()} {ticker} filtered by {filt.name}: {result.reason}")
                        break  # fail fast
                except Exception as e:
                    # On filter error, be conservative and filter out
                    filtered_out[filt.name].append(f"{ticker} (filter error {e})")
                    passed_all = False
                    break

            if passed_all:
                filtered_tickers.append(ticker)

        # Deterministic sorted output
        filtered_tickers = sorted(filtered_tickers)

        snapshot = UniverseSnapshot(
            date=as_of_date,
            tickers=filtered_tickers,
            filtered_out={k: sorted(v) for k, v in sorted(filtered_out.items())},
            size=len(filtered_tickers)
        )

        # Cache
        self._universe_cache[cache_key] = snapshot
        return snapshot

    def get_universe_over_time(self, start: str, end: str, freq: str = "B") -> pd.DataFrame:
        """
        Get universe over time – DataFrame with dates index, tickers columns, bool whether included

        Args:
            start: Start date
            end: End date
            freq: Frequency for sampling, e.g., B (business daily), W (weekly), ME (month end)

        Returns:
            DataFrame index dates, columns tickers, values bool (True if included as of date)

        Raises:
            ValueError
        """
        if not isinstance(start, str) or not isinstance(end, str):
            raise TypeError("start and end must be str")
        dates = pd.date_range(start=start, end=end, freq=freq)
        # For each date, get universe
        # For efficiency, we can iterate and build matrix
        all_tickers = sorted(self.config.tickers)
        matrix = pd.DataFrame(False, index=dates, columns=all_tickers)

        for date in dates:
            snap = self.get_universe_at_date(date)
            for ticker in snap.tickers:
                if ticker in matrix.columns:
                    matrix.loc[date, ticker] = True

        return matrix.sort_index().sort_index(axis=1)

    def get_universe_size_over_time(self, start: str, end: str, freq: str = "B") -> pd.Series:
        """
        Universe size over time – number of tickers passing filters each day

        Args:
            start: Start date
            end: End date
            freq: Frequency

        Returns:
            Series index dates, value size

        Raises:
            ValueError
        """
        over_time = self.get_universe_over_time(start, end, freq)
        size_series = over_time.sum(axis=1)
        size_series.name = "universe_size"
        return size_series.sort_index()

    def get_survivorship_bias_score(self, start: str, end: str, benchmark_ticker: str = "SPY") -> Dict:
        """
        Calculate survivorship bias score

        Definition:
        - Static universe = all tickers in config (current survivors, includes only currently alive)
        - Dynamic PIT universe = tickers that actually existed and passed filters as of each date
        - Survivorship bias occurs if static universe includes tickers that didn't exist at that time (future IPO) or excludes delisted that did exist

        Score calculation:
        - Universe size difference: average (static_size - dynamic_size) / static_size * 100%
        - Return difference: backtest equal-weight returns of static vs dynamic PIT and compare CAGR
        - Bias Score = (CAGR_static - CAGR_dynamic) / |CAGR_dynamic| * 100% if dynamic CAGR !=0 else 0
        - If bias > 5%, significant survivorship bias

        Args:
            start: Start date
            end: End date
            benchmark_ticker: Benchmark for comparison (not used for bias but for context)

        Returns:
            Dict with metrics: static_size, dynamic_avg_size, size_diff_pct, static_cagr, dynamic_cagr, bias_score_pct, bias_interpretation

        Raises:
            ValueError
        """
        if not isinstance(start, str) or not isinstance(end, str):
            raise TypeError("start and end must be str")

        # Static universe size = len(config.tickers)
        static_tickers = sorted(self.config.tickers)
        static_size = len(static_tickers)

        # Dynamic over time
        over_time = self.get_universe_over_time(start, end, freq="B")
        dynamic_avg_size = float(over_time.sum(axis=1).mean()) if not over_time.empty else 0.0
        size_diff = static_size - dynamic_avg_size
        size_diff_pct = (size_diff / static_size * 100) if static_size != 0 else 0.0

        # Calculate equal-weight returns for static vs dynamic if data available
        # For each date, static portfolio = all static tickers equal weight (if data exists), dynamic = PIT filtered
        data_store = self.data_manager._data_store
        # Build daily returns for each ticker
        returns_dict = {}
        for ticker in static_tickers:
            df = data_store.get(ticker)
            if df is not None and not df.empty and 'Close' in df.columns:
                close = df['Close'].sort_index()
                # Filter to date range
                close_range = close[(close.index >= pd.to_datetime(start)) & (close.index <= pd.to_datetime(end))]
                if not close_range.empty:
                    ret = close_range.pct_change().dropna()
                    returns_dict[ticker] = ret

        if not returns_dict:
            # No data, return size-based score only
            return {
                "static_universe_size": static_size,
                "dynamic_avg_size": dynamic_avg_size,
                "size_diff": size_diff,
                "size_diff_pct": round(size_diff_pct, 2),
                "static_cagr": None,
                "dynamic_cagr": None,
                "bias_score_pct": None,
                "bias_interpretation": f"Size diff {size_diff_pct:.1f}% indicates {'high' if abs(size_diff_pct)>10 else 'low'} survivorship bias (no returns data)",
                "method": "size_diff_only"
            }

        # Align returns
        returns_df = pd.DataFrame(returns_dict).sort_index().dropna(how='all')
        if returns_df.empty:
            return {
                "static_universe_size": static_size,
                "dynamic_avg_size": dynamic_avg_size,
                "size_diff_pct": round(size_diff_pct, 2),
                "bias_score_pct": None,
                "interpretation": "No returns data"
            }

        # Static equal weight: each day equal weight among all tickers that have data that day (ignores delisting/IPO)
        # For true static, we use all tickers equally weighted, but if data missing, we use available
        static_weights = {t: 1 / len(returns_df.columns) for t in returns_df.columns}
        static_port_ret = (returns_df * pd.Series(static_weights)).sum(axis=1)

        # Dynamic PIT: for each date, use universe as of that date
        dynamic_rets = []
        for date in returns_df.index:
            # Get PIT universe as of date
            snap = self.get_universe_at_date(date)
            pit_tickers = [t for t in snap.tickers if t in returns_df.columns]
            if not pit_tickers:
                dynamic_rets.append(0.0)
                continue
            # Equal weight among PIT tickers
            w = 1 / len(pit_tickers)
            day_ret = returns_df.loc[date, pit_tickers].mean() if pit_tickers else 0.0
            # Actually equal weight mean
            dynamic_rets.append(float(day_ret) if not pd.isna(day_ret) else 0.0)

        dynamic_port_ret = pd.Series(dynamic_rets, index=returns_df.index)

        # Calculate CAGR for both
        def calc_cagr(ret_series):
            if ret_series.empty:
                return 0.0
            equity = (1 + ret_series).cumprod()
            total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0
            years = (ret_series.index[-1] - ret_series.index[0]).days / 365.25
            if years <= 0:
                return 0.0
            cagr = (1 + total_ret) ** (1 / years) - 1
            return float(cagr)

        static_cagr = calc_cagr(static_port_ret)
        dynamic_cagr = calc_cagr(dynamic_port_ret)

        if dynamic_cagr != 0:
            bias_score = (static_cagr - dynamic_cagr) / abs(dynamic_cagr) * 100
        else:
            bias_score = 0.0

        interpretation = "Low survivorship bias"
        if abs(bias_score) > 20:
            interpretation = "Very high survivorship bias – static universe significantly overstates returns"
        elif abs(bias_score) > 10:
            interpretation = "High survivorship bias"
        elif abs(bias_score) > 5:
            interpretation = "Moderate survivorship bias"
        elif abs(size_diff_pct) > 10:
            interpretation = f"High size bias {size_diff_pct:.1f}% difference, but return bias low"

        return {
            "static_universe_size": static_size,
            "dynamic_avg_size": round(dynamic_avg_size, 2),
            "size_diff": round(size_diff, 2),
            "size_diff_pct": round(size_diff_pct, 2),
            "static_cagr": round(static_cagr * 100, 2),
            "dynamic_cagr": round(dynamic_cagr * 100, 2),
            "bias_score_pct": round(bias_score, 2),
            "bias_interpretation": interpretation,
            "method": "size_and_return_diff",
            "start": start,
            "end": end
        }

    def clear_cache(self) -> None:
        """Clear internal cache for fresh recomputation"""
        self._universe_cache.clear()
