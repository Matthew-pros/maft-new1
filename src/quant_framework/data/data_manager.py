"""
01 Data – DataManager Institutional Level
- yfinance, local cache (parquet), auto re-download missing, validation tickers,
  duplicate removal, timezone normalization UTC, corporate actions auto_adjust=True
- coverage, missing, history length, first/last trading day, avg daily volume
- reproducible deterministic
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

TICKER_PATTERN = re.compile(r'^[A-Z0-9\.\-\^/=]+$')


@dataclass(frozen=True)
class CoverageRecord:
    ticker: str
    first_day: Optional[pd.Timestamp]
    last_day: Optional[pd.Timestamp]
    total_days: int
    expected_days: int
    coverage_pct: float
    missing_pct: float
    avg_volume: Optional[float]
    history_years: float


class DataManager:
    """Institutional DataManager – deterministic, no lookahead"""

    def __init__(self, cache_dir: Union[str, Path] = "data/cache", auto_adjust: bool = True,
                 timezone: str = "UTC", remove_duplicates: bool = True):
        if not isinstance(cache_dir, (str, Path)):
            raise TypeError("cache_dir must be str or Path")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_adjust = bool(auto_adjust)
        self.timezone = str(timezone)
        self.remove_duplicates = bool(remove_duplicates)
        self._data_store: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def validate_tickers(tickers: List[str]) -> List[str]:
        if not isinstance(tickers, list):
            raise TypeError("tickers must be list")
        cleaned = []
        seen = set()
        for raw in tickers:
            if not isinstance(raw, str):
                continue
            t = raw.strip().upper()
            if not t or len(t) > 20:
                continue
            if not TICKER_PATTERN.match(t):
                continue
            if t not in seen:
                seen.add(t)
                cleaned.append(t)
        if not cleaned:
            raise ValueError("No valid tickers after validation")
        return sorted(cleaned)

    def _cache_path(self, ticker: str) -> Path:
        safe = re.sub(r'[^A-Z0-9]', '_', ticker)
        h = hashlib.md5(ticker.encode()).hexdigest()[:8]
        return self.cache_dir / f"{safe}_{h}.parquet"

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if self.remove_duplicates:
            df = df[~df.index.duplicated(keep='last')]
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize(self.timezone)
            except Exception:
                df.index = pd.to_datetime(df.index).tz_localize('UTC')
        else:
            df.index = df.index.tz_convert(self.timezone)
        return df.sort_index()

    def download(self, tickers: List[str], start: str = "2010-01-01",
                 end: Optional[str] = None, interval: str = "1d",
                 use_cache: bool = True, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        if not HAS_YF and not use_cache:
            raise RuntimeError("yfinance not installed and cache disabled")
        validated = self.validate_tickers(tickers)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now(tz=self.timezone)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(self.timezone)

        result: Dict[str, pd.DataFrame] = {}
        for ticker in validated:
            cache_file = self._cache_path(ticker)
            cached_df = None
            if use_cache and cache_file.exists() and not force_refresh:
                try:
                    cached_df = pd.read_parquet(cache_file)
                    cached_df = self._normalize_timezone(cached_df)
                except Exception:
                    cached_df = None

            need_download = True
            dl_start = start_dt
            if cached_df is not None and not cached_df.empty and not force_refresh:
                c_first = cached_df.index.min()
                c_last = cached_df.index.max()
                if c_first <= start_dt and c_last >= end_dt - timedelta(days=1):
                    need_download = False
                    df_slice = cached_df.loc[(cached_df.index >= start_dt) & (cached_df.index <= end_dt)]
                    result[ticker] = df_slice
                    self._data_store[ticker] = df_slice
                    continue
                else:
                    dl_start = min(start_dt, c_first)

            if need_download:
                if not HAS_YF:
                    if cached_df is not None:
                        result[ticker] = cached_df
                        self._data_store[ticker] = cached_df
                        continue
                    else:
                        result[ticker] = pd.DataFrame()
                        continue
                try:
                    hist = yf.Ticker(ticker).history(start=dl_start, end=end_dt, interval=interval,
                                                     auto_adjust=self.auto_adjust, actions=True)
                    if hist.empty:
                        hist = yf.download(ticker, start=dl_start, end=end_dt, interval=interval,
                                           auto_adjust=self.auto_adjust, progress=False)
                    if not hist.empty:
                        hist = self._normalize_timezone(hist)
                        if cached_df is not None and not cached_df.empty:
                            combined = pd.concat([cached_df, hist])
                            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                        else:
                            combined = hist
                        if use_cache:
                            try:
                                combined.to_parquet(cache_file)
                            except Exception:
                                combined.to_csv(cache_file.with_suffix('.csv'))
                        final_df = combined.loc[(combined.index >= start_dt) & (combined.index <= end_dt)] if not combined.empty else combined
                        result[ticker] = final_df
                        self._data_store[ticker] = final_df
                    else:
                        if cached_df is not None:
                            result[ticker] = cached_df
                            self._data_store[ticker] = cached_df
                        else:
                            result[ticker] = pd.DataFrame()
                except Exception:
                    if cached_df is not None:
                        result[ticker] = cached_df
                        self._data_store[ticker] = cached_df
                    else:
                        result[ticker] = pd.DataFrame()

        return {k: result[k] for k in sorted(result.keys())}

    def get_coverage_report(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for ticker, df in sorted(self._data_store.items()):
            if df.empty:
                rows.append({"ticker": ticker, "first_day": None, "last_day": None, "total_days": 0,
                             "expected_days": 0, "coverage_pct": 0.0, "missing_pct": 100.0, "avg_volume": None, "history_years": 0.0})
                continue
            first = df.index.min(); last = df.index.max(); total = len(df)
            exp_start = pd.to_datetime(start) if start else first
            exp_end = pd.to_datetime(end) if end else last
            expected = len(pd.bdate_range(exp_start, exp_end))
            coverage = (total / expected * 100) if expected > 0 else 0
            missing = 100 - coverage
            avg_vol = float(df['Volume'].mean()) if 'Volume' in df.columns else None
            years = (last - first).days / 365.25 if first and last else 0
            rows.append({"ticker": ticker, "first_day": first, "last_day": last, "total_days": total,
                         "expected_days": expected, "coverage_pct": round(coverage, 2),
                         "missing_pct": round(missing, 2), "avg_volume": avg_vol, "history_years": round(years, 2)})
        return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    def get_missing_values_report(self) -> pd.DataFrame:
        rows = []
        for ticker, df in sorted(self._data_store.items()):
            if df.empty:
                rows.append({"ticker": ticker, "column": "ALL", "missing_count": 0, "missing_pct": 100.0, "total_rows": 0})
                continue
            total = len(df)
            for col in df.columns:
                missing = df[col].isna().sum()
                pct = (missing / total * 100) if total > 0 else 0
                rows.append({"ticker": ticker, "column": col, "missing_count": int(missing),
                             "missing_pct": round(float(pct), 2), "total_rows": total})
        return pd.DataFrame(rows).sort_values(["ticker", "column"]).reset_index(drop=True)

    def get_history_length_report(self) -> pd.DataFrame:
        cov = self.get_coverage_report()
        return cov[["ticker", "first_day", "last_day", "total_days", "history_years"]].sort_values("ticker")

    def get_trading_days_report(self) -> pd.DataFrame:
        rows = []
        for ticker, df in sorted(self._data_store.items()):
            if df.empty:
                rows.append({"ticker": ticker, "first_trading_day": None, "last_trading_day": None})
            else:
                rows.append({"ticker": ticker, "first_trading_day": df.index.min(), "last_trading_day": df.index.max()})
        return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    def get_volume_report(self) -> pd.DataFrame:
        rows = []
        for ticker, df in sorted(self._data_store.items()):
            if df.empty or 'Volume' not in df.columns:
                rows.append({"ticker": ticker, "avg_daily_volume": None, "max_volume": None, "min_volume": None})
            else:
                vol = df['Volume'].dropna()
                rows.append({"ticker": ticker,
                             "avg_daily_volume": float(vol.mean()) if not vol.empty else None,
                             "max_volume": float(vol.max()) if not vol.empty else None,
                             "min_volume": float(vol.min()) if not vol.empty else None})
        return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    def get_full_diagnostics(self) -> Dict[str, pd.DataFrame]:
        return {
            "coverage": self.get_coverage_report(),
            "missing_values": self.get_missing_values_report(),
            "history_length": self.get_history_length_report(),
            "trading_days": self.get_trading_days_report(),
            "volume": self.get_volume_report()
        }
