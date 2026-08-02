"""
03 Feature Engineering – Feature Store
- All features computed only once, stored as DataFrame
- Feature registry, lazy evaluated, no double compute
- Deterministic, type hints, docstring, validation
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from enum import Enum


class FeatureRegistry(Enum):
    MOMENTUM = "momentum"
    RSI = "rsi"
    ROLLING_CAGR = "rolling_cagr"
    ROLLING_VOLATILITY = "rolling_volatility"
    ROLLING_BETA = "rolling_beta"
    ROLLING_CORRELATION = "rolling_correlation"
    ROLLING_MAX_DRAWDOWN = "rolling_max_drawdown"
    ATR = "atr"
    ADX = "adx"
    MOVING_AVERAGE = "moving_average"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    RELATIVE_STRENGTH = "relative_strength"
    SECTOR_RELATIVE_STRENGTH = "sector_relative_strength"
    Z_SCORE = "z_score"
    ROLLING_SHARPE = "rolling_sharpe"
    ROLLING_SORTINO = "rolling_sortino"
    ROLLING_CALMAR = "rolling_calmar"
    PERCENTILE_RANK = "percentile_rank"
    CROSS_SECTIONAL_RANK = "cross_sectional_rank"


def _validate_price_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be DataFrame")
    if df.empty:
        raise ValueError("df empty")
    if 'Close' not in df.columns:
        raise ValueError("df must contain Close")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be DatetimeIndex")


def momentum(close: pd.Series, window: int = 252) -> pd.Series:
    """Momentum = Close / Close.shift(window) -1, economic = herding, trend persistence"""
    if window <= 0:
        raise ValueError("window must be >0")
    return close / close.shift(window) - 1


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI 0-100, economic = mean-reversion due to inventory pressure"""
    if window <= 0:
        raise ValueError("window must be >0")
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def rolling_cagr(close: pd.Series, window: int = 252) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be >1")
    total_ret = close / close.shift(window)
    years = window / 252
    return total_ret ** (1 / years) - 1 if years != 0 else total_ret - 1


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be >1")
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_beta(returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
    if window <= 2:
        raise ValueError("window must be >2")
    cov = returns.rolling(window).cov(benchmark_returns)
    var_bench = benchmark_returns.rolling(window).var()
    return cov / var_bench.replace(0, np.nan)


def rolling_correlation(returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
    if window <= 2:
        raise ValueError("window must be >2")
    return returns.rolling(window).corr(benchmark_returns)


def rolling_max_drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    def _mdd(s):
        peak = s.cummax()
        dd = s / peak - 1
        return dd.min()
    return close.rolling(window).apply(_mdd, raw=False)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be >0")
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be >0")
    up_move = high.diff()
    down_move = low.diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr_val.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def moving_average(close: pd.Series, window: int, ma_type: str = "sma") -> pd.Series:
    if window <= 0:
        raise ValueError("window must be >0")
    if ma_type == "sma":
        return close.rolling(window).mean()
    elif ma_type == "ema":
        return close.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("ma_type must be sma or ema")


def golden_cross(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    return (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))


def death_cross(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    return (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))


def relative_strength(close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    return close / benchmark_close.replace(0, np.nan)


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be >1")
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be >1")
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return mean / std.replace(0, np.nan) * np.sqrt(252)


def rolling_sortino(returns: pd.Series, window: int = 60) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be >1")
    def _sortino(s):
        mean = s.mean()
        downside = s[s < 0]
        ds = downside.std() if len(downside) > 1 else s.std()
        return mean / ds * np.sqrt(252) if ds and not np.isnan(ds) and ds != 0 else np.nan
    return returns.rolling(window).apply(_sortino, raw=False)


def rolling_calmar(close: pd.Series, window: int = 252) -> pd.Series:
    cagr = rolling_cagr(close, window)
    mdd = rolling_max_drawdown(close, window).abs().replace(0, np.nan)
    return cagr / mdd


def percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False)


def cross_sectional_rank(df_wide: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_wide, pd.DataFrame):
        raise TypeError("df_wide must be DataFrame")
    return df_wide.rank(axis=1, pct=True) * 100


class FeatureStore:
    """Lazy evaluated feature store – computed only once"""

    def __init__(self, price_data: Dict[str, pd.DataFrame], benchmark_ticker: str = "SPY"):
        if not isinstance(price_data, dict):
            raise TypeError("price_data must be dict")
        for ticker, df in price_data.items():
            if not isinstance(ticker, str):
                raise TypeError("ticker key must be str")
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Data for {ticker} must be DataFrame")
            if not df.empty:
                _validate_price_df(df)
        self.price_data = {k: v.sort_index() for k, v in sorted(price_data.items())}
        self.benchmark_ticker = benchmark_ticker
        self._feature_cache: Dict = {}
        self._compute_count: Dict[str, int] = {}

    def _cache_key(self, ticker: str, feature: str, params):
        return (ticker, feature, params)

    def _get_returns(self, ticker: str) -> pd.Series:
        df = self.price_data.get(ticker)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df['Close'].pct_change()

    def _get_benchmark_returns(self) -> pd.Series:
        if self.benchmark_ticker and self.benchmark_ticker in self.price_data:
            df = self.price_data[self.benchmark_ticker]
            return df['Close'].pct_change()
        if self.price_data:
            first = next(iter(self.price_data.values()))
            return pd.Series(0, index=first.index)
        return pd.Series(dtype=float)

    def get_feature(self, ticker: str, feature, window: int = 20, **kwargs):
        if ticker not in self.price_data:
            raise ValueError(f"Ticker {ticker} not in price_data")
        kw_items = tuple(sorted(kwargs.items()))
        key = self._cache_key(ticker, feature.value, (window,) + kw_items)
        if key in self._feature_cache:
            return self._feature_cache[key].copy()

        df = self.price_data[ticker]
        close = df['Close']
        high = df['High'] if 'High' in df.columns else close
        low = df['Low'] if 'Low' in df.columns else close
        returns = self._get_returns(ticker)
        bench_close = self.price_data.get(self.benchmark_ticker, {}).get('Close') if isinstance(self.price_data.get(self.benchmark_ticker), pd.DataFrame) else None
        if bench_close is None and self.benchmark_ticker in self.price_data:
            bench_close = self.price_data[self.benchmark_ticker]['Close']
        bench_ret = self._get_benchmark_returns()

        result_df = pd.DataFrame(index=df.index)

        if feature == FeatureRegistry.MOMENTUM:
            result_df['momentum'] = momentum(close, window)
        elif feature == FeatureRegistry.RSI:
            result_df['rsi'] = rsi(close, window)
        elif feature == FeatureRegistry.ROLLING_CAGR:
            result_df['rolling_cagr'] = rolling_cagr(close, window)
        elif feature == FeatureRegistry.ROLLING_VOLATILITY:
            result_df['rolling_vol'] = rolling_volatility(returns, window)
        elif feature == FeatureRegistry.ROLLING_BETA:
            if bench_close is not None:
                aligned_ret, aligned_bench = returns.align(bench_ret, join='inner')
                result_df['rolling_beta'] = rolling_beta(aligned_ret, aligned_bench, window)
            else:
                result_df['rolling_beta'] = np.nan
        elif feature == FeatureRegistry.ROLLING_CORRELATION:
            if bench_close is not None:
                aligned_ret, aligned_bench = returns.align(bench_ret, join='inner')
                result_df['rolling_corr'] = rolling_correlation(aligned_ret, aligned_bench, window)
            else:
                result_df['rolling_corr'] = np.nan
        elif feature == FeatureRegistry.ROLLING_MAX_DRAWDOWN:
            result_df['rolling_mdd'] = rolling_max_drawdown(close, window)
        elif feature == FeatureRegistry.ATR:
            result_df['atr'] = atr(high, low, close, window)
        elif feature == FeatureRegistry.ADX:
            result_df['adx'] = adx(high, low, close, window)
        elif feature == FeatureRegistry.MOVING_AVERAGE:
            ma_type = kwargs.get('ma_type', 'sma')
            result_df[f'{ma_type}_{window}'] = moving_average(close, window, ma_type)
        elif feature == FeatureRegistry.GOLDEN_CROSS:
            fast = kwargs.get('fast', 50)
            slow = kwargs.get('slow', 200)
            fast_ma = moving_average(close, fast, kwargs.get('ma_type', 'sma'))
            slow_ma = moving_average(close, slow, kwargs.get('ma_type', 'sma'))
            result_df['golden_cross'] = golden_cross(fast_ma, slow_ma)
        elif feature == FeatureRegistry.DEATH_CROSS:
            fast = kwargs.get('fast', 50)
            slow = kwargs.get('slow', 200)
            fast_ma = moving_average(close, fast, kwargs.get('ma_type', 'sma'))
            slow_ma = moving_average(close, slow, kwargs.get('ma_type', 'sma'))
            result_df['death_cross'] = death_cross(fast_ma, slow_ma)
        elif feature == FeatureRegistry.RELATIVE_STRENGTH:
            if bench_close is not None:
                aligned = pd.concat([close, bench_close], axis=1, join='inner')
                aligned.columns = ['close', 'bench']
                result_df['rs'] = aligned['close'] / aligned['bench'].replace(0, np.nan)
            else:
                result_df['rs'] = np.nan
        elif feature == FeatureRegistry.SECTOR_RELATIVE_STRENGTH:
            sector_ticker = kwargs.get('sector_ticker', 'XLK')
            sector_close = self.price_data.get(sector_ticker, {}).get('Close') if sector_ticker in self.price_data else None
            if sector_close is not None:
                aligned = pd.concat([close, sector_close], axis=1, join='inner')
                aligned.columns = ['close', 'sector']
                result_df['sector_rs'] = aligned['close'] / aligned['sector'].replace(0, np.nan)
            else:
                result_df['sector_rs'] = np.nan
        elif feature == FeatureRegistry.Z_SCORE:
            result_df['z_score'] = z_score(close, window)
        elif feature == FeatureRegistry.ROLLING_SHARPE:
            result_df['rolling_sharpe'] = rolling_sharpe(returns, window)
        elif feature == FeatureRegistry.ROLLING_SORTINO:
            result_df['rolling_sortino'] = rolling_sortino(returns, window)
        elif feature == FeatureRegistry.ROLLING_CALMAR:
            result_df['rolling_calmar'] = rolling_calmar(close, window)
        elif feature == FeatureRegistry.PERCENTILE_RANK:
            result_df['pct_rank'] = percentile_rank(close, window)
        elif feature == FeatureRegistry.CROSS_SECTIONAL_RANK:
            universe_wide = kwargs.get('universe_wide')
            if isinstance(universe_wide, pd.DataFrame) and ticker in universe_wide.columns:
                cs_rank = cross_sectional_rank(universe_wide)
                result_df['cs_rank'] = cs_rank[ticker]
            else:
                result_df['cs_rank'] = percentile_rank(close, window)
        else:
            raise ValueError(f"Unsupported feature: {feature}")

        self._feature_cache[key] = result_df.copy()
        self._compute_count[feature.value] = self._compute_count.get(feature.value, 0) + 1
        return result_df.copy()

    def get_cross_sectional_rank(self, feature: FeatureRegistry = FeatureRegistry.MOMENTUM, window: int = 20):
        wide_dict = {}
        for ticker in self.price_data.keys():
            try:
                fdf = self.get_feature(ticker, feature, window=window)
                col = fdf.columns[0]
                wide_dict[ticker] = fdf[col]
            except Exception:
                continue
        if not wide_dict:
            return pd.DataFrame()
        wide = pd.DataFrame(wide_dict).sort_index()
        ranked = wide.rank(axis=1, pct=True) * 100
        return ranked

    def cache_stats(self):
        return dict(self._compute_count)

    def clear_cache(self):
        self._feature_cache.clear()
        self._compute_count.clear()
