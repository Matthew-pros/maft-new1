
"""
04 Signal Engine – pure functions / classes, deterministic, independent
"""

from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from ..features.feature_store import FeatureStore, FeatureRegistry

@dataclass(frozen=True)
class SignalResult:
    ticker: str
    date: pd.Timestamp
    signal: int
    confidence: float
    metadata: dict

class BaseSignal:
    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be non-empty str")
        self.name = name
    def generate(self, feature_store: FeatureStore, ticker: str) -> pd.DataFrame:
        raise NotImplementedError

class TEMACDSignal(BaseSignal):
    def __init__(self, len1: int = 5, len2: int = 50, len3: int = 222, fast: int = 61, slow: int = 70, sig: int = 15):
        super().__init__("TQQQ_TEMACD")
        if not all(isinstance(x, int) and x > 0 for x in [len1, len2, len3, fast, slow, sig]):
            raise ValueError("All lengths must be positive int")
        self.len1 = len1; self.len2 = len2; self.len3 = len3; self.fast = fast; self.slow = slow; self.sig = sig
    def generate(self, feature_store: FeatureStore, ticker: str) -> pd.DataFrame:
        if not isinstance(feature_store, FeatureStore):
            raise TypeError("feature_store must be FeatureStore")
        if ticker not in feature_store.price_data:
            raise ValueError(f"{ticker} not in feature_store")
        ema5 = feature_store.get_feature(ticker, FeatureRegistry.MOVING_AVERAGE, window=self.len1, ma_type='ema')
        ema50 = feature_store.get_feature(ticker, FeatureRegistry.MOVING_AVERAGE, window=self.len2, ma_type='ema')
        ema222 = feature_store.get_feature(ticker, FeatureRegistry.MOVING_AVERAGE, window=self.len3, ma_type='ema')
        df_price = feature_store.price_data[ticker]
        close = df_price['Close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        a_macd = macd.ewm(span=self.sig, adjust=False).mean()
        delta = macd - a_macd
        ema5_s = ema5.iloc[:, 0]
        ema50_s = ema50.iloc[:, 0]
        ema222_s = ema222.iloc[:, 0]
        delta_prev = delta.shift(1)
        e5_prev = ema5_s.shift(1)
        e50_prev = ema50_s.shift(1)
        e222_prev = ema222_s.shift(1)
        long_cond = ((delta > 0) & (delta_prev <= 0)) | ((ema5_s > ema50_s) & (e5_prev <= e50_prev)) | ((ema5_s > ema222_s) & (e5_prev <= e222_prev)) | ((ema50_s > ema222_s) & (e50_prev <= e222_prev))
        short_cond = ((delta < 0) & (delta_prev >= 0)) | ((ema5_s < ema50_s) & (e5_prev >= e50_prev)) | ((ema5_s < ema222_s) & (e5_prev >= e222_prev)) | ((ema50_s < ema222_s) & (e50_prev >= e222_prev))
        signals = pd.DataFrame(index=df_price.index)
        signals['signal'] = 0
        signals.loc[long_cond, 'signal'] = 1
        signals.loc[short_cond, 'signal'] = -1
        signals['confidence'] = 0.8
        signals.loc[long_cond, 'confidence'] = 0.9
        signals.loc[short_cond, 'confidence'] = 0.9
        return signals[['signal', 'confidence']]

class BTC_TEMACD_Signal(BaseSignal):
    def __init__(self):
        super().__init__("BTC_TEMACD")
        self.base = TEMACDSignal(len1=5, len2=64, len3=170, fast=32, slow=40, sig=94)
    def generate(self, feature_store: FeatureStore, ticker: str) -> pd.DataFrame:
        return self.base.generate(feature_store, ticker)

class SignalEngine:
    def __init__(self, feature_store: FeatureStore):
        if not isinstance(feature_store, FeatureStore):
            raise TypeError("feature_store must be FeatureStore")
        self.feature_store = feature_store
        self._signals: Dict[str, BaseSignal] = {}
    def register_signal(self, signal: BaseSignal) -> None:
        if not isinstance(signal, BaseSignal):
            raise TypeError("signal must be BaseSignal")
        self._signals[signal.name] = signal
    def generate_all(self, tickers: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        if not isinstance(tickers, list):
            raise TypeError("tickers must be list")
        tickers_sorted = sorted(list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()])))
        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        for ticker in tickers_sorted:
            if ticker not in self.feature_store.price_data:
                continue
            result[ticker] = {}
            for sig_name in sorted(self._signals.keys()):
                sig_obj = self._signals[sig_name]
                try:
                    df_sig = sig_obj.generate(self.feature_store, ticker)
                    result[ticker][sig_name] = df_sig
                except Exception:
                    idx = self.feature_store.price_data[ticker].index
                    result[ticker][sig_name] = pd.DataFrame({"signal": 0, "confidence": 0.0}, index=idx)
        return result
    def get_combined_signal(self, ticker: str, method: str = "or") -> pd.DataFrame:
        if ticker not in self.feature_store.price_data:
            raise ValueError(f"{ticker} not in data")
        all_sigs = []
        for sig_name in self._signals:
            df = self._signals[sig_name].generate(self.feature_store, ticker)
            all_sigs.append(df['signal'])
        if not all_sigs:
            idx = self.feature_store.price_data[ticker].index
            return pd.DataFrame({"signal": 0, "confidence": 0.0}, index=idx)
        import pandas as pd
        combined = pd.concat(all_sigs, axis=1)
        if method == "or":
            final = combined.apply(lambda row: 1 if (row == 1).any() else (-1 if (row == -1).any() else 0), axis=1)
        elif method == "and":
            final = combined.apply(lambda row: 1 if (row == 1).all() else (-1 if (row == -1).all() else 0), axis=1)
        elif method == "majority":
            final = combined.apply(lambda row: 1 if (row == 1).sum() > (row == -1).sum() else (-1 if (row == -1).sum() > (row == 1).sum() else 0), axis=1)
        else:
            raise ValueError("method must be or/and/majority")
        return pd.DataFrame({"signal": final, "confidence": 0.8}, index=combined.index)
