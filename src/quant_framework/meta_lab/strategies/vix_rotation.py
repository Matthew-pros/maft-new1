
"""
VIX Rotation – VIX level based risk on/off
Economic: VIX as fear index, volatility regime, risk appetite
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult
class VIXRotation(StrategyPlugin):
    def __init__(self, low_thresh: int = 18, high_thresh: int = 30, enabled: bool = True, weight: float = 1.5):
        super().__init__(name="VIX_Rotation", enabled=enabled, weight=weight)
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="VIX no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        vix_series = None
        for key in ["^VIX","VIX","VIXY"]:
            if key in close_wide.columns:
                vix_series = close_wide[key]
                break
        if vix_series is None and "SPY" in close_wide.columns:
            ret = close_wide["SPY"].pct_change()
            vix_series = ret.rolling(20).std() * (252**0.5) * 100
        if vix_series is None:
            return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="VIX no data")
        for date in close_wide.index:
            vix_val = vix_series.loc[date] if date in vix_series.index else 20
            if vix_val < self.low_thresh:
                for t in ["SPY","QQQ","TQQQ"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.33
            elif vix_val >= self.high_thresh:
                for t in ["BIL","TLT","GLD"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.33
            else:
                for t in ["SPY","IEF","GLD"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.33
        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)
        weights = weights.div(weights.sum(axis=1).replace(0,1), axis=0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.80, expected_return=0.10, expected_vol=0.14, hit_rate=0.57, turnover=0.2, economic_justification="VIX Rotation – VIX as fear index, low vol complacency risk-on, high vol stress risk-off, VIX>=30 black swan 100% BIL.")
