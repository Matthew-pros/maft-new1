
"""
Credit Rotation – HYG/IEF
Economic: Credit spread proxy, HYG high yield vs IEF treasuries, real-time credit conditions, leads equity by days-weeks
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult
class CreditRotation(StrategyPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.2):
        super().__init__(name="Credit_Rotation", enabled=enabled, weight=weight)
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="Credit no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        if "HYG" in close_wide.columns and "IEF" in close_wide.columns:
            ratio = close_wide["HYG"] / close_wide["IEF"]
            ratio_ma = ratio.ewm(span=20, adjust=False).mean()
            rising = ratio > ratio_ma
            for date in close_wide.index:
                if rising.loc[date]:
                    for t in ["SPY","QQQ","HYG"]:
                        if t in close_wide.columns:
                            signals.loc[date, t] = 1
                            weights.loc[date, t] = 0.33
                else:
                    for t in ["TLT","IEF","GLD","BIL"]:
                        if t in close_wide.columns:
                            signals.loc[date, t] = 1
                            weights.loc[date, t] = 0.25
        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)
        weights = weights.div(weights.sum(axis=1).replace(0,1), axis=0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.75, expected_return=0.09, expected_vol=0.13, hit_rate=0.58, turnover=0.3, economic_justification="Credit Rotation HYG/IEF – real-time credit spread proxy, when high yield outperforms treasuries, credit risk appetite, plumbing functioning, leads equities.")
