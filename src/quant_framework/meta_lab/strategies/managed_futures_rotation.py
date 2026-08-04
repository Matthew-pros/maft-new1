
"""
Managed Futures Rotation – DBMF, KMLM, or trend following on futures
Economic: Managed futures risk premium, time series momentum, crisis alpha, alternative
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult
class ManagedFuturesRotation(StrategyPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__(name="Managed_Futures_Rotation", enabled=enabled, weight=weight)
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="Managed Futures no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        for col in close_wide.columns:
            ma200 = close_wide[col].ewm(span=200, adjust=False).mean()
            trend_up = close_wide[col] > ma200
            signals.loc[trend_up, col] = 1
        trend_count = (close_wide.gt(close_wide.ewm(span=200, adjust=False).mean())).sum(axis=1)
        for date in close_wide.index:
            count = trend_count.loc[date]
            if count > len(close_wide.columns)*0.6:
                for t in ["DBMF","KMLM","SPY"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.33
            else:
                for t in ["BIL","TLT"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.5
        signals = signals.shift(1).fillna(0)
        weights = weights.div(weights.sum(axis=1).replace(0,1), axis=0).shift(1).fillna(0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.65, expected_return=0.09, expected_vol=0.13, hit_rate=0.54, turnover=0.3, economic_justification="Managed Futures Rotation – time series momentum, crisis alpha, alternative risk premium, low correlation to equities, positive skew via trend following.")
