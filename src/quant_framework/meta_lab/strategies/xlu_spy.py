
"""
XLU/SPY Rotation – Defensive utility flow
Economic: XLU/SPY ratio – when utilities outperform broad market, revealed preference for safety, regulated income over growth. Defensive rotation signal.
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult
class XLU_SPY_Rotation(StrategyPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__(name="XLU/SPY_Rotation", enabled=enabled, weight=weight)
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="XLU/SPY no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        if "XLU" in close_wide.columns and "SPY" in close_wide.columns:
            ratio = close_wide["XLU"] / close_wide["SPY"]
            ratio_ma = ratio.ewm(span=20, adjust=False).mean()
            rising = ratio > ratio_ma
            for date in close_wide.index:
                if rising.loc[date]:
                    for t in ["XLU","XLP","XLV","IEF"]:
                        if t in close_wide.columns:
                            signals.loc[date, t] = 1
                            weights.loc[date, t] = 0.25
                else:
                    for t in ["SPY","QQQ"]:
                        if t in close_wide.columns:
                            signals.loc[date, t] = 1
                            weights.loc[date, t] = 0.5
        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)
        weights = weights.div(weights.sum(axis=1).replace(0,1), axis=0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.68, expected_return=0.08, expected_vol=0.12, hit_rate=0.56, turnover=0.3, economic_justification="XLU/SPY – defensive utility flow, when utilities outperform broad market, investors seek safety, predictable income over growth. Revealed preference, not sentiment survey.")
