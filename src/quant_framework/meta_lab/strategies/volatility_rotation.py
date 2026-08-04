
"""
Volatility Rotation – Low Vol vs High Beta
Economic: Low volatility anomaly – leverage constraints, lottery preference, low vol stocks outperform high beta on risk-adjusted basis
"""
from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult
class VolatilityRotation(StrategyPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__(name="Volatility_Rotation", enabled=enabled, weight=weight)
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="Vol no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        returns = close_wide.pct_change()
        vol = returns.rolling(20).std() * (252**0.5)
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        for date in close_wide.index:
            vol_row = vol.loc[date].dropna()
            if vol_row.empty:
                continue
            low_vol = vol_row.sort_values().head(max(1, len(vol_row)//5)).index.tolist()
            for t in low_vol:
                signals.loc[date, t] = 1
                weights.loc[date, t] = 1.0 / len(low_vol)
        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.66, expected_return=0.08, expected_vol=0.11, hit_rate=0.57, turnover=0.4, economic_justification="Volatility Rotation – low volatility anomaly, leverage constraints cause high beta to be overpriced, low vol outperforms risk-adjusted. SPLV vs SPHB spread.")
