
"""
Correlation Rotation – low correlation assets get higher weight
Economic: Diversification benefit, low correlation = more diversification, correlation breakdown in crisis
"""
from typing import Dict
import pandas as pd
import numpy as np
from .base import StrategyPlugin, StrategyResult
class CorrelationRotation(StrategyPlugin):
    def __init__(self, window: int = 60, enabled: bool = True, weight: float = 1.0):
        super().__init__(name="Correlation_Rotation", enabled=enabled, weight=weight)
        self.window = int(window)
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(), confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0, economic_justification="Corr no data")
        close_wide = pd.DataFrame(closes).sort_index().ffill()
        returns = close_wide.pct_change()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
        for date in close_wide.index:
            end_idx = close_wide.index.get_loc(date)
            start_idx = max(0, end_idx - self.window)
            window_rets = returns.iloc[start_idx:end_idx+1]
            if len(window_rets) < 20:
                continue
            corr_matrix = window_rets.corr()
            avg_corr = {}
            for col in corr_matrix.columns:
                corr_vals = corr_matrix[col].drop(col).dropna()
                avg_corr[col] = corr_vals.mean() if not corr_vals.empty else 0
            avg_corr_series = pd.Series(avg_corr).dropna()
            if avg_corr_series.empty:
                continue
            low_corr = avg_corr_series.sort_values().head(max(1, len(avg_corr_series)//3)).index.tolist()
            for t in low_corr:
                signals.loc[date, t] = 1
                weights.loc[date, t] = 1.0 / len(low_corr)
        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)
        return StrategyResult(name=self.name, signals=signals, weights=weights, confidence=0.64, expected_return=0.07, expected_vol=0.12, hit_rate=0.53, turnover=0.5, economic_justification="Correlation Rotation – low average correlation assets provide more diversification, correlation breakdown in crisis, low correlation premium.")
