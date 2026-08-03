"""
Golden Cross – EMA50 cross above EMA200
Economic: Long-term institutional bull/bear line, 50-day = quarterly earnings cycle, 200-day = yearly macro regime. Golden Cross signals long-term capital rotation into asset.
"""

from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult


class GoldenCross(StrategyPlugin):
    def __init__(self, fast: int = 50, slow: int = 200, enabled: bool = True, weight: float = 1.0):
        super().__init__(name="Golden_Cross", enabled=enabled, weight=weight)
        self.fast = int(fast)
        self.slow = int(slow)

    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {}
        for ticker, df in data.items():
            if not df.empty and 'Close' in df.columns:
                closes[ticker] = df['Close']
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(),
                                  confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0,
                                  economic_justification="Golden Cross – no data")

        close_wide = pd.DataFrame(closes).sort_index().ffill()

        ema_fast = close_wide.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close_wide.ewm(span=self.slow, adjust=False).mean()

        # Golden Cross when fast crosses above slow
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        # For each ticker, detect cross
        for col in close_wide.columns:
            fast = ema_fast[col]
            slow = ema_slow[col]
            cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            # Hold long while fast > slow
            holding = (fast > slow).astype(int)
            # Signal 1 when holding, but entry signal when cross_up
            signals[col] = holding

        # Shift to avoid lookahead
        signals = signals.shift(1).fillna(0)

        # Equal weight among those in golden cross
        weights = signals.div(signals.sum(axis=1).replace(0, 1), axis=0).fillna(0)

        return StrategyResult(
            name=self.name,
            signals=signals,
            weights=weights,
            confidence=0.65,
            expected_return=0.10,
            expected_vol=0.15,
            hit_rate=0.55,
            turnover=0.2,
            economic_justification="Golden Cross 50>200 – institutional bull market definition, slow-moving capital, yearly macro regime. When price above 200EMA, long-term trend up, low turnover, robust."
        )
