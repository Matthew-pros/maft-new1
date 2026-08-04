"""
Hedging System – reduces drawdowns during high volatility equity crashes
Economic: From TradeQuantiX – handful of systems specifically for hedging in high vol environments. On its own not interesting, but as portfolio component reduces drawdowns when long-only systems hate most. Long volatility or long BIL in high vol.

No single market environment hurts all systems at same time when combined with momentum, trend, mean reversion, hedging.
"""

from typing import Dict
import pandas as pd
from .base import StrategyPlugin, StrategyResult


class HedgingSystem(StrategyPlugin):
    def __init__(self, vix_threshold: int = 25, enabled: bool = True, weight: float = 0.8):
        super().__init__(name="Hedging_System", enabled=enabled, weight=weight)
        self.vix_threshold = int(vix_threshold)

    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        closes = {t: df['Close'] for t, df in data.items() if not df.empty and 'Close' in df.columns}
        if not closes:
            return StrategyResult(name=self.name, signals=pd.DataFrame(), weights=pd.DataFrame(),
                                  confidence=0.0, expected_return=0.0, expected_vol=0.0, hit_rate=0.0, turnover=0.0,
                                  economic_justification="Hedging no data")

        close_wide = pd.DataFrame(closes).sort_index().ffill()
        signals = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
        weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)

        # VIX proxy
        vix_series = None
        for key in ["^VIX", "VIX", "VIXY"]:
            if key in close_wide.columns:
                vix_series = close_wide[key]
                break
        if vix_series is None and "SPY" in close_wide.columns:
            # Realized vol proxy
            ret = close_wide["SPY"].pct_change()
            vix_series = ret.rolling(20).std() * (252**0.5) * 100

        if vix_series is None:
            # No VIX, assume low vol -> no hedge
            return StrategyResult(name=self.name, signals=signals, weights=weights,
                                  confidence=0.5, expected_return=0.02, expected_vol=0.05, hit_rate=0.6, turnover=0.2,
                                  economic_justification="Hedging – no VIX data, assume no hedge needed")

        for date in close_wide.index:
            vix_val = vix_series.loc[date] if date in vix_series.index else 20
            if vix_val > self.vix_threshold:
                # High vol – hedge: long BIL, TLT, GLD, VIXY
                for t in ["BIL", "TLT", "GLD", "VIXY"]:
                    if t in close_wide.columns:
                        signals.loc[date, t] = 1
                        weights.loc[date, t] = 0.25
            else:
                # Low vol – no hedge, flat
                pass

        signals = signals.shift(1).fillna(0)
        weights = weights.shift(1).fillna(0)

        return StrategyResult(
            name=self.name,
            signals=signals,
            weights=weights,
            confidence=0.75,
            expected_return=0.02,  # low return on its own, but reduces drawdown
            expected_vol=0.05,
            hit_rate=0.60,
            turnover=0.2,
            economic_justification="Hedging System – long volatility or long BIL/TLT/GLD in high vol environments. On its own equity curve not interesting, loses small money in calm markets. As portfolio component, reduces drawdowns during equity crashes. No single environment hurts all systems at same time. Diversification benefit, Tail risk hedge."
        )
