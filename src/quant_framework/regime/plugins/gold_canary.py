
from typing import Dict
import pandas as pd
from .base import CanaryPlugin, CanaryResult
class GoldCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=1.0):
        super().__init__(name="GoldCanary", enabled=enabled, weight=weight)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "GLD" not in data or "SPY" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.64, turnover=0.14, value=0.0)
        gld=data["GLD"]["Close"].dropna(); spy=data["SPY"]["Close"].dropna()
        if len(gld)<20 or len(spy)<20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.64, turnover=0.14, value=0.0)
        rg=gld.iloc[-1]/gld.iloc[-21]-1; rs=spy.iloc[-1]/spy.iloc[-21]-1; ratio=rg-rs
        if ratio>0.015:
            signal="SafeHaven"; conf=0.7; prob=0.7
        elif ratio<-0.015:
            signal="RiskOn"; conf=0.6; prob=0.6
        else:
            signal="Neutral"; conf=0.3; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.68, turnover=0.15, value=float(ratio), timestamp=gld.index[-1])
