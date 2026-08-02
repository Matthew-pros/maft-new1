
from typing import Dict
import pandas as pd
from .base import CanaryPlugin, CanaryResult
class DollarCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=0.8):
        super().__init__(name="DollarCanary", enabled=enabled, weight=weight)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "UUP" not in data or "SPY" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.60, turnover=0.12, value=0.0)
        uup=data["UUP"]["Close"].dropna(); spy=data["SPY"]["Close"].dropna()
        if len(uup)<20 or len(spy)<20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.60, turnover=0.12, value=0.0)
        ru=uup.iloc[-1]/uup.iloc[-21]-1; rs=spy.iloc[-1]/spy.iloc[-21]-1; ratio=ru - rs
        if ratio>0.01:
            signal="RiskOff"; conf=0.6; prob=0.6
        elif ratio<-0.01:
            signal="RiskOn"; conf=0.55; prob=0.55
        else:
            signal="Neutral"; conf=0.3; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.62, turnover=0.13, value=float(ratio), timestamp=uup.index[-1])
