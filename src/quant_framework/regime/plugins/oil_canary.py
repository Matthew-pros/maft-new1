
from typing import Dict
import pandas as pd
from .base import CanaryPlugin, CanaryResult
class OilCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=1.0):
        super().__init__(name="OilCanary", enabled=enabled, weight=weight)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "XLE" in data and "SPY" in data:
            xle=data["XLE"]["Close"].dropna(); spy=data["SPY"]["Close"].dropna()
            if len(xle)>=20 and len(spy)>=20:
                rx=xle.iloc[-1]/xle.iloc[-21]-1; rs=spy.iloc[-1]/spy.iloc[-21]-1; ratio=rx-rs
                if ratio>0.015:
                    signal="Inflation"; conf=0.7; prob=0.7
                elif ratio<-0.015:
                    signal="Deflation"; conf=0.6; prob=0.6
                else:
                    signal="Neutral"; conf=0.3; prob=0.5
                return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.67, turnover=0.16, value=float(ratio), timestamp=xle.index[-1])
        return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.65, turnover=0.16, value=0.0)
