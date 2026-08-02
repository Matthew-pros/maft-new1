
from typing import Dict
import pandas as pd
import numpy as np
from .base import CanaryPlugin, CanaryResult
class ConsumerCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=1.0, threshold=0.005):
        super().__init__(name="ConsumerCanary", enabled=enabled, weight=weight)
        self.threshold=threshold
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "XLY" not in data or "XLP" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.66, turnover=0.12, value=0.0)
        xly=data["XLY"]["Close"].dropna(); xlp=data["XLP"]["Close"].dropna()
        if len(xly)<20 or len(xlp)<20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.66, turnover=0.12, value=0.0)
        r1=xly.iloc[-1]/xly.iloc[-21]-1; r2=xlp.iloc[-1]/xlp.iloc[-21]-1; ratio=r1-r2
        if ratio>self.threshold:
            signal="RiskOn"; conf=0.7; prob=0.7
        elif ratio<-self.threshold:
            signal="RiskOff"; conf=0.7; prob=0.7
        else:
            signal="Neutral"; conf=0.3; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.69, turnover=0.14, value=float(ratio), timestamp=xly.index[-1])
