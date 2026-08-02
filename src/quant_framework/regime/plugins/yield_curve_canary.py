
from typing import Dict
import pandas as pd
from .base import CanaryPlugin, CanaryResult
class YieldCurveCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=1.0):
        super().__init__(name="YieldCurveCanary", enabled=enabled, weight=weight)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "TLT" not in data or "IEF" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.65, turnover=0.1, value=0.0)
        tlt=data["TLT"]["Close"].dropna(); ief=data["IEF"]["Close"].dropna()
        if len(tlt)<20 or len(ief)<20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.65, turnover=0.1, value=0.0)
        r_tlt=tlt.iloc[-1]/tlt.iloc[-21]-1; r_ief=ief.iloc[-1]/ief.iloc[-21]-1; ratio=r_tlt - r_ief
        if ratio>0.01:
            signal="RiskOff"; conf=0.65; prob=0.65
        elif ratio<-0.01:
            signal="RiskOn"; conf=0.6; prob=0.6
        else:
            signal="Neutral"; conf=0.3; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.66, turnover=0.11, value=float(ratio), timestamp=tlt.index[-1])
