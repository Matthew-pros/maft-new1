
from typing import Dict
import pandas as pd
import numpy as np
from .base import CanaryPlugin, CanaryResult
class QQQCanary(CanaryPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.0, threshold: float = 0.005):
        super().__init__(name="QQQCanary", enabled=enabled, weight=weight)
        self.threshold = float(threshold)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "QQQ" not in data or "SPY" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.68, turnover=0.12, value=0.0)
        qqq = data["QQQ"]["Close"].dropna()
        spy = data["SPY"]["Close"].dropna()
        if len(qqq) < 20 or len(spy) < 20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.68, turnover=0.12, value=0.0)
        q_ret = qqq.iloc[-1]/qqq.iloc[-21]-1 if len(qqq)>21 else 0
        s_ret = spy.iloc[-1]/spy.iloc[-21]-1 if len(spy)>21 else 0
        ratio = q_ret - s_ret
        if ratio > self.threshold:
            signal="RiskOn"; conf=min(0.9,0.5+ratio*8); prob=0.75
        elif ratio < -self.threshold:
            signal="RiskOff"; conf=min(0.9,0.5+abs(ratio)*8); prob=0.65
        else:
            signal="Neutral"; conf=0.3; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(np.clip(conf,0,1)), probability=float(np.clip(prob,0,1)), historical_accuracy=0.70, turnover=0.18, value=float(ratio), timestamp=qqq.index[-1])
