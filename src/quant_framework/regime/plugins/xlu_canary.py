
from typing import Dict
import pandas as pd
import numpy as np
from .base import CanaryPlugin, CanaryResult
class XLUCanary(CanaryPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.0, threshold: float = 0.005):
        super().__init__(name="XLUCanary", enabled=enabled, weight=weight)
        self.threshold = float(threshold)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "XLU" not in data or "SPY" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.65, turnover=0.1, value=0.0)
        xlu = data["XLU"]["Close"].dropna()
        spy = data["SPY"]["Close"].dropna()
        if len(xlu) < 20 or len(spy) < 20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.65, turnover=0.1, value=0.0)
        xlu_ret = xlu.iloc[-1] / xlu.iloc[-21] - 1 if len(xlu) > 21 else 0
        spy_ret = spy.iloc[-1] / spy.iloc[-21] - 1 if len(spy) > 21 else 0
        ratio = xlu_ret - spy_ret
        if ratio > self.threshold:
            signal = "RiskOff"; confidence = min(0.9, 0.5 + ratio * 10); prob = 0.8 if ratio > 0.01 else 0.65
        elif ratio < -self.threshold:
            signal = "RiskOn"; confidence = min(0.9, 0.5 + abs(ratio) * 10); prob = 0.7
        else:
            signal = "Neutral"; confidence = 0.3; prob = 0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(np.clip(confidence, 0, 1)), probability=float(np.clip(prob, 0, 1)), historical_accuracy=0.72, turnover=0.15, value=float(ratio), timestamp=xlu.index[-1])
