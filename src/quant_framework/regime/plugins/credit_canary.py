
from typing import Dict
import pandas as pd
import numpy as np
from .base import CanaryPlugin, CanaryResult
class CreditSpreadCanary(CanaryPlugin):
    def __init__(self, enabled: bool = True, weight: float = 1.2, threshold: float = -0.012):
        super().__init__(name="CreditSpreadCanary", enabled=enabled, weight=weight)
        self.threshold = float(threshold)
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        if "HYG" not in data or "IEF" not in data:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.75, turnover=0.08, value=0.0)
        hyg = data["HYG"]["Close"].dropna()
        ief = data["IEF"]["Close"].dropna()
        if len(hyg)<20 or len(ief)<20:
            return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.75, turnover=0.08, value=0.0)
        h_ret = hyg.iloc[-1]/hyg.iloc[-21]-1
        i_ret = ief.iloc[-1]/ief.iloc[-21]-1
        ratio = h_ret - i_ret
        if ratio < self.threshold:
            signal="RiskOff"; conf=0.85; prob=0.85
        elif ratio > 0.01:
            signal="RiskOn"; conf=0.75; prob=0.75
        else:
            signal="Neutral"; conf=0.4; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.78, turnover=0.07, value=float(ratio), timestamp=hyg.index[-1])
