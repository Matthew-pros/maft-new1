
from typing import Dict
import pandas as pd
import numpy as np
from .base import CanaryPlugin, CanaryResult
class VIXCanary(CanaryPlugin):
    def __init__(self, enabled=True, weight=1.5, risk_on_thresh=18, risk_off_thresh=25, black_swan_thresh=30):
        super().__init__(name="VIXCanary", enabled=enabled, weight=weight)
        self.risk_on_thresh=risk_on_thresh; self.risk_off_thresh=risk_off_thresh; self.black_swan_thresh=black_swan_thresh
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        vix_val=None; ts=None
        for key in ["^VIX","VIX","VIXY"]:
            if key in data and not data[key].empty:
                vix_val=float(data[key]["Close"].iloc[-1]); ts=data[key].index[-1]; break
        if vix_val is None:
            if "SPY" in data and len(data["SPY"])>20:
                ret=data["SPY"]["Close"].pct_change().dropna().tail(20)
                vix_val=float(ret.std()*np.sqrt(252)*100); ts=data["SPY"].index[-1]
            else:
                return CanaryResult(name=self.name, signal="Neutral", confidence=0.0, probability=0.5, historical_accuracy=0.80, turnover=0.05, value=0.0)
        if vix_val>=self.black_swan_thresh:
            signal="BlackSwan"; conf=0.95; prob=0.9
        elif vix_val>=self.risk_off_thresh:
            signal="RiskOff"; conf=0.8; prob=0.75
        elif vix_val<=self.risk_on_thresh:
            signal="RiskOn"; conf=0.75; prob=0.7
        else:
            signal="Neutral"; conf=0.4; prob=0.5
        return CanaryResult(name=self.name, signal=signal, confidence=float(conf), probability=float(prob), historical_accuracy=0.82, turnover=0.06, value=float(vix_val), timestamp=ts)
