"""
05 Regime Engine – plugin system, each rule separate class, extensible
"""

from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

from .plugins.base import CanaryPlugin, CanaryResult
from ..config.settings import RegimeConfig


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    confidence: float
    probability: float
    contributing_canaries: List[CanaryResult]
    timestamp: Optional[pd.Timestamp]


class RegimeEngine:
    def __init__(self, config: RegimeConfig, plugins: List[CanaryPlugin]):
        if not isinstance(config, RegimeConfig):
            raise TypeError("config must be RegimeConfig")
        if not isinstance(plugins, list) or not all(isinstance(p, CanaryPlugin) for p in plugins):
            raise TypeError("plugins must be list of CanaryPlugin")
        self.config = config
        enabled_names = set(config.enabled_plugins)
        self.plugins = [p for p in plugins if p.name in enabled_names and p.is_enabled()]
        self.plugins = sorted(self.plugins, key=lambda x: x.name)

    def evaluate(self, data: Dict[str, pd.DataFrame]) -> RegimeResult:
        if not isinstance(data, dict):
            raise TypeError("data must be dict")
        canary_results: List[CanaryResult] = []
        for plugin in self.plugins:
            try:
                res = plugin.evaluate(data)
                canary_results.append(res)
            except Exception:
                continue

        risk_on_score = 0.0
        risk_off_score = 0.0
        black_swan_score = 0.0
        inflation_score = 0.0

        for cr in canary_results:
            w = next((p.weight for p in self.plugins if p.name == cr.name), 1.0)
            weighted = cr.confidence * w * cr.historical_accuracy
            if cr.signal == "RiskOn":
                risk_on_score += weighted
            elif cr.signal in ("RiskOff", "Defensive"):
                risk_off_score += weighted
            elif cr.signal == "BlackSwan":
                black_swan_score += weighted * 2
            elif cr.signal in ("Inflation", "SafeHaven"):
                inflation_score += weighted

        if black_swan_score > 0.5 or any(cr.signal == "BlackSwan" and cr.confidence > 0.8 for cr in canary_results):
            regime = "BLACK_SWAN_VOL_SHOCK"
            conf = min(1.0, black_swan_score)
            prob = 0.9
        elif risk_off_score > 1.0 and any(cr.name == "CreditSpreadCanary" and cr.signal == "RiskOff" for cr in canary_results):
            regime = "CREDIT_STRESS"
            conf = min(1.0, risk_off_score / 2)
            prob = 0.8
        elif any(cr.name == "XLUCanary" and cr.signal == "RiskOff" for cr in canary_results) and \
             any(cr.name == "ConsumerCanary" and cr.signal == "RiskOff" for cr in canary_results):
            regime = "DEFENSIVE_ROTATION"
            conf = 0.75
            prob = 0.7
        elif any(cr.name == "VIXCanary" and cr.signal == "Neutral" and 18 <= cr.value < 25 for cr in canary_results):
            regime = "RANGE_BOUND_WHIPSAW"
            conf = 0.65
            prob = 0.6
        elif inflation_score > 0.6:
            regime = "INFLATION_GROWTH"
            conf = min(1.0, inflation_score)
            prob = 0.7
        else:
            if risk_on_score >= risk_off_score:
                regime = "RISK_ON_GROWTH"
                conf = min(1.0, 0.5 + risk_on_score / 3)
                prob = 0.7
            else:
                regime = "DEFENSIVE_ROTATION"
                conf = 0.6
                prob = 0.6

        latest_ts = None
        for cr in canary_results:
            if cr.timestamp and (latest_ts is None or cr.timestamp > latest_ts):
                latest_ts = cr.timestamp

        return RegimeResult(
            regime=regime,
            confidence=float(conf),
            probability=float(prob),
            contributing_canaries=sorted(canary_results, key=lambda x: x.name),
            timestamp=latest_ts
        )

    def list_plugins(self) -> List[str]:
        return sorted([p.name for p in self.plugins])
