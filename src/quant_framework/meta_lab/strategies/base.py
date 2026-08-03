"""
Base Strategy Plugin – institutional
Each strategy is a plugin, independent, testable, reusable, economically justified
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class StrategyResult:
    """Immutable result from a strategy plugin"""
    name: str
    signals: pd.DataFrame  # index dates, columns tickers, values signal -1,0,1 or weight
    weights: pd.DataFrame  # index dates, columns tickers, values weight 0-1
    confidence: float  # 0-1 overall confidence of strategy
    expected_return: float  # annualized expected return
    expected_vol: float  # annualized vol
    hit_rate: float  # historical win rate
    turnover: float  # average turnover
    economic_justification: str
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self):
        return {
            "name": self.name,
            "signals_shape": self.signals.shape if self.signals is not None else None,
            "weights_shape": self.weights.shape if self.weights is not None else None,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "expected_vol": self.expected_vol,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "economic_justification": self.economic_justification,
            "timestamp": str(self.timestamp) if self.timestamp else None
        }


class StrategyPlugin(ABC):
    """Abstract base for all strategy plugins – modular, reusable, no lookahead"""

    def __init__(self, name: str, enabled: bool = True, weight: float = 1.0):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be non-empty str")
        self.name = name
        self.enabled = bool(enabled)
        self.weight = float(weight)
        if self.weight < 0:
            raise ValueError("weight must be >=0")

    @abstractmethod
    def generate(self, data: Dict[str, pd.DataFrame]) -> StrategyResult:
        """
        Generate signals and weights from market data – no lookahead, past only

        Args:
            data: Dict ticker -> OHLCV DataFrame

        Returns:
            StrategyResult immutable

        Economic justification must be in result
        """
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_weight(self, weight: float) -> None:
        if weight < 0:
            raise ValueError("weight must be >=0")
        self.weight = float(weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled}, weight={self.weight})"
