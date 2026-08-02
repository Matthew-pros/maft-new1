
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class CanaryResult:
    name: str
    signal: str
    confidence: float
    probability: float
    historical_accuracy: float
    turnover: float
    value: float
    timestamp: Optional[pd.Timestamp] = None
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0,1]")
        if not 0 <= self.historical_accuracy <= 1:
            raise ValueError("historical_accuracy must be in [0,1]")
    def to_dict(self) -> Dict:
        return {"name": self.name, "signal": self.signal, "confidence": self.confidence,
                "probability": self.probability, "historical_accuracy": self.historical_accuracy,
                "turnover": self.turnover, "value": self.value,
                "timestamp": str(self.timestamp) if self.timestamp else None}

class CanaryPlugin(ABC):
    def __init__(self, name: str, enabled: bool = True, weight: float = 1.0):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be non-empty str")
        self.name = name
        self.enabled = bool(enabled)
        self.weight = float(weight)
        if self.weight <= 0:
            raise ValueError("weight must be >0")
    @abstractmethod
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> CanaryResult:
        raise NotImplementedError
    def is_enabled(self) -> bool:
        return self.enabled
    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled}, weight={self.weight})"
