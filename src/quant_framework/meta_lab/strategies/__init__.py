"""Strategy plugins for Meta Lab"""
from .base import StrategyPlugin, StrategyResult
from .rsi_rotation import RSIRotation
from .momentum_rotation import MomentumRotation
from .golden_cross import GoldenCross
from .xlu_spy import XLU_SPY_Rotation
from .qqq_spy import QQQ_SPY_Rotation
from .credit_rotation import CreditRotation
from .vix_rotation import VIXRotation
from .managed_futures_rotation import ManagedFuturesRotation
from .volatility_rotation import VolatilityRotation
from .correlation_rotation import CorrelationRotation
from .overnight_drift import OvernightDrift
from .attention_stocks import AttentionStocks
from .mean_reversion_intraday import MeanReversionIntraday
from .hedging_system import HedgingSystem
from .pairs_trading import PairsTrading

__all__ = [
    "StrategyPlugin", "StrategyResult",
    "RSIRotation", "MomentumRotation", "GoldenCross",
    "XLU_SPY_Rotation", "QQQ_SPY_Rotation", "CreditRotation",
    "VIXRotation", "ManagedFuturesRotation", "VolatilityRotation", "CorrelationRotation",
    "OvernightDrift", "AttentionStocks", "MeanReversionIntraday", "HedgingSystem", "PairsTrading"
]
