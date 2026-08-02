"""Regime plugins package"""
from .base import CanaryPlugin, CanaryResult
from .xlu_canary import XLUCanary
from .qqq_canary import QQQCanary
from .credit_canary import CreditSpreadCanary
from .consumer_canary import ConsumerCanary
from .vix_canary import VIXCanary
from .yield_curve_canary import YieldCurveCanary
from .dollar_canary import DollarCanary
from .oil_canary import OilCanary
from .gold_canary import GoldCanary

__all__ = [
    "CanaryPlugin", "CanaryResult",
    "XLUCanary", "QQQCanary", "CreditSpreadCanary", "ConsumerCanary",
    "VIXCanary", "YieldCurveCanary", "DollarCanary", "OilCanary", "GoldCanary"
]
