"""
00 Config – Institutional Research Configuration
Pure declarative config, no business logic. Deterministic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class DataConfig:
    cache_dir: Path = Path("data/cache")
    auto_adjust: bool = True
    auto_download_missing: bool = True
    timezone: str = "UTC"
    interval: str = "1d"
    start_date: str = "2010-01-01"
    end_date: str = ""
    max_retries: int = 3
    validate_tickers: bool = True
    remove_duplicates: bool = True
    corporate_actions: bool = True

    def __post_init__(self):
        if self.interval not in ["1d", "1wk", "1mo", "1h"]:
            raise ValueError(f"Invalid interval: {self.interval}")
        if not isinstance(self.cache_dir, Path):
            raise TypeError("cache_dir must be Path")


@dataclass(frozen=True)
class UniverseConfig:
    name: str = "quant_rick_30"
    tickers: List[str] = field(default_factory=lambda: [
        "TQQQ", "QQQ", "SPY", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
        "AVGO", "BRK.B", "LLY", "JPM", "V", "UNH", "NFLX", "COST", "AMD", "SMCI",
        "XLK", "XLV", "XLE", "XLF", "XLI", "XLB", "XLP", "XLU", "GLD", "IEF",
        "TLT", "BIL", "UUP", "VIXY", "SVXY", "MES=F", "BTC-USD"
    ])
    benchmark: str = "SPY"

    def __post_init__(self):
        if not self.tickers:
            raise ValueError("Universe tickers cannot be empty")
        object.__setattr__(self, 'tickers', sorted(list(dict.fromkeys([t.strip().upper() for t in self.tickers if t.strip()]))))


@dataclass(frozen=True)
class ExecutionConfig:
    commission_fixed: float = 0.0
    commission_pct: float = 0.0005
    slippage_bps: float = 2.0
    partial_fill_prob: float = 1.0
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "daily"
    cash_drag: float = 0.0
    allow_fractional: bool = True
    min_lot_size: int = 1
    execution_delay: int = 1
    execution_price: Literal["next_open", "next_close"] = "next_open"
    initial_capital: float = 100000.0

    def __post_init__(self):
        if self.commission_pct < 0 or self.commission_pct > 0.1:
            raise ValueError("commission_pct out of range")
        if self.min_lot_size < 1:
            raise ValueError("min_lot_size must be >=1")


@dataclass(frozen=True)
class PortfolioConfig:
    method: Literal["equal_weight", "inverse_vol", "risk_parity", "max_diversification", "min_variance", "vol_targeting", "kelly"] = "equal_weight"
    target_vol: float = 0.20
    kelly_fraction: float = 0.5
    max_position_weight: float = 0.40
    min_position_weight: float = 0.0
    cash_allocation: float = 0.0

    def __post_init__(self):
        if not 0 < self.target_vol < 2:
            raise ValueError("target_vol must be in (0,2)")
        if not 0 <= self.cash_allocation < 1:
            raise ValueError("cash_allocation must be in [0,1)")


@dataclass(frozen=True)
class RegimeConfig:
    enabled_plugins: List[str] = field(default_factory=lambda: [
        "XLUCanary", "QQQCanary", "CreditSpreadCanary", "ConsumerCanary",
        "VIXCanary", "YieldCurveCanary", "DollarCanary", "OilCanary", "GoldCanary"
    ])
    confidence_threshold: float = 0.6
    use_expert_system: bool = True

    def __post_init__(self):
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0,1]")


@dataclass(frozen=True)
class PerformanceConfig:
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    benchmark: str = "SPY"
    calculate_psr: bool = True
    calculate_dsr: bool = True
    dsr_trials: int = 10


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    random_seed: int = 42
    project_name: str = "TQQQ_TEMACD_Institutional"

    def __post_init__(self):
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be int")
        if not self.project_name:
            raise ValueError("project_name cannot be empty")

    def to_dict(self) -> Dict:
        return {
            "data": self.data.__dict__,
            "universe": self.universe.__dict__,
            "execution": self.execution.__dict__,
            "portfolio": self.portfolio.__dict__,
            "regime": self.regime.__dict__,
            "performance": self.performance.__dict__,
            "random_seed": self.random_seed,
            "project_name": self.project_name,
        }
