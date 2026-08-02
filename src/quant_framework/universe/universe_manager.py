"""
02 Universe – Universe Manager
Independent module, no business logic from other modules except config and data.
"""

from typing import List, Dict
from dataclasses import dataclass
import pandas as pd

from ..config.settings import UniverseConfig
from ..data.asset_metadata import AssetMetadataEngine


@dataclass(frozen=True)
class Universe:
    name: str
    tickers: List[str]
    benchmark: str
    metadata: Dict[str, object]


class UniverseManager:
    def __init__(self, config: UniverseConfig, metadata_engine: AssetMetadataEngine):
        if not isinstance(config, UniverseConfig):
            raise TypeError("config must be UniverseConfig")
        if not isinstance(metadata_engine, AssetMetadataEngine):
            raise TypeError("metadata_engine must be AssetMetadataEngine")
        self.config = config
        self.metadata_engine = metadata_engine

    def get_universe(self) -> Universe:
        tickers = self.config.tickers
        meta = self.metadata_engine.get_all_metadata(tickers)
        return Universe(name=self.config.name, tickers=tickers, benchmark=self.config.benchmark, metadata=meta)

    def filter_by_macro(self, macro_bucket: str) -> List[str]:
        if not isinstance(macro_bucket, str):
            raise TypeError("macro_bucket must be str")
        return self.metadata_engine.get_by_macro_bucket(macro_bucket, self.config.tickers)

    def filter_by_risk(self, risk_bucket: str) -> List[str]:
        if not isinstance(risk_bucket, str):
            raise TypeError("risk_bucket must be str")
        return self.metadata_engine.get_by_risk_bucket(risk_bucket, self.config.tickers)

    def get_by_asset_class(self, asset_class: str) -> List[str]:
        if not isinstance(asset_class, str):
            raise TypeError("asset_class must be str")
        all_meta = self.metadata_engine.get_all_metadata(self.config.tickers)
        filtered = [t for t, m in all_meta.items() if m.asset_class.lower() == asset_class.lower()]
        return sorted(filtered)
