"""Research Engine – automated SSRN / Google Scholar alpha harvesting"""
from .paper import Paper
from .research_engine import ResearchEngine
from .strategy_generator import StrategyGenerator, StrategySpec
from .multicharts_generator import MultiChartsGenerator

__all__ = ["Paper", "ResearchEngine", "StrategyGenerator", "StrategySpec", "MultiChartsGenerator"]
