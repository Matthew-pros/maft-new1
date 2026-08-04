"""
Institutional-Grade Quantitative Research Framework
Version: 1.0.0
Architecture: 00 Config -> 12 Reporting pipeline, fully decoupled modules
"""
__version__ = "1.0.0"
__author__ = "Quant Engineer"

from .config.settings import ResearchConfig

__all__ = ["ResearchConfig"]
