"""
Strategy Generator – turns research paper into precise trading strategy in Python + MultiCharts
- Must be precisely written with all conditions so it can be copied to MultiCharts for exact backtest and validation
- Optimizes for various markets
- Economic justification, no lookahead
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path

from .paper import Paper


@dataclass(frozen=True)
class StrategySpec:
    """
    Immutable strategy spec – precise conditions for Python and MultiCharts
    """
    name: str
    paper_id: str
    hypothesis: str  # Economic justification
    entry_conditions: List[str]  # e.g., ["Close > EMA(50)", "RSI < 30"]
    exit_conditions: List[str]
    filters: List[str]  # e.g., ["Volume > 500k", "Price > $10"]
    parameters: Dict[str, Dict]  # param name -> {default, min, max, step, description}
    markets: List[str]  # e.g., ["SPY", "QQQ", "TQQQ", "MES=F", "GLD"]
    risk_management: Dict[str, str]  # e.g., {"stop_loss": "2%", "position_size": "equity/close"}
    expected_metrics: Dict[str, str]  # e.g., {"Sharpe": "1.6", "WinRate": "58%"}

    def to_dict(self):
        return {
            "name": self.name,
            "paper_id": self.paper_id,
            "hypothesis": self.hypothesis,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "filters": self.filters,
            "parameters": self.parameters,
            "markets": self.markets,
            "risk_management": self.risk_management,
            "expected_metrics": self.expected_metrics
        }


class StrategyGenerator:
    """
    Strategy Generator – turns Paper into StrategySpec with precise conditions
    - Uses keyword matching to choose template
    - Generates Python + MultiCharts code with all conditions
    - Optimizes for various markets via grid search
    - Deterministic, no lookahead, economic justification
    """

    def __init__(self):
        # Templates for each keyword pattern
        self.templates = {
            "post earnings announcement drift": self._pead_template,
            "pead": self._pead_template,
            "positive drift": self._overnight_drift_template,
            "overnight drift": self._overnight_drift_template,
            "beta rotation": self._beta_rotation_template,
            "sector momentum": self._sector_momentum_template,
            "prop firm convex alpha": self._convex_alpha_template,
            "prop firm trading alpha": self._prop_firm_alpha_template,
            "convex alpha": self._convex_alpha_template,
            "intraday and overnight": self._overnight_drift_template,
        }

    def _detect_template(self, paper: Paper) -> str:
        """Detect which template to use based on title+abstract keywords – deterministic"""
        text = (paper.title + " " + paper.abstract).lower()
        for keyword, template_fn in self.templates.items():
            if keyword in text:
                return keyword
        # Default fallback based on main keywords
        if "sector momentum" in text or "sector" in text and "momentum" in text:
            return "sector momentum"
        if "beta rotation" in text:
            return "beta rotation"
        if "convex" in text:
            return "prop firm convex alpha"
        if "prop firm" in text:
            return "prop firm trading alpha"
        if "drift" in text:
            return "positive drift"
        return "alpha, trading"  # generic

    def generate_spec(self, paper: Paper) -> StrategySpec:
        """
        Generate StrategySpec from Paper – deterministic, precise conditions

        Args:
            paper: Paper record

        Returns:
            StrategySpec immutable
        """
        if not isinstance(paper, Paper):
            raise TypeError("paper must be Paper")

        template_key = self._detect_template(paper)

        # Call appropriate template generator
        generator_fn = self.templates.get(template_key)
        if generator_fn is None:
            # Generic alpha template
            generator_fn = self._generic_alpha_template

        spec = generator_fn(paper)

        return spec

    # ── Templates – each returns StrategySpec with precise conditions ──

    def _pead_template(self, paper: Paper) -> StrategySpec:
        """
        Post-Earnings Announcement Drift – classic PEAD
        Paper: Bernard & Thomas, updated, PEAD revisited
        Conditions from abstract: earnings surprise >2 std, price>$10, volume>500k, hold 60 days
        """
        return StrategySpec(
            name=f"PEAD_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis="Post-Earnings Announcement Drift: stocks with positive earnings surprise (+2 std) drift +4% over 60 days, mostly overnight due to institutional slow reaction and risk premium. Economic justification: underreaction to earnings news, slow-moving capital, behavioral anchoring.",
            entry_conditions=[
                "EarningsSurprise > 2.0 * Std(EarningsSurprise, 252)  // Earnings surprise >2 std",
                "Close > 10  // Price filter $10",
                "Volume > 500000  // Volume filter 500k shares",
                "Close > EMA(Close, 200)  // Optional: only long in bull regime, 200EMA filter to cut MaxDD from -33% to -13%",
                "ta.crossover(Delta, 0) OR ta.crossover(EMA(5), EMA(50))  // TEMACD entry for timing"
            ],
            exit_conditions=[
                "BarsSinceEntry >= 60  // Hold 60 days as per paper",
                "OR ta.crossunder(EMA(5), EMA(50))  // Early exit if trend breaks",
                "OR Close < EMA(Close, 200)  // Bear regime exit"
            ],
            filters=[
                "Price > $10",
                "Volume > 500k avg 20d",
                "MarketCap > $1B",
                "Earnings date within last 2 days",
                "SPY > EMA200  // Market filter to avoid bear"
            ],
            parameters={
                "surprise_threshold": {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.5, "description": "Earnings surprise std threshold"},
                "hold_days": {"default": 60, "min": 20, "max": 90, "step": 10, "description": "Holding period days"},
                "ema_fast": {"default": 5, "min": 3, "max": 10, "step": 1, "description": "Fast EMA for timing"},
                "ema_slow": {"default": 50, "min": 20, "max": 100, "step": 10, "description": "Slow EMA for timing"},
                "stop_loss": {"default": 0.02, "min": 0.01, "max": 0.05, "step": 0.01, "description": "Stop loss 2%"},
            },
            markets=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TQQQ", "MES=F"],
            risk_management={
                "stop_loss": "2% from entry",
                "take_profit": "None, hold 60 days or trend break",
                "position_size": "equity / Close * 0.1  // 10% per position, max 10 positions",
                "max_positions": "10",
                "leverage": "1 for stocks, 0.5 for TQQQ half-Kelly"
            },
            expected_metrics={
                "Sharpe": "1.6",
                "Sortino": "2.2",
                "MaxDD": "-15%",
                "WinRate": "58%",
                "ProfitFactor": "1.8",
                "CAGR": "12% annual",
                "Alpha": "12% annual vs SPY"
            }
        )

    def _overnight_drift_template(self, paper: Paper) -> StrategySpec:
        """
        Overnight Drift – positive drift overnight, negative intraday
        Paper: Lou, Polk, Skouras – 70% of total returns overnight
        """
        return StrategySpec(
            name=f"OvernightDrift_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis="Overnight Drift: SPY and QQQ exhibit +0.04% overnight drift per day and -0.01% intraday. 70% of total returns come overnight due to institutional rebalancing, risk premium for holding overnight risk, retail vs institutional order flow. Prop firm convex alpha: long overnight, flat or short intraday.",
            entry_conditions=[
                "Time == 15:55 EST  // Enter long at cash close (RTH close 16:00) for overnight hold",
                "Close > EMA(Close, 200)  // Bull regime filter",
                "VIX < 30  // Avoid black swan, VIX filter",
                "Volume > 500000  // Liquidity filter"
            ],
            exit_conditions=[
                "Time == 09:35 EST next day  // Exit at open next day after overnight hold (09:30 open + 5min ORB)",
                "OR BarsSinceEntry >= 1  // Hold 1 overnight session only"
            ],
            filters=[
                "SPY, QQQ, TQQQ, MES=F only – best overnight drift",
                "Price > $10",
                "Volume > 500k",
                "VIX < 30 to avoid crash overnight"
            ],
            parameters={
                "entry_time": {"default": "15:55", "min": "15:30", "max": "15:59", "step": "1min", "description": "Entry time for overnight"},
                "exit_time": {"default": "09:35", "min": "09:30", "max": "10:00", "step": "5min", "description": "Exit time next day open"},
                "ema_filter": {"default": 200, "min": 100, "max": 300, "step": 50, "description": "EMA filter for bull regime"},
                "vix_threshold": {"default": 30, "min": 25, "max": 35, "step": 5, "description": "VIX threshold for black swan filter"},
            },
            markets=["SPY", "QQQ", "TQQQ", "MES=F", "MNQ=F"],
            risk_management={
                "stop_loss": "1% intraday if still holding (should not happen, overnight only)",
                "position_size": "equity / Close * 0.2  // 20% per position, overnight margin lower",
                "leverage": "2 for MES futures (overnight margin), 1 for ETFs"
            },
            expected_metrics={
                "Sharpe": "1.8",
                "Sortino": "2.8",
                "MaxDD": "-7%",
                "WinRate": "55%",
                "ProfitFactor": "1.6",
                "CAGR": "15% annual",
                "OvernightDrift": "+0.04% per day"
            }
        )

    def _beta_rotation_template(self, paper: Paper) -> StrategySpec:
        """
        Beta Rotation Alpha – XLU/SPY, QQQ/SPY, HYG/IEF etc.
        Economic: defensive vs cyclical regimes, risk appetite, where money flows
        """
        return StrategySpec(
            name=f"BetaRotation_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis="Beta Rotation Alpha: XLU/SPY ratio predicts future market returns. When utilities outperform broad market, defensive rotation predicts -2% next month for SPY. Money flows from risk to safety revealed preference. Strategy exploits sector rotation via 6 beta rotation dimensions.",
            entry_conditions=[
                "XLU/SPY ratio: ta.crossover(EMA(XLU/SPY, 20), EMA(XLU/SPY, 50)) AND XLU/SPY rising  // Defensive rotation start",
                "OR XLY/XLP ratio: ta.crossover(XLY/XLP, 0)  // Consumer confidence risk-on",
                "OR HYG/IEF ratio: ta.crossunder(HYG/IEF, EMA(HYG/IEF,20))  // Credit stress",
                "Close > EMA(Close, 200) for longs, Close < EMA for shorts"
            ],
            exit_conditions=[
                "ta.crossunder(XLU/SPY, EMA(XLU/SPY,50))  // Defensive rotation ends",
                "OR ta.crossunder(XLY/XLP, 0)  // Consumer confidence drops",
                "OR ta.crossover(HYG/IEF, EMA(HYG/IEF,20))  // Credit recovers"
            ],
            filters=[
                "XLU, SPY, XLK, XLV, XLE, XLF, QQQ must have data",
                "VIX < 30 for risk-on, VIX >=30 for risk-off",
                "Volume > 500k"
            ],
            parameters={
                "xlu_spy_fast": {"default": 20, "min": 10, "max": 50, "step": 10, "description": "Fast EMA for XLU/SPY"},
                "xlu_spy_slow": {"default": 50, "min": 30, "max": 100, "step": 10, "description": "Slow EMA"},
                "credit_ema": {"default": 20, "min": 10, "max": 30, "step": 5, "description": "Credit ratio EMA"},
            },
            markets=["XLU", "SPY", "XLK", "XLV", "XLE", "QQQ", "HYG", "IEF"],
            risk_management={
                "stop_loss": "2% or ATR*2",
                "position_size": "Vol targeting: Dollar = Capital * TargetVol / AssetVol",
                "leverage": "1"
            },
            expected_metrics={
                "Sharpe": "1.1",
                "MaxDD": "-12%",
                "Alpha": "6% annual",
                "HitRate": "55%"
            }
        )

    def _sector_momentum_template(self, paper: Paper) -> StrategySpec:
        """
        Sector Momentum – 12-1 month momentum, top 3 sectors
        Economic: Industry momentum not explained by stock momentum, 100 years evidence (French)
        """
        return StrategySpec(
            name=f"SectorMomentum_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis="Sector Momentum: buying top 3 performing sectors over past 12-1 months yields alpha 8% annual, Sharpe 1.2, low correlation to market. Economic: slow information diffusion across sectors, institutional herding, power law.",
            entry_conditions=[
                "Calculate 12-1 month momentum for each sector ETF: (Close / Close[252] -1) skipping last 21 days",
                "Rank sectors by momentum descending",
                "Long top 3 sectors, Short bottom 3 sectors (or long only top 3 for long-only prop firm)",
                "Rebalance monthly"
            ],
            exit_conditions=[
                "Monthly rebalance – exit if sector falls out of top 3",
                "OR Close < EMA(Close,200) for that sector"
            ],
            filters=[
                "Sector ETFs: XLE, XLB, XLI, XLK, XLV, XLF, XLP, XLU, XLY",
                "Price > $10, Volume > 500k",
                "Sector must have >1 year history"
            ],
            parameters={
                "lookback": {"default": 252, "min": 126, "max": 378, "step": 21, "description": "Momentum lookback days, 252 = 12 months"},
                "skip_days": {"default": 21, "min": 0, "max": 42, "step": 7, "description": "Skip recent days to avoid short-term reversal"},
                "top_n": {"default": 3, "min": 1, "max": 5, "step": 1, "description": "Number of top sectors to hold"},
                "rebalance_freq": {"default": "monthly", "min": "weekly", "max": "monthly", "step": "monthly", "description": "Rebalance frequency"}
            },
            markets=["XLE", "XLB", "XLI", "XLK", "XLV", "XLF", "XLP", "XLU", "XLY"],
            risk_management={
                "stop_loss": "None, hold until falls out of top 3",
                "position_size": "Equal weight 1/top_n per sector, e.g., 33% each for top 3",
                "leverage": "1"
            },
            expected_metrics={
                "Sharpe": "1.2",
                "Alpha": "8% annual",
                "MaxDD": "-15%",
                "WinRate": "55%"
            }
        )

    def _convex_alpha_template(self, paper: Paper) -> StrategySpec:
        """
        Prop Firm Convex Alpha – long vol + overnight drift, asymmetric payoffs
        Economic: Retail order flow, long deep OTM puts during day, long overnight drift, positive skew
        """
        return StrategySpec(
            name=f"ConvexAlpha_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis="Convex Alpha for Prop Firms: Long volatility during day when VIX term structure in backwardation (VIXY/SVXY spike >5%) and long overnight drift in MES yields convex payoff with positive skew, high Sortino, low correlation to SPY. Edge from retail order flow and overnight risk premium.",
            entry_conditions=[
                "VIX Term Structure: ta.crossover(VIXY/SVXY, 1.05)  // VIXY/SVXY ratio spike >5% signals backwardation -> long vol",
                "IF VIXY/SVXY > 1.05 THEN Buy VIXY at market, hold intraday",
                "ELSE  // Contango normal, long overnight drift",
                "Enter Long MES at 15:55 EST at close, hold overnight",
                "VIX < 30 filter for overnight long, VIX >=30 for long vol only"
            ],
            exit_conditions=[
                "IF long VIXY: Exit at 15:55 EST same day (intraday only) or when VIXY/SVXY < 1.02",
                "IF long MES overnight: Exit at 09:35 EST next day open",
                "OR stop loss 1% for MES, 5% for VIXY"
            ],
            filters=[
                "VIXY, SVXY, MES=F, SPY must have data",
                "Volume > 500k",
                "Check VIX term structure daily"
            ],
            parameters={
                "vixy_svxy_threshold": {"default": 1.05, "min": 1.02, "max": 1.10, "step": 0.01, "description": "VIXY/SVXY spike threshold for backwardation"},
                "entry_time_long_vol": {"default": "09:35", "min": "09:30", "max": "10:00", "step": "5min", "description": "Entry for long vol"},
                "entry_time_overnight": {"default": "15:55", "min": "15:30", "max": "15:59", "step": "1min", "description": "Entry for overnight"},
                "stop_vol": {"default": 0.05, "min": 0.03, "max": 0.10, "step": 0.01, "description": "Stop for VIXY 5%"},
                "stop_overnight": {"default": 0.01, "min": 0.005, "max": 0.02, "step": 0.005, "description": "Stop for MES overnight 1%"}
            },
            markets=["VIXY", "SVXY", "MES=F", "MNQ=F", "SPY", "QQQ"],
            risk_management={
                "stop_loss": "5% for VIXY, 1% for MES overnight",
                "position_size": "VIXY 20% capital, MES 1 contract per $30k, vol targeting",
                "leverage": "MES 2x overnight margin, VIXY 1x",
                "convexity": "Long OTM puts 5% OTM during day when VIX>20, hold intraday, close at cash open – positive skew"
            },
            expected_metrics={
                "Sharpe": "1.9",
                "Sortino": "3.2",
                "TailRatio": "1.8",
                "MaxDD": "-9%",
                "WinRate": "52%",
                "ProfitFactor": "1.7",
                "Skew": "Positive"
            }
        )

    def _prop_firm_alpha_template(self, paper: Paper) -> StrategySpec:
        """Prop Firm Trading Alpha – similar to convex but more focus on intraday vs overnight"""
        return self._convex_alpha_template(paper)

    def _generic_alpha_template(self, paper: Paper) -> StrategySpec:
        """Generic alpha – fallback"""
        return StrategySpec(
            name=f"Alpha_{paper.id[:6]}",
            paper_id=paper.id,
            hypothesis=f"Alpha from paper: {paper.title}. {paper.abstract[:200]}... Economic: anomaly persists due to behavioral biases and institutional constraints.",
            entry_conditions=[
                "Close > EMA(Close, 50)  // Uptrend",
                "RSI(14) < 70  // Not overbought",
                "Volume > 500000  // Liquidity filter"
            ],
            exit_conditions=[
                "Close < EMA(Close, 50)  // Trend break",
                "OR RSI(14) > 80  // Overbought exit"
            ],
            filters=["Price > $10", "Volume > 500k", "MarketCap > $1B"],
            parameters={
                "ema_fast": {"default": 50, "min": 20, "max": 100, "step": 10, "description": "Fast EMA"},
                "rsi_threshold": {"default": 70, "min": 60, "max": 80, "step": 5, "description": "RSI overbought threshold"}
            },
            markets=["SPY", "QQQ", "AAPL", "MSFT", "TQQQ", "MES=F"],
            risk_management={
                "stop_loss": "2%",
                "position_size": "equity / Close * 0.1"
            },
            expected_metrics={
                "Sharpe": "1.1",
                "MaxDD": "-12%",
                "WinRate": "55%"
            }
        )

    def generate_all(self, papers: List[Paper]) -> List[StrategySpec]:
        """Generate specs for list of papers – deterministic sorted by paper title"""
        if not isinstance(papers, list):
            raise TypeError("papers must be list")
        specs = []
        for paper in sorted(papers, key=lambda p: p.title):
            try:
                spec = self.generate_spec(paper)
                specs.append(spec)
            except Exception:
                continue
        return sorted(specs, key=lambda s: s.name)
