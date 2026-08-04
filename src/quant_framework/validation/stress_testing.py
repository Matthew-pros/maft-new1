
"""
11 Stress Testing – historical and synthetic
"""

from typing import List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class StressScenario:
    name: str
    start: str
    end: str
    description: str

HISTORICAL_SCENARIOS = [
    StressScenario("2008 Financial Crisis", "2007-10-01", "2009-03-31", "GFC bear market -50%"),
    StressScenario("2020 COVID Crash", "2020-02-15", "2020-04-15", "COVID crash -34% in 33 days"),
    StressScenario("2022 Bear Market", "2022-01-01", "2022-12-31", "Rate hike bear -25%"),
    StressScenario("Dotcom Bust", "2000-03-01", "2002-10-31", "Nasdaq -78%"),
    StressScenario("Inflation Shock 2022", "2021-11-01", "2022-10-31", "High inflation + rates up"),
]

class StressTester:
    def __init__(self, scenarios: List[StressScenario] = None):
        self.scenarios = scenarios or HISTORICAL_SCENARIOS

    def run_historical(self, equity_curve: pd.Series) -> Dict[str, Dict]:
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            raise ValueError("equity_curve must be non-empty Series")
        results = {}
        for scen in self.scenarios:
            try:
                start = pd.to_datetime(scen.start)
                end = pd.to_datetime(scen.end)
                sliced = equity_curve[(equity_curve.index >= start) & (equity_curve.index <= end)]
                if sliced.empty:
                    results[scen.name] = {"total_return": None, "max_dd": None, "note": "No data in period"}
                    continue
                total_ret = sliced.iloc[-1] / sliced.iloc[0] - 1
                peak = sliced.cummax()
                dd = sliced / peak - 1
                max_dd = dd.min()
                results[scen.name] = {"total_return": float(total_ret), "max_dd": float(max_dd), "description": scen.description, "start": scen.start, "end": scen.end}
            except Exception as e:
                results[scen.name] = {"error": str(e)}
        return results

    def run_synthetic(self, returns: pd.Series, shock_type: str = "volatility", shock_factor: float = 2.0) -> Dict:
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")
        if shock_factor <= 0:
            raise ValueError("shock_factor must be >0")
        if shock_type == "volatility":
            stressed = returns * shock_factor
        elif shock_type == "rate":
            stressed = returns.copy()
            stressed.iloc[-60:] -= 0.01 * shock_factor / 252
        elif shock_type == "correlation":
            stressed = returns * (1 + (shock_factor - 1) * 0.5)
        else:
            raise ValueError("shock_type must be volatility, rate, correlation")
        equity = (1 + stressed).cumprod()
        total_ret = equity.iloc[-1] - 1
        peak = equity.cummax()
        dd = equity / peak - 1
        max_dd = dd.min()
        return {"shock_type": shock_type, "shock_factor": shock_factor, "total_return": float(total_ret), "max_dd": float(max_dd), "volatility": float(stressed.std() * np.sqrt(252)), "original_volatility": float(returns.std() * np.sqrt(252))}
