
"""
08 Performance Analytics – institutional metrics including PSR/DSR
"""
from typing import Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass
try:
    from scipy.stats import norm, skew, kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    class norm:
        @staticmethod
        def cdf(x):
            import math
            return 0.5*(1+math.erf(x/math.sqrt(2)))
        @staticmethod
        def ppf(q):
            return 0.0

def probabilistic_sharpe_ratio(returns: pd.Series, benchmark: float = 0.0, periods_per_year: int = 252) -> float:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("returns must be non-empty Series")
    r = returns.dropna().values
    if len(r) < 10:
        return 0.0
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std * np.sqrt(periods_per_year)
    n = len(r)
    if HAS_SCIPY:
        g3 = skew(r)
        g4 = kurtosis(r, fisher=False)
    else:
        g3 = 0.0
        g4 = 3.0
    if not np.isfinite(g3):
        g3 = 0.0
    if not np.isfinite(g4) or g4 < 1:
        g4 = 3.0
    denom = np.sqrt(1 - g3*sr + (g4-1)/4 * sr**2)
    if denom == 0 or not np.isfinite(denom):
        denom = 1.0
    psr = norm.cdf((sr - benchmark) * np.sqrt(n-1) / denom)
    return float(psr)

def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 10, periods_per_year: int = 252) -> float:
    gamma = 0.5772156649
    M = max(2, n_trials)
    try:
        inv1 = norm.ppf(1 - 1/M) if HAS_SCIPY else 2.0
        inv2 = norm.ppf(1 - 1/(M * np.e)) if HAS_SCIPY else 2.5
    except Exception:
        inv1 = 2.0
        inv2 = 2.5
    var_sr = 0.15
    sr_star = np.sqrt(var_sr) * ((1 - gamma) * inv1 + gamma * inv2)
    return probabilistic_sharpe_ratio(returns, benchmark=sr_star, periods_per_year=periods_per_year)

@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    win_rate: float
    profit_factor: float
    psr: float
    dsr: float
    annual_vol: float
    calmar: float

class PerformanceAnalytics:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        if not isinstance(risk_free_rate, (int, float)):
            raise TypeError("risk_free_rate must be numeric")
        if not isinstance(periods_per_year, int) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive int")
        self.risk_free_rate = float(risk_free_rate)
        self.periods_per_year = int(periods_per_year)

    def compute(self, equity_curve: pd.Series, benchmark: pd.Series = None):
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            raise ValueError("equity_curve must be non-empty Series")
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            raise TypeError("equity_curve index must be DatetimeIndex")
        equity_curve = equity_curve.sort_index()
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            raise ValueError("returns empty after pct_change")
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        sharpe = (mean_ret - self.risk_free_rate/252) / std_ret * np.sqrt(self.periods_per_year) if std_ret != 0 else 0.0
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else std_ret
        sortino = (mean_ret - self.risk_free_rate/252) / downside_std * np.sqrt(self.periods_per_year) if downside_std != 0 else 0.0
        peak = equity_curve.cummax()
        dd = equity_curve / peak - 1
        max_dd = dd.min()
        win_rate = (returns > 0).mean() * 100
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        psr = probabilistic_sharpe_ratio(returns, benchmark=0.0, periods_per_year=self.periods_per_year)
        dsr = deflated_sharpe_ratio(returns, n_trials=10, periods_per_year=self.periods_per_year)
        annual_vol = std_ret * np.sqrt(self.periods_per_year)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        return PerformanceMetrics(
            total_return=float(total_return), cagr=float(cagr), sharpe=float(sharpe),
            sortino=float(sortino), max_dd=float(max_dd), win_rate=float(win_rate),
            profit_factor=float(profit_factor), psr=float(psr), dsr=float(dsr),
            annual_vol=float(annual_vol), calmar=float(calmar)
        )
