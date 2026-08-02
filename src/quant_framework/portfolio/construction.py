
"""
06 Portfolio Construction – fully separated from signal engine
Supports: Equal Weight, Inverse Vol, Risk Parity, Max Div, Min Var, Vol Targeting, Kelly, Dynamic Sizing, Cash Allocation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import numpy as np
import pandas as pd
from ..config.settings import PortfolioConfig

@dataclass(frozen=True)
class PortfolioWeights:
    weights: Dict[str, float]
    method: str
    cash_weight: float
    timestamp: Optional[pd.Timestamp] = None
    diagnostics: Dict = None
    def to_dict(self):
        return {"weights": self.weights, "method": self.method, "cash_weight": self.cash_weight, "timestamp": str(self.timestamp) if self.timestamp else None, "diagnostics": self.diagnostics or {}}

class PortfolioConstructor:
    def __init__(self, config: PortfolioConfig):
        if not isinstance(config, PortfolioConfig):
            raise TypeError("config must be PortfolioConfig")
        self.config = config

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        clipped = {}
        for t, w in weights.items():
            w_clipped = max(self.config.min_position_weight, min(self.config.max_position_weight, w))
            clipped[t] = w_clipped
        total = sum(clipped.values())
        cash = self.config.cash_allocation
        target_equity = 1 - cash
        if total == 0:
            n = len(clipped)
            return {t: target_equity / n for t in clipped}
        factor = target_equity / total if total != 0 else 0
        normalized = {t: w * factor for t, w in clipped.items()}
        return dict(sorted(normalized.items()))

    def equal_weight(self, tickers: List[str], cash_allocation: Optional[float] = None):
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be non-empty list")
        tickers_sorted = sorted(list(dict.fromkeys(tickers)))
        cash = cash_allocation if cash_allocation is not None else self.config.cash_allocation
        n = len(tickers_sorted)
        equity_weight = (1 - cash) / n
        weights = {t: equity_weight for t in tickers_sorted}
        return PortfolioWeights(weights=weights, method="equal_weight", cash_weight=cash, diagnostics={"n": n})

    def inverse_volatility(self, volatilities: pd.Series, cash_allocation: Optional[float] = None):
        if not isinstance(volatilities, pd.Series) or volatilities.empty:
            raise ValueError("volatilities must be non-empty Series")
        if (volatilities <= 0).any():
            raise ValueError("volatilities must be >0")
        cash = cash_allocation if cash_allocation is not None else self.config.cash_allocation
        inv_vol = 1 / volatilities
        inv_vol = inv_vol / inv_vol.sum() * (1 - cash)
        weights = inv_vol.to_dict()
        weights = self._apply_weight_constraints(weights)
        return PortfolioWeights(weights=weights, method="inverse_vol", cash_weight=cash, diagnostics={"avg_vol": float(volatilities.mean())})

    def risk_parity(self, cov_matrix: pd.DataFrame, cash_allocation: Optional[float] = None, max_iter: int = 100):
        if not isinstance(cov_matrix, pd.DataFrame) or cov_matrix.empty:
            raise ValueError("cov_matrix must be non-empty DataFrame")
        tickers = sorted(cov_matrix.columns.tolist())
        n = len(tickers)
        w = np.ones(n) / n
        cov = cov_matrix.values
        for _ in range(max_iter):
            portfolio_vol = np.sqrt(w @ cov @ w)
            if portfolio_vol == 0:
                break
            mrc = cov @ w / portfolio_vol
            rc = w * mrc
            target_rc = portfolio_vol / n
            rc_safe = np.where(rc == 0, 1e-8, rc)
            w = w * (target_rc / rc_safe)
            w = w / w.sum()
        cash = cash_allocation if cash_allocation is not None else self.config.cash_allocation
        w = w * (1 - cash)
        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
        weights = self._apply_weight_constraints(weights)
        return PortfolioWeights(weights=weights, method="risk_parity", cash_weight=cash, diagnostics={"iterations": max_iter})

    def volatility_targeting(self, volatilities: pd.Series, target_vol: Optional[float] = None, cash_allocation: Optional[float] = None):
        if not isinstance(volatilities, pd.Series) or volatilities.empty:
            raise ValueError("volatilities must be non-empty Series")
        t_vol = target_vol if target_vol is not None else self.config.target_vol
        cash = cash_allocation if cash_allocation is not None else self.config.cash_allocation
        exposure = t_vol / volatilities.replace(0, np.nan)
        exposure = exposure.dropna()
        if exposure.empty:
            raise ValueError("No valid volatilities")
        weights_raw = exposure / exposure.sum() * (1 - cash)
        weights = weights_raw.to_dict()
        weights = self._apply_weight_constraints(weights)
        return PortfolioWeights(weights=weights, method="volatility_targeting", cash_weight=cash, diagnostics={"target_vol": t_vol})

    def kelly_fraction(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, fraction: Optional[float] = None, cash_allocation: Optional[float] = None):
        if not isinstance(expected_returns, pd.Series) or expected_returns.empty:
            raise ValueError("expected_returns must be non-empty Series")
        kelly_f = fraction if fraction is not None else self.config.kelly_fraction
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            mu = expected_returns.values
            w = inv_cov @ mu
            w = w * kelly_f
            w = np.clip(w, 0, None)
            if w.sum() == 0:
                w = np.ones(len(expected_returns)) / len(expected_returns) * kelly_f
            else:
                if w.sum() > 1:
                    w = w / w.sum() * (1 - (cash_allocation or self.config.cash_allocation))
        except Exception:
            w = np.ones(len(expected_returns)) / len(expected_returns) * kelly_f
        tickers = sorted(expected_returns.index.tolist())
        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
        weights = self._apply_weight_constraints(weights)
        total_w = sum(weights.values())
        cash_final = max(0.0, 1 - total_w)
        return PortfolioWeights(weights=weights, method="kelly", cash_weight=cash_final, diagnostics={"kelly_fraction": kelly_f})

    def dynamic_position_sizing(self, base_weights: Dict[str, float], volatility_regime: float, cash_allocation: Optional[float] = None):
        if not isinstance(base_weights, dict) or not base_weights:
            raise ValueError("base_weights must be non-empty dict")
        if not isinstance(volatility_regime, (int, float)) or volatility_regime <= 0:
            raise ValueError("volatility_regime must be positive")
        scale = 1 / volatility_regime
        scale = max(0.2, min(1.5, scale))
        scaled = {t: w * scale for t, w in base_weights.items()}
        total = sum(scaled.values())
        cash = cash_allocation if cash_allocation is not None else self.config.cash_allocation
        if total > 0:
            scaled = {t: w / total * (1 - cash) for t, w in scaled.items()}
        scaled = self._apply_weight_constraints(scaled)
        return PortfolioWeights(weights=scaled, method="dynamic_sizing", cash_weight=cash, diagnostics={"vol_regime": volatility_regime, "scale": scale})
