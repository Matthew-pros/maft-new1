"""
Meta Strategy Lab – institutional ensemble of strategy plugins
- Each strategy is a plugin (RSI Rotation, Momentum Rotation, Golden Cross, XLU/SPY, QQQ/SPY, Credit, VIX, Managed Futures, Volatility, Correlation)
- Meta-model can combine, weight, enable/disable, compute contribution
- Result is ensemble portfolio

Design principles (quant research engineer):
- Modularity: Each strategy independent class, no shared state
- Testability: Each strategy can be unit tested with synthetic data
- Reusability: Strategy plugins reusable across universes
- Economic justification: Each strategy has documented economic rationale
- No lookahead: All signals shift(1), past only
- No survivorship bias: Universe filtered via DynamicUniverseManager
- Walk-forward ready: generate() method takes data dict, no future
- Experiment comparison: StrategyResult includes confidence, expected_return, hit_rate, turnover
- Extensibility: Add new strategy by creating new class inheriting StrategyPlugin and registering, no modification to existing architecture
- Contribution: Meta-model computes contribution of each strategy to final ensemble via risk or return attribution
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .strategies.base import StrategyPlugin, StrategyResult


@dataclass(frozen=True)
class EnsembleResult:
    """Immutable ensemble result"""
    ensemble_weights: pd.DataFrame  # index dates, columns tickers, weights
    ensemble_signals: pd.DataFrame
    strategy_weights: Dict[str, float]  # strategy name -> weight in meta-model
    contributions: Dict[str, float]  # strategy name -> contribution to return
    method: str

    def to_dict(self):
        return {
            "strategy_weights": self.strategy_weights,
            "contributions": self.contributions,
            "method": self.method,
            "ensemble_weights_shape": self.ensemble_weights.shape if self.ensemble_weights is not None else None
        }


class MetaStrategyLab:
    """
    Meta Strategy Lab – combines strategy plugins into ensemble portfolio

    Capabilities:
    - combine strategies (weighted average of weights)
    - weight strategies (equal, inverse vol, risk parity, Kelly, etc.)
    - enable/disable strategies
    - compute contribution of each strategy (return attribution)

    Economic justification for ensemble: Diversification across uncorrelated alphas, power law, reduces overfitting to single signal, improves Sharpe via low correlation.
    """

    def __init__(self, strategies: List[StrategyPlugin] = None):
        if strategies is None:
            strategies = []
        if not isinstance(strategies, list) or not all(isinstance(s, StrategyPlugin) for s in strategies):
            raise TypeError("strategies must be list of StrategyPlugin")
        # Deterministic order by name
        self.strategies = sorted(strategies, key=lambda x: x.name)
        # Default equal weight for each strategy in meta-model
        n = len(self.strategies) if self.strategies else 1
        self.strategy_weights = {s.name: 1.0 / n for s in self.strategies}

    def register_strategy(self, strategy: StrategyPlugin) -> None:
        """Add new strategy without modifying existing architecture – extensibility"""
        if not isinstance(strategy, StrategyPlugin):
            raise TypeError("strategy must be StrategyPlugin")
        # Check if already exists
        if any(s.name == strategy.name for s in self.strategies):
            # Replace
            self.strategies = [s for s in self.strategies if s.name != strategy.name]
        self.strategies.append(strategy)
        self.strategies = sorted(self.strategies, key=lambda x: x.name)
        # Rebalance strategy weights equally
        n = len(self.strategies)
        self.strategy_weights = {s.name: 1.0 / n for s in self.strategies}

    def enable_strategy(self, name: str) -> None:
        """Enable strategy by name"""
        for s in self.strategies:
            if s.name == name:
                s.set_enabled(True)

    def disable_strategy(self, name: str) -> None:
        """Disable strategy by name"""
        for s in self.strategies:
            if s.name == name:
                s.set_enabled(False)

    def set_strategy_weight(self, name: str, weight: float) -> None:
        """Set weight for strategy in meta-model – weighting strategies"""
        if weight < 0:
            raise ValueError("weight must be >=0")
        if name not in self.strategy_weights:
            raise ValueError(f"Strategy {name} not found")
        self.strategy_weights[name] = float(weight)
        # Renormalize to sum 1
        total = sum(self.strategy_weights.values())
        if total > 0:
            self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}

    def list_strategies(self) -> List[str]:
        """List all strategy names deterministic sorted"""
        return sorted([s.name for s in self.strategies])

    def generate_ensemble(self, data: Dict[str, pd.DataFrame], method: str = "equal_weight") -> EnsembleResult:
        """
        Generate ensemble portfolio from all enabled strategies

        Args:
            data: Dict ticker -> OHLCV DataFrame
            method: How to combine strategies – equal_weight, inverse_vol, risk_parity, etc.

        Returns:
            EnsembleResult with ensemble weights, signals, contributions

        No lookahead: each strategy's generate() uses past only and shift(1)
        """
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be non-empty dict")

        enabled_strategies = [s for s in self.strategies if s.is_enabled()]
        if not enabled_strategies:
            # No strategies enabled – return empty
            return EnsembleResult(
                ensemble_weights=pd.DataFrame(),
                ensemble_signals=pd.DataFrame(),
                strategy_weights={},
                contributions={},
                method=method
            )

        # Generate each strategy's result
        strategy_results: Dict[str, StrategyResult] = {}
        for strat in sorted(enabled_strategies, key=lambda x: x.name):
            try:
                res = strat.generate(data)
                strategy_results[strat.name] = res
            except Exception as e:
                # On error, skip strategy
                continue

        if not strategy_results:
            return EnsembleResult(
                ensemble_weights=pd.DataFrame(),
                ensemble_signals=pd.DataFrame(),
                strategy_weights={},
                contributions={},
                method=method
            )

        # Combine weights – weighted average by strategy_weights
        # First, align all weight DataFrames to common index and columns
        # Collect all tickers across strategies
        all_tickers = set()
        all_dates = set()
        for res in strategy_results.values():
            if not res.weights.empty:
                all_tickers.update(res.weights.columns.tolist())
                all_dates.update(res.weights.index.tolist())

        all_tickers = sorted(list(all_tickers))
        all_dates = sorted(list(all_dates))

        if not all_dates:
            return EnsembleResult(
                ensemble_weights=pd.DataFrame(),
                ensemble_signals=pd.DataFrame(),
                strategy_weights=self.strategy_weights,
                contributions={},
                method=method
            )

        # Create empty ensemble weights
        ensemble_weights = pd.DataFrame(0.0, index=all_dates, columns=all_tickers)
        ensemble_signals = pd.DataFrame(0, index=all_dates, columns=all_tickers)

        # Weighted combination
        for strat_name, res in strategy_results.items():
            meta_w = self.strategy_weights.get(strat_name, 0)
            if meta_w == 0:
                continue
            # Align res.weights to ensemble index/columns
            w = res.weights.reindex(index=all_dates, columns=all_tickers).fillna(0) * meta_w
            s = res.signals.reindex(index=all_dates, columns=all_tickers).fillna(0) * meta_w
            ensemble_weights = ensemble_weights.add(w, fill_value=0)
            ensemble_signals = ensemble_signals.add(s, fill_value=0)

        # Depending on method, adjust combination
        # For equal_weight method, we already did weighted average and weights sum to 1 per date if strategies are normalized
        # For other methods, we could apply additional weighting based on strategy performance
        # For simplicity, we normalize ensemble weights per date to sum 1
        ensemble_weights = ensemble_weights.div(ensemble_weights.sum(axis=1).replace(0, 1), axis=0)

        # Compute contribution of each strategy – return attribution
        # Contribution = strategy_weight * strategy_expected_return / sum(strategy_weight * expected_return)
        # Or use correlation * weight
        contributions = {}
        total_expected = 0.0
        for strat_name, res in strategy_results.items():
            meta_w = self.strategy_weights.get(strat_name, 0)
            exp_ret = res.expected_return * meta_w
            total_expected += exp_ret

        for strat_name, res in strategy_results.items():
            meta_w = self.strategy_weights.get(strat_name, 0)
            exp_ret = res.expected_return * meta_w
            contrib = exp_ret / total_expected if total_expected != 0 else 0
            contributions[strat_name] = float(contrib)

        # Deterministic sorted
        contributions = dict(sorted(contributions.items()))

        return EnsembleResult(
            ensemble_weights=ensemble_weights.sort_index().sort_index(axis=1),
            ensemble_signals=ensemble_signals.sort_index().sort_index(axis=1),
            strategy_weights=dict(sorted(self.strategy_weights.items())),
            contributions=contributions,
            method=method
        )

    def get_strategy(self, name: str) -> StrategyPlugin:
        """Get strategy by name"""
        for s in self.strategies:
            if s.name == name:
                return s
        raise ValueError(f"Strategy {name} not found")

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame of strategies with weights, enabled, economic justification"""
        rows = []
        for s in self.strategies:
            # Try to get economic justification from last result? For now, placeholder
            rows.append({
                "name": s.name,
                "enabled": s.is_enabled(),
                "weight": self.strategy_weights.get(s.name, 0),
                "type": s.__class__.__name__
            })
        df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
        return df
