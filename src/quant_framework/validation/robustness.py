"""
11 Robustness Framework – institutional
Automatically performs:
- Monte Carlo
- Bootstrap
- Noise Injection
- Parameter Perturbation
- Randomized Start Dates
- Random Missing Data
- Transaction Cost Sensitivity
- Slippage Sensitivity

Result is robustness score 0-10.
"""

from typing import Dict, List, Callable, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class RobustnessResult:
    test_name: str
    original_metric: float
    stressed_metrics: List[float]
    mean_stressed: float
    std_stressed: float
    worst_case: float
    robustness_score: float  # 0-10, higher is more robust
    details: Dict


class RobustnessFramework:
    """
    Robustness Framework – deterministic, no randomness without seed
    Each test uses fixed seed for reproducibility
    """

    def __init__(self, random_seed: int = 42, n_simulations: int = 100):
        if not isinstance(random_seed, int):
            raise TypeError("random_seed must be int")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be >0")
        self.random_seed = int(random_seed)
        self.n_simulations = int(n_simulations)

    @staticmethod
    def _validate_returns(returns: pd.Series) -> None:
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")

    def monte_carlo(self, returns: pd.Series, metric_fn: Callable[[pd.Series], float]) -> RobustnessResult:
        """
        Monte Carlo – shuffle returns order (preserves distribution, destroys autocorrelation)
        Economic justification: tests if edge depends on specific sequence (path dependency)
        """
        self._validate_returns(returns)
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable")

        original = metric_fn(returns)
        # Deterministic shuffle using seed
        rng = np.random.RandomState(self.random_seed)
        stressed = []
        for i in range(self.n_simulations):
            # Deterministic permutation via rng
            perm = rng.permutation(len(returns))
            shuffled = returns.iloc[perm].reset_index(drop=True)
            shuffled.index = returns.index  # keep same index for determinism in time?
            # Actually keep original index order for metric that uses time? But shuffle order destroys time
            # Use shuffled values but original index
            shuffled_series = pd.Series(shuffled.values, index=returns.index)
            try:
                m = metric_fn(shuffled_series)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        # Robustness score: higher if mean close to original and std low
        # Score = 10 * (1 - |original-mean|/(|original|+1)) * (1 / (1+std))
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name="Monte Carlo Shuffle",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"n_sim": self.n_simulations, "method": "shuffle returns"}
        )

    def bootstrap(self, returns: pd.Series, metric_fn: Callable[[pd.Series], float]) -> RobustnessResult:
        """
        Bootstrap – sample with replacement, same length
        Economic: tests stability under resampling of market regimes
        """
        self._validate_returns(returns)
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable")

        original = metric_fn(returns)
        rng = np.random.RandomState(self.random_seed + 1)
        stressed = []
        for _ in range(self.n_simulations):
            sample_idx = rng.choice(len(returns), size=len(returns), replace=True)
            sample = returns.iloc[sample_idx]
            sample.index = returns.index  # keep index for determinism? Actually bootstrap loses time, but we keep original index
            sample = pd.Series(sample.values, index=returns.index)
            try:
                m = metric_fn(sample)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name="Bootstrap",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"n_sim": self.n_simulations, "method": "bootstrap with replacement"}
        )

    def noise_injection(self, returns: pd.Series, metric_fn: Callable[[pd.Series], float],
                        noise_level: float = 0.1) -> RobustnessResult:
        """
        Noise Injection – add Gaussian noise to returns
        Economic: tests if edge is noise or real signal
        """
        self._validate_returns(returns)
        if not 0 <= noise_level <= 1:
            raise ValueError("noise_level must be in [0,1]")

        original = metric_fn(returns)
        rng = np.random.RandomState(self.random_seed + 2)
        stressed = []
        for _ in range(self.n_simulations):
            noise = rng.normal(0, returns.std() * noise_level, len(returns))
            noisy = returns + noise
            noisy = pd.Series(noisy, index=returns.index)
            try:
                m = metric_fn(noisy)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name=f"Noise Injection {noise_level*100:.0f}%",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"noise_level": noise_level}
        )

    def parameter_perturbation(self, base_params: Dict, param_to_perturb: str,
                               returns_fn: Callable[[Dict], pd.Series],
                               metric_fn: Callable[[pd.Series], float],
                               perturbation_pct: float = 0.1) -> RobustnessResult:
        """
        Parameter Perturbation – vary one param +/- perturbation_pct

        Args:
            base_params: Dict of params
            param_to_perturb: Key to perturb
            returns_fn: Function that takes params dict and returns returns Series
            metric_fn: Metric function
            perturbation_pct: e.g., 0.1 = +/-10%

        Returns:
            RobustnessResult
        """
        if not isinstance(base_params, dict) or param_to_perturb not in base_params:
            raise ValueError("base_params must contain param_to_perturb")
        if not callable(returns_fn) or not callable(metric_fn):
            raise TypeError("returns_fn and metric_fn must be callable")

        original_returns = returns_fn(base_params)
        original = metric_fn(original_returns)

        base_val = base_params[param_to_perturb]
        if not isinstance(base_val, (int, float)):
            raise ValueError("param_to_perturb must be numeric")

        stressed = []
        # Deterministic perturbations: linspace -pct to +pct
        perturbations = np.linspace(-perturbation_pct, perturbation_pct, self.n_simulations)
        for pert in perturbations:
            new_params = base_params.copy()
            new_params[param_to_perturb] = base_val * (1 + pert)
            try:
                ret = returns_fn(new_params)
                m = metric_fn(ret)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        # Robustness higher if small std and mean close to original
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s * 5))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name=f"Param Perturbation {param_to_perturb} ±{perturbation_pct*100:.0f}%",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"param": param_to_perturb, "perturbation_pct": perturbation_pct}
        )

    def randomized_start_dates(self, returns: pd.Series, metric_fn: Callable[[pd.Series], float],
                               min_start_pct: float = 0.0, max_start_pct: float = 0.3) -> RobustnessResult:
        """
        Randomized Start Dates – start backtest at different points in history
        Economic: tests if edge depends on specific start date (e.g., lucky 2009 bottom)
        """
        self._validate_returns(returns)
        original = metric_fn(returns)
        n = len(returns)
        # Deterministic start indices: linspace of start percentages
        starts = np.linspace(int(n * min_start_pct), int(n * max_start_pct), self.n_simulations, dtype=int)

        stressed = []
        for s in starts:
            truncated = returns.iloc[s:]
            if len(truncated) < 30:
                continue
            try:
                m = metric_fn(truncated)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name="Randomized Start Dates",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"min_start_pct": min_start_pct, "max_start_pct": max_start_pct}
        )

    def random_missing_data(self, returns: pd.Series, metric_fn: Callable[[pd.Series], float],
                            missing_pct: float = 0.05) -> RobustnessResult:
        """
        Random Missing Data – randomly drop 5% of returns (set to 0)
        Economic: tests robustness to missing data, holidays, data gaps
        """
        self._validate_returns(returns)
        if not 0 <= missing_pct <= 0.5:
            raise ValueError("missing_pct must be in [0,0.5]")

        original = metric_fn(returns)
        rng = np.random.RandomState(self.random_seed + 3)
        stressed = []

        for _ in range(self.n_simulations):
            mask = rng.rand(len(returns)) < missing_pct
            modified = returns.copy()
            modified[mask] = 0.0  # treat missing as flat
            try:
                m = metric_fn(modified)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        score = 10 * (1 - abs(original - mean_s) / (abs(original) + 1)) * (1 / (1 + std_s * 2))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name=f"Random Missing Data {missing_pct*100:.0f}%",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"missing_pct": missing_pct}
        )

    def transaction_cost_sensitivity(self, returns: pd.Series, equity_fn: Callable[[float], pd.Series],
                                     metric_fn: Callable[[pd.Series], float],
                                     cost_range: List[float] = None) -> RobustnessResult:
        """
        Transaction Cost Sensitivity – vary commission from 0 to 10 bps

        Args:
            returns: Base returns without costs? Actually equity_fn should take cost_bps and return equity/returns
            equity_fn: Function cost_bps -> returns Series
            metric_fn: Metric function
            cost_range: List of cost bps to test

        Returns:
            RobustnessResult
        """
        if cost_range is None:
            cost_range = [0, 1, 2, 5, 10]  # bps

        # Original at 2 bps (default)
        original_returns = equity_fn(2.0)
        original = metric_fn(original_returns)

        stressed = []
        for cost in cost_range:
            try:
                ret = equity_fn(float(cost))
                m = metric_fn(ret)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        # Higher robustness if not sensitive to costs
        score = 10 * (1 - abs(original - worst) / (abs(original) + 1))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name="Transaction Cost Sensitivity",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"cost_range_bps": cost_range}
        )

    def slippage_sensitivity(self, returns: pd.Series, equity_fn: Callable[[float], pd.Series],
                             metric_fn: Callable[[pd.Series], float],
                             slippage_range: List[float] = None) -> RobustnessResult:
        """Slippage sensitivity similar to transaction cost"""
        if slippage_range is None:
            slippage_range = [0, 1, 2, 5, 10]

        original_returns = equity_fn(2.0)
        original = metric_fn(original_returns)

        stressed = []
        for slip in slippage_range:
            try:
                ret = equity_fn(float(slip))
                m = metric_fn(ret)
                if np.isfinite(m):
                    stressed.append(float(m))
            except Exception:
                continue

        mean_s = float(np.mean(stressed)) if stressed else 0.0
        std_s = float(np.std(stressed)) if stressed else 0.0
        worst = float(np.min(stressed)) if stressed else 0.0
        score = 10 * (1 - abs(original - worst) / (abs(original) + 1))
        score = float(np.clip(score, 0, 10))

        return RobustnessResult(
            test_name="Slippage Sensitivity",
            original_metric=float(original),
            stressed_metrics=stressed,
            mean_stressed=mean_s,
            std_stressed=std_s,
            worst_case=worst,
            robustness_score=score,
            details={"slippage_range_bps": slippage_range}
        )

    def compute_overall_robustness_score(self, results: List[RobustnessResult]) -> Dict:
        """
        Compute overall robustness score as average of individual scores, weighted.

        Args:
            results: List of RobustnessResult

        Returns:
            Dict with overall_score, breakdown, worst_test, etc – deterministic

        Raises:
            TypeError
        """
        if not isinstance(results, list) or not results:
            raise ValueError("results must be non-empty list")

        scores = [r.robustness_score for r in results]
        overall = float(np.mean(scores))
        worst = min(results, key=lambda r: r.robustness_score)

        breakdown = {r.test_name: round(r.robustness_score, 2) for r in sorted(results, key=lambda x: x.test_name)}

        return {
            "overall_robustness_score": round(overall, 2),
            "overall_robustness_score_10": round(overall, 2),  # same scale 0-10
            "mean_score": round(float(np.mean(scores)), 2),
            "median_score": round(float(np.median(scores)), 2),
            "min_score": round(float(np.min(scores)), 2),
            "max_score": round(float(np.max(scores)), 2),
            "std_score": round(float(np.std(scores)), 2),
            "worst_test": worst.test_name,
            "worst_score": round(worst.robustness_score, 2),
            "breakdown": breakdown,
            "num_tests": len(results),
            "interpretation": "Robust" if overall >= 7 else "Moderate" if overall >= 4 else "Fragile"
        }
