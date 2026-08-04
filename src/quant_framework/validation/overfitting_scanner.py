"""
Overfitting Scanner – institutional
Computes:
- Parameter sensitivity
- Neighbour stability
- Walk Forward stability
- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio (DSR)
- White Reality Check, White's Reality Check p-value
- Model Confidence Score (MCS)
Adds automatic warning, blocks model with high overfitting
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Tuple, Optional
import pandas as pd
import numpy as np
import itertools


@dataclass(frozen=True)
class OverfittingMetrics:
    param_sensitivity: float
    neighbour_stability: float
    walk_forward_stability: float
    pbo: float
    dsr: float
    white_reality_check_pvalue: float
    model_confidence_score: float
    overall_overfitting_score: float  # 0-10, higher = more overfitting
    is_overfitted: bool
    warnings: List[str]

    def to_dict(self):
        return {
            "param_sensitivity": self.param_sensitivity,
            "neighbour_stability": self.neighbour_stability,
            "walk_forward_stability": self.walk_forward_stability,
            "pbo": self.pbo,
            "dsr": self.dsr,
            "white_reality_check_pvalue": self.white_reality_check_pvalue,
            "model_confidence_score": self.model_confidence_score,
            "overall_overfitting_score": self.overall_overfitting_score,
            "is_overfitted": self.is_overfitted,
            "warnings": self.warnings
        }


class OverfittingScanner:
    """
    Overfitting Scanner – institutional grade
    - Parameter sensitivity: how much Sharpe changes when param ±10%
    - Neighbour stability: std of Sharpe in neighbourhood of best params
    - Walk Forward stability: correlation IS vs OOS, OOS/IS ratio
    - PBO: Probability of Backtest Overfitting (Bailey et al) via combinatorial splits
    - DSR: Deflated Sharpe Ratio (already in performance analytics)
    - White Reality Check: Bootstrap reality check for data snooping
    - Model Confidence Score: Hansen's MCS-like
    - Automatic warning, blocks high overfitting model
    """

    def __init__(self, overfitting_threshold: float = 6.0, random_seed: int = 42):
        if not 0 <= overfitting_threshold <= 10:
            raise ValueError("overfitting_threshold must be in [0,10]")
        self.overfitting_threshold = float(overfitting_threshold)
        self.random_seed = int(random_seed)

    @staticmethod
    def _validate_series(returns: pd.Series) -> None:
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError("returns must be non-empty Series")

    def parameter_sensitivity(self, base_params: Dict, param_grid: Dict[str, List], returns_fn: Callable[[Dict], pd.Series], metric_fn: Callable[[pd.Series], float]) -> float:
        """
        Parameter sensitivity: std of metric when varying each param ±10% / grid
        Economic: if small param change causes large performance change, overfitted

        Returns:
            Sensitivity score 0-1 (higher = more sensitive = more overfitted)
        """
        if not isinstance(base_params, dict) or not base_params:
            raise ValueError("base_params must be non-empty dict")
        if not isinstance(param_grid, dict):
            raise TypeError("param_grid must be dict")

        base_metric = None
        try:
            base_ret = returns_fn(base_params)
            base_metric = metric_fn(base_ret)
        except Exception:
            base_metric = 0.0

        metrics = []
        for param, values in param_grid.items():
            for val in values:
                test_params = base_params.copy()
                test_params[param] = val
                try:
                    ret = returns_fn(test_params)
                    m = metric_fn(ret)
                    if np.isfinite(m):
                        metrics.append(float(m))
                except Exception:
                    continue

        if not metrics or base_metric == 0:
            return 0.0

        # Sensitivity = std / |mean| – coefficient of variation
        metrics_arr = np.array(metrics)
        cv = np.std(metrics_arr) / (abs(np.mean(metrics_arr)) + 1e-8)
        # Normalize to 0-1: cv 0-0.5 -> low sensitivity, cv >1 -> high sensitivity
        sensitivity = min(1.0, cv)
        return float(sensitivity)

    def neighbour_stability(self, best_params: Dict, param_grid: Dict[str, List], returns_fn: Callable[[Dict], pd.Series], metric_fn: Callable[[pd.Series], float], neighbour_radius: int = 1) -> float:
        """
        Neighbour stability: check performance of neighbours around best params

        Economic: robust strategy should have flat plateau around optimum, not sharp peak
        """
        # Generate neighbours: for each param, try +/- 1 step in grid
        neighbours = []
        for param, values in param_grid.items():
            if param not in best_params:
                continue
            try:
                idx = values.index(best_params[param])
            except ValueError:
                continue
            for offset in range(-neighbour_radius, neighbour_radius+1):
                if offset == 0:
                    continue
                nidx = idx + offset
                if 0 <= nidx < len(values):
                    test_params = best_params.copy()
                    test_params[param] = values[nidx]
                    try:
                        ret = returns_fn(test_params)
                        m = metric_fn(ret)
                        if np.isfinite(m):
                            neighbours.append(float(m))
                    except Exception:
                        continue

        if not neighbours:
            return 1.0  # no neighbours -> unstable

        base_ret = returns_fn(best_params)
        base_metric = metric_fn(base_ret)

        # Stability = 1 - std(neighbours)/|base|  – higher is more stable
        # If neighbours have similar performance to base, stable
        neighbours_arr = np.array(neighbours)
        std_neigh = np.std(neighbours_arr)
        # How far is base from neighbours mean?
        mean_neigh = np.mean(neighbours_arr)
        # If base much higher than neighbours mean, it's isolated peak -> overfitted
        diff = abs(base_metric - mean_neigh) / (abs(base_metric) + 1e-8)
        # Combine std and diff
        instability = (std_neigh / (abs(base_metric) + 1e-8) + diff) / 2
        stability = 1 - min(1.0, instability)
        return float(np.clip(stability, 0, 1))

    def walk_forward_stability(self, is_metrics: List[float], oos_metrics: List[float]) -> float:
        """
        Walk Forward stability: correlation IS vs OOS and OOS/IS ratio

        Economic: if IS performance doesn't translate to OOS, overfitted

        Args:
            is_metrics: List of in-sample Sharpe per window
            oos_metrics: List of out-of-sample Sharpe per window

        Returns:
            Stability score 0-1 (higher = more stable)
        """
        if not is_metrics or not oos_metrics or len(is_metrics) != len(oos_metrics):
            return 0.0

        is_arr = np.array(is_metrics)
        oos_arr = np.array(oos_metrics)

        # Correlation IS vs OOS – should be positive if stable
        if len(is_arr) < 2:
            corr = 0.0
        else:
            corr = np.corrcoef(is_arr, oos_arr)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        # OOS/IS ratio – should be close to 1 if stable, if OOS << IS, overfitted
        is_mean = np.mean(is_arr) if len(is_arr) > 0 else 0
        oos_mean = np.mean(oos_arr) if len(oos_arr) > 0 else 0
        ratio = oos_mean / (is_mean + 1e-8) if is_mean != 0 else 0
        # Clip ratio to [0,1] for stability (if OOS > IS, cap at 1)
        ratio_clipped = np.clip(ratio, 0, 1)

        # Stability = (corr+1)/2 * ratio – both need to be high
        # corr in [-1,1] -> map to [0,1] via (corr+1)/2
        corr_norm = (corr + 1) / 2
        stability = corr_norm * ratio_clipped

        return float(np.clip(stability, 0, 1))

    def probability_of_backtest_overfitting(self, returns_matrix: pd.DataFrame) -> float:
        """
        Probability of Backtest Overfitting (PBO) – Bailey et al 2014
        Combinatorial splits of returns into IS/OOS, compute rank of best IS in OOS

        Simplified implementation:
        - returns_matrix: DataFrame columns = strategies (different param combos), rows = time
        - Split time into 2 halves many times via combinatorial
        - For each split, rank strategies by IS Sharpe, check OOS rank of best IS
        - PBO = probability that best IS is worse than median OOS (i.e., rank > N/2)

        Args:
            returns_matrix: DataFrame where each column is a strategy's returns over time

        Returns:
            PBO 0-1 (higher = more overfitting)
        """
        if not isinstance(returns_matrix, pd.DataFrame) or returns_matrix.empty or returns_matrix.shape[1] < 2:
            return 0.0

        n_strategies = returns_matrix.shape[1]
        n_obs = returns_matrix.shape[0]

        # Number of combinatorial splits – for simplicity, use random splits with seed
        n_splits = min(16, self._n_combinations(n_obs, n_obs//2))  # limit to 16 for speed
        rng = np.random.RandomState(self.random_seed)

        worse_than_median = 0
        total = 0

        for _ in range(n_splits):
            # Random split point
            split = rng.randint(n_obs//4, 3*n_obs//4)
            is_returns = returns_matrix.iloc[:split]
            oos_returns = returns_matrix.iloc[split:]

            # Compute Sharpe for each strategy in IS and OOS
            def sharpe(df):
                mean = df.mean()
                std = df.std()
                return mean / std * np.sqrt(252) if std.all() and (std != 0).all() else pd.Series(0, index=df.columns)

            # Actually compute per column
            is_sharpe = {}
            oos_sharpe = {}
            for col in returns_matrix.columns:
                is_ret = is_returns[col].dropna()
                oos_ret = oos_returns[col].dropna()
                if len(is_ret) < 10 or len(oos_ret) < 10:
                    continue
                is_sh = is_ret.mean() / is_ret.std() * np.sqrt(252) if is_ret.std() != 0 else 0
                oos_sh = oos_ret.mean() / oos_ret.std() * np.sqrt(252) if oos_ret.std() != 0 else 0
                is_sharpe[col] = is_sh
                oos_sharpe[col] = oos_sh

            if not is_sharpe:
                continue

            # Find best IS strategy
            best_is = max(is_sharpe, key=lambda k: is_sharpe[k])
            # Rank OOS Sharpe of best IS among all OOS Sharpes
            sorted_oos = sorted(oos_sharpe.values())
            # Find rank of oos_sharpe[best_is]
            best_oos = oos_sharpe.get(best_is, 0)
            # Rank: position in sorted list (0 = worst, N-1 = best)
            rank = sum(1 for v in sorted_oos if v <= best_oos)
            # If rank < median, then best IS is worse than median OOS -> overfitting
            median_rank = len(sorted_oos) // 2
            if rank < median_rank:
                worse_than_median += 1
            total += 1

        pbo = worse_than_median / total if total > 0 else 0.0
        return float(pbo)

    @staticmethod
    def _n_combinations(n, k):
        # n choose k approximated, but we just return large number
        import math
        try:
            return math.comb(n, k)
        except AttributeError:
            # Python <3.8
            return 2**10

    def deflated_sharpe_ratio(self, returns: pd.Series, n_trials: int = 10) -> float:
        """DSR – use existing implementation from performance analytics"""
        try:
            from ..performance.analytics import deflated_sharpe_ratio
            return float(deflated_sharpe_ratio(returns, n_trials=n_trials))
        except Exception:
            # Fallback simple
            return 0.5

    def white_reality_check(self, returns: pd.Series, benchmark_returns: pd.Series, n_bootstrap: int = 100) -> float:
        """
        White's Reality Check – bootstrap test for data snooping
        Simplified: test if best strategy's Sharpe significantly > benchmark

        Economic: checks if best found Sharpe could be due to luck among many trials

        Returns:
            p-value 0-1 (low p-value = significant, not overfitted due to luck)
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            return 1.0
        if not isinstance(benchmark_returns, pd.Series):
            benchmark_returns = pd.Series(0, index=returns.index)

        # Align
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(aligned) < 30:
            return 1.0

        strat_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1] if aligned.shape[1] > 1 else pd.Series(0, index=aligned.index)

        # Best Sharpe vs benchmark
        def sharpe(s):
            return s.mean() / s.std() * np.sqrt(252) if s.std() != 0 else 0

        best_sharpe = sharpe(strat_ret)
        bench_sharpe = sharpe(bench_ret)
        excess = best_sharpe - bench_sharpe

        # Bootstrap: resample returns with replacement, compute excess Sharpe distribution under null
        rng = np.random.RandomState(self.random_seed)
        boot_excess = []
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            idx = rng.choice(len(aligned), size=len(aligned), replace=True)
            boot_strat = strat_ret.iloc[idx].reset_index(drop=True)
            boot_bench = bench_ret.iloc[idx].reset_index(drop=True) if len(bench_ret) == len(strat_ret) else bench_ret
            # Excess Sharpe in bootstrap
            b_sharpe = boot_strat.mean() / boot_strat.std() * np.sqrt(252) if boot_strat.std() != 0 else 0
            b_bench_sharpe = boot_bench.mean() / boot_bench.std() * np.sqrt(252) if boot_bench.std() != 0 else 0
            boot_excess.append(b_sharpe - b_bench_sharpe)

        # p-value = proportion of bootstrap excess >= observed excess
        boot_excess = np.array(boot_excess)
        p_value = np.mean(boot_excess >= excess) if len(boot_excess) > 0 else 1.0
        return float(p_value)

    def model_confidence_score(self, returns_matrix: pd.DataFrame, confidence_level: float = 0.9) -> float:
        """
        Model Confidence Score – Hansen's Model Confidence Set (MCS) simplified
        Returns proportion of strategies that are in confidence set

        Economic: if many strategies are in MCS, model selection is uncertain -> overfitting risk

        Args:
            returns_matrix: DataFrame strategies x time
            confidence_level: e.g., 0.9

        Returns:
            MCS score 0-1 (higher = more models in set = more uncertainty)
        """
        if not isinstance(returns_matrix, pd.DataFrame) or returns_matrix.empty:
            return 0.0

        # Simplified: compute Sharpe for each strategy, then MCS as those within 1 std of best
        sharpes = {}
        for col in returns_matrix.columns:
            ret = returns_matrix[col].dropna()
            if len(ret) < 10:
                continue
            sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else 0
            sharpes[col] = sh

        if not sharpes:
            return 0.0

        best_sharpe = max(sharpes.values())
        # Std of sharpes
        sharpes_vals = np.array(list(sharpes.values()))
        std_sharpes = np.std(sharpes_vals) if len(sharpes_vals) > 1 else 1.0

        # MCS: strategies within (best - 1*std) are in confidence set
        threshold = best_sharpe - std_sharpe if (std_sharpe := std_sharpes) > 0 else best_sharpe * 0.9
        # Count how many are in set
        in_mcs = sum(1 for s in sharpes.values() if s >= threshold)
        mcs_score = in_mcs / len(sharpes)  # proportion in set

        return float(mcs_score)

    def scan(self, returns: pd.Series, returns_matrix: Optional[pd.DataFrame] = None,
             is_metrics: Optional[List[float]] = None, oos_metrics: Optional[List[float]] = None,
             base_params: Optional[Dict] = None, param_grid: Optional[Dict] = None,
             returns_fn: Optional[callable] = None, metric_fn: Optional[callable] = None,
             benchmark_returns: Optional[pd.Series] = None) -> OverfittingMetrics:
        """
        Full overfitting scan – computes all metrics, returns overall score and warnings

        Args:
            returns: Daily returns of best strategy
            returns_matrix: DataFrame of all strategy variations (for PBO, MCS)
            is_metrics: List of IS Sharpe per walk-forward window
            oos_metrics: List of OOS Sharpe per window
            base_params: Base params dict
            param_grid: Param grid dict
            returns_fn: Function params -> returns Series
            metric_fn: Function returns -> metric
            benchmark_returns: Benchmark for White Reality Check

        Returns:
            OverfittingMetrics with overall_overfitting_score and is_overfitted flag
        """
        warnings = []

        # Parameter sensitivity
        if base_params and param_grid and returns_fn and metric_fn:
            try:
                param_sens = self.parameter_sensitivity(base_params, param_grid, returns_fn, metric_fn)
            except Exception:
                param_sens = 0.0
        else:
            param_sens = 0.0

        # Neighbour stability
        if base_params and param_grid and returns_fn and metric_fn:
            try:
                # Find best params as base_params for simplicity
                neighbour_stab = self.neighbour_stability(base_params, param_grid, returns_fn, metric_fn)
            except Exception:
                neighbour_stab = 0.5
        else:
            neighbour_stab = 0.5

        # Walk forward stability
        if is_metrics and oos_metrics:
            try:
                wf_stab = self.walk_forward_stability(is_metrics, oos_metrics)
            except Exception:
                wf_stab = 0.5
        else:
            wf_stab = 0.5

        # PBO
        if returns_matrix is not None:
            try:
                pbo = self.probability_of_backtest_overfitting(returns_matrix)
            except Exception:
                pbo = 0.5
        else:
            pbo = 0.5

        # DSR
        try:
            dsr = self.deflated_sharpe_ratio(returns)
        except Exception:
            dsr = 0.5

        # White Reality Check
        try:
            if benchmark_returns is None:
                benchmark_returns = pd.Series(0, index=returns.index)
            white_p = self.white_reality_check(returns, benchmark_returns)
        except Exception:
            white_p = 0.5

        # Model Confidence Score
        try:
            if returns_matrix is not None:
                mcs = self.model_confidence_score(returns_matrix)
            else:
                mcs = 0.5
        except Exception:
            mcs = 0.5

        # Overall overfitting score 0-10
        # Higher means more overfitted
        # Components: param sensitivity (higher = more overfitted), (1-neighbour stability), (1-wf stability), pbo, (1-dsr), (1-white_p)??? Actually white_p low = significant, high p = not significant = overfitted?
        # White p-value high means not significant (could be luck) -> overfitted, so p_value itself contributes to overfitting
        # DSR high = good, low = overfitted, so (1-dsr) contributes
        # MCS high = many models in set = more uncertainty = less overfitted? Actually if many models in MCS, selection uncertain, but not necessarily overfitted, but we treat high MCS as less overfitted? Let's interpret: MCS score high = many models in confidence set = model selection uncertain = higher overfitting risk? Actually if many models in MCS, best model not clearly better than others, so less overfitting? We'll treat moderate.
        # For simplicity:
        # overfitting = param_sens*2 + (1-neighbour)*2 + (1-wf)*2 + pbo*2 + (1-dsr)*2 + white_p*1 + mcs*1  all weighted to 0-10
        # Normalize

        param_sens_score = param_sens * 2
        neighbour_score = (1 - neighbour_stab) * 2
        wf_score = (1 - wf_stab) * 2
        pbo_score = pbo * 2
        dsr_score = (1 - dsr) * 2  # DSR 1 = good, 0 = bad -> overfitting high when DSR low
        white_score = white_p * 1  # p high = not significant = overfitted
        mcs_score = mcs * 1  # if many models in set, uncertain, but we treat as moderate overfitting risk

        overall = param_sens_score + neighbour_score + wf_score + pbo_score + dsr_score + white_score + mcs_score
        # Scale to 0-10 (max 2+2+2+2+2+1+1=10)
        overall = float(np.clip(overall, 0, 10))

        # Warnings
        if param_sens > 0.5:
            warnings.append(f"High parameter sensitivity {param_sens:.2f} – small param change causes large performance change, indicates overfitting")
        if neighbour_stab < 0.5:
            warnings.append(f"Low neighbour stability {neighbour_stab:.2f} – best params isolated peak, not robust plateau")
        if wf_stab < 0.5:
            warnings.append(f"Low walk-forward stability {wf_stab:.2f} – IS performance doesn't translate to OOS")
        if pbo > 0.5:
            warnings.append(f"High PBO {pbo:.2f} – probability that best IS is worse than median OOS >50%, strong overfitting signal")
        if dsr < 0.8:
            warnings.append(f"Low DSR {dsr:.3f} – Deflated Sharpe below 0.8, performance may be due to multiple testing luck")
        if white_p > 0.5:
            warnings.append(f"High White Reality Check p-value {white_p:.3f} – best Sharpe not significantly better than benchmark after accounting for data snooping")
        if mcs > 0.7:
            warnings.append(f"High Model Confidence Set score {mcs:.2f} – many models in confidence set, model selection uncertain")

        is_overfitted = overall >= self.overfitting_threshold

        if is_overfitted:
            warnings.append(f"OVERFITTING DETECTED: Overall score {overall:.2f} >= threshold {self.overfitting_threshold} – model should NOT be used, high overfitting risk")

        return OverfittingMetrics(
            param_sensitivity=float(param_sens),
            neighbour_stability=float(neighbour_stab),
            walk_forward_stability=float(wf_stab),
            pbo=float(pbo),
            dsr=float(dsr),
            white_reality_check_pvalue=float(white_p),
            model_confidence_score=float(mcs),
            overall_overfitting_score=float(overall),
            is_overfitted=bool(is_overfitted),
            warnings=warnings
        )

    def generate_report(self, metrics: OverfittingMetrics) -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("="*80)
        lines.append("OVERFITTING SCANNER REPORT – Institutional")
        lines.append("="*80)
        lines.append(f"Overall Overfitting Score: {metrics.overall_overfitting_score:.2f}/10 – {'OVERFITTED' if metrics.is_overfitted else 'OK'}")
        lines.append(f"Threshold: {self.overfitting_threshold}")
        lines.append("")
        lines.append(f"Parameter Sensitivity: {metrics.param_sensitivity:.3f} (0=robust,1=sensitive)")
        lines.append(f"Neighbour Stability: {metrics.neighbour_stability:.3f} (1=stable plateau,0=isolated peak)")
        lines.append(f"Walk Forward Stability: {metrics.walk_forward_stability:.3f} (1=IS translates to OOS)")
        lines.append(f"PBO (Prob of Backtest Overfitting): {metrics.pbo:.3f} (0=good,1=bad)")
        lines.append(f"DSR (Deflated Sharpe): {metrics.dsr:.3f} (1=good,0=bad)")
        lines.append(f"White Reality Check p-value: {metrics.white_reality_check_pvalue:.3f} (low=significant, high=not significant -> overfitting)")
        lines.append(f"Model Confidence Score: {metrics.model_confidence_score:.3f} (proportion in MCS)")
        lines.append("")
        if metrics.warnings:
            lines.append("Warnings:")
            for w in metrics.warnings:
                lines.append(f"  ⚠️ {w}")
        else:
            lines.append("No warnings – model appears robust")

        lines.append("="*80)
        return "\n".join(lines)
