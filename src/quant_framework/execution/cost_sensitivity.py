"""
Execution Engine – Cost Sensitivity Analysis
Tests: 0 bp, 2 bp, 5 bp, 10 bp, 20 bp, 50 bp, 100 bp
Computes: CAGR sensitivity, Sharpe sensitivity, Turnover sensitivity
Result as heatmap
"""

from typing import List, Dict, Callable, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass(frozen=True)
class CostSensitivityResult:
    cost_bps: List[float]
    cagr_list: List[float]
    sharpe_list: List[float]
    turnover_list: List[float]
    cagr_sensitivity: float
    sharpe_sensitivity: float
    turnover_sensitivity: float

    def to_dict(self):
        return {
            "cost_bps": self.cost_bps,
            "cagr_list": self.cagr_list,
            "sharpe_list": self.sharpe_list,
            "turnover_list": self.turnover_list,
            "cagr_sensitivity": self.cagr_sensitivity,
            "sharpe_sensitivity": self.sharpe_sensitivity,
            "turnover_sensitivity": self.turnover_sensitivity
        }


class CostSensitivityAnalyzer:
    """
    Cost Sensitivity Analyzer – tests execution costs 0,2,5,10,20,50,100 bp
    Computes sensitivity of CAGR, Sharpe, Turnover to costs
    Shows result as heatmap
    """

    def __init__(self, cost_bps_list: List[float] = None):
        if cost_bps_list is None:
            cost_bps_list = [0, 2, 5, 10, 20, 50, 100]
        if not isinstance(cost_bps_list, list) or not cost_bps_list:
            raise ValueError("cost_bps_list must be non-empty list")
        # Deterministic sorted
        self.cost_bps_list = sorted(list(dict.fromkeys(cost_bps_list)))

    def analyze(self, backtest_fn: Callable[[float], Dict], metric_fn: Callable[[Dict], Dict]) -> CostSensitivityResult:
        """
        Analyze cost sensitivity

        Args:
            backtest_fn: Function that takes cost_bps (float) and returns backtest result dict with equity, returns, positions etc
            metric_fn: Function that takes backtest result dict and returns dict with cagr, sharpe, turnover

        Returns:
            CostSensitivityResult

        Raises:
            TypeError
        """
        if not callable(backtest_fn) or not callable(metric_fn):
            raise TypeError("backtest_fn and metric_fn must be callable")

        cagr_list = []
        sharpe_list = []
        turnover_list = []

        for cost in self.cost_bps_list:
            try:
                result = backtest_fn(float(cost))
                metrics = metric_fn(result)
                cagr_list.append(float(metrics.get('cagr', 0)))
                sharpe_list.append(float(metrics.get('sharpe', 0)))
                turnover_list.append(float(metrics.get('turnover', 0)))
            except Exception as e:
                # On error, use 0
                cagr_list.append(0.0)
                sharpe_list.append(0.0)
                turnover_list.append(0.0)

        # Sensitivity = (max - min) / |mean| or std/mean – how much metric changes with cost
        def sensitivity(values: List[float]) -> float:
            if not values:
                return 0.0
            arr = np.array(values)
            mean = np.mean(arr)
            if mean == 0:
                return 0.0
            # Range sensitivity
            rng = np.max(arr) - np.min(arr)
            sens = abs(rng) / (abs(mean) + 1e-8)
            return float(sens)

        cagr_sens = sensitivity(cagr_list)
        sharpe_sens = sensitivity(sharpe_list)
        turnover_sens = sensitivity(turnover_list)

        return CostSensitivityResult(
            cost_bps=self.cost_bps_list,
            cagr_list=cagr_list,
            sharpe_list=sharpe_list,
            turnover_list=turnover_list,
            cagr_sensitivity=float(cagr_sens),
            sharpe_sensitivity=float(sharpe_sens),
            turnover_sensitivity=float(turnover_sens)
        )

    def plot_heatmap(self, result: CostSensitivityResult, output_path: str = "reports/cost_sensitivity_heatmap.png") -> str:
        """
        Plot heatmap of cost sensitivity – deterministic

        Args:
            result: CostSensitivityResult
            output_path: Path to save PNG

        Returns:
            Path to saved PNG
        """
        from pathlib import Path
        import matplotlib.pyplot as plt

        # Create DataFrame for heatmap: rows = costs, columns = metrics normalized
        # Normalize each metric to 0-1 for heatmap visibility
        def normalize(lst):
            arr = np.array(lst)
            min_v = np.min(arr)
            max_v = np.max(arr)
            if max_v == min_v:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)

        cagr_norm = normalize(result.cagr_list)
        sharpe_norm = normalize(result.sharpe_list)
        turnover_norm = normalize(result.turnover_list)

        # For heatmap, we want rows = costs, columns = metrics, values = normalized
        data = np.array([cagr_norm, sharpe_norm, turnover_norm]).T
        # data shape (n_costs, 3)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['CAGR', 'Sharpe', 'Turnover'])
        ax.set_yticks(range(len(result.cost_bps)))
        ax.set_yticklabels([f"{c} bp" for c in result.cost_bps])

        ax.set_title('Cost Sensitivity Heatmap (normalized 0-1, green=high, red=low)', fontweight='bold')

        # Annotate values
        for i in range(len(result.cost_bps)):
            for j in range(3):
                if j == 0:
                    orig_val = result.cagr_list[i]
                elif j == 1:
                    orig_val = result.sharpe_list[i]
                else:
                    orig_val = result.turnover_list[i]
                ax.text(j, i, f"{orig_val:.2f}", ha='center', va='center', fontsize=8, color='black')

        plt.colorbar(im, ax=ax, label='Normalized')

        # Add sensitivity text
        fig.text(0.5, 0.01, f"CAGR sens {result.cagr_sensitivity:.3f} | Sharpe sens {result.sharpe_sensitivity:.3f} | Turnover sens {result.turnover_sensitivity:.3f} (lower = more robust)", ha='center', fontsize=9)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        return str(output_path)

    def generate_report(self, result: CostSensitivityResult) -> str:
        """Human-readable report"""
        lines = []
        lines.append("="*80)
        lines.append("EXECUTION COST SENSITIVITY REPORT")
        lines.append("="*80)
        lines.append(f"Costs tested: {result.cost_bps} bp")
        lines.append("")
        lines.append(f"{'Cost bp':<10} {'CAGR %':<12} {'Sharpe':<10} {'Turnover':<12}")
        lines.append("-"*60)
        for i, cost in enumerate(result.cost_bps):
            cagr = result.cagr_list[i] * 100 if i < len(result.cagr_list) else 0
            sharpe = result.sharpe_list[i] if i < len(result.sharpe_list) else 0
            turnover = result.turnover_list[i] if i < len(result.turnover_list) else 0
            lines.append(f"{cost:<10} {cagr:<12.2f} {sharpe:<10.2f} {turnover:<12.2f}")
        lines.append("")
        lines.append(f"CAGR sensitivity: {result.cagr_sensitivity:.4f} (lower = more robust to costs)")
        lines.append(f"Sharpe sensitivity: {result.sharpe_sensitivity:.4f}")
        lines.append(f"Turnover sensitivity: {result.turnover_sensitivity:.4f}")
        lines.append("")
        if result.cagr_sensitivity > 0.5:
            lines.append("⚠️ High CAGR sensitivity to costs – strategy may be overfitted to low costs, fragile")
        else:
            lines.append("✅ Low CAGR sensitivity – robust to transaction costs")
        if result.sharpe_sensitivity > 0.5:
            lines.append("⚠️ High Sharpe sensitivity – risk-adjusted returns degrade quickly with costs")
        else:
            lines.append("✅ Sharpe robust to costs")

        lines.append("="*80)
        return "\n".join(lines)
