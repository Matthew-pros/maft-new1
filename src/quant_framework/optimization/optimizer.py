
"""
09 Optimization – grid/random search, deterministic, only on train
"""

from typing import Dict, List, Callable, Any
import itertools
import numpy as np

class Optimizer:
    def __init__(self, param_grid: Dict[str, List[Any]], random_seed: int = 42):
        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError("param_grid must be non-empty dict")
        self.param_grid = {k: sorted(list(dict.fromkeys(v))) for k, v in param_grid.items()}
        self.random_seed = int(random_seed)

    def grid_combinations(self) -> List[Dict[str, Any]]:
        keys = sorted(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return sorted(combos, key=lambda x: str(x))

    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], maximize: bool = True) -> Dict:
        if not callable(objective_fn):
            raise TypeError("objective_fn must be callable")
        combos = self.grid_combinations()
        results = []
        best_score = -np.inf if maximize else np.inf
        best_params = None
        for params in combos:
            try:
                score = objective_fn(params)
                if not isinstance(score, (int, float)) or not np.isfinite(score):
                    continue
                results.append({"params": params, "score": float(score)})
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
            except Exception:
                continue
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=maximize)
        return {"best_params": best_params, "best_score": float(best_score) if best_params else None, "all_results": results_sorted, "total_evaluated": len(results_sorted)}
