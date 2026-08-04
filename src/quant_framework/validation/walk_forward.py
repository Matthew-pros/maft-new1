
"""
10 Walk Forward Validation
"""

from typing import List, Dict, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

class WalkForwardValidator:
    def __init__(self, train_days: int = 252*3, test_days: int = 252, step_days: int = 63):
        if train_days <= 0 or test_days <= 0 or step_days <= 0:
            raise ValueError("train_days, test_days, step_days must be >0")
        self.train_days = int(train_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days)

    def generate_windows(self, start: str, end: str) -> List[WalkForwardWindow]:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= end_dt:
            raise ValueError("start must be < end")
        windows = []
        current_train_start = start_dt
        while True:
            train_end = current_train_start + pd.Timedelta(days=self.train_days)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=self.test_days)
            if test_end > end_dt:
                break
            windows.append(WalkForwardWindow(train_start=current_train_start, train_end=train_end, test_start=test_start, test_end=test_end))
            current_train_start = current_train_start + pd.Timedelta(days=self.step_days)
        return sorted(windows, key=lambda w: w.train_start)

    def validate(self, data: pd.DataFrame, strategy_fn: Callable, metric_fn: Callable) -> Dict:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("data must be non-empty DataFrame")
        if not callable(strategy_fn) or not callable(metric_fn):
            raise TypeError("strategy_fn and metric_fn must be callable")
        windows = self.generate_windows(start=str(data.index.min().date()), end=str(data.index.max().date()))
        results = []
        oos_scores = []
        for w in windows:
            train = data[(data.index >= w.train_start) & (data.index <= w.train_end)]
            test = data[(data.index >= w.test_start) & (data.index <= w.test_end)]
            if train.empty or test.empty:
                continue
            try:
                model = strategy_fn(train)
                score = metric_fn(test, model)
                results.append({"train_start": w.train_start, "train_end": w.train_end, "test_start": w.test_start, "test_end": w.test_end, "oos_score": float(score), "train_len": len(train), "test_len": len(test)})
                oos_scores.append(float(score))
            except Exception:
                continue
        avg_oos = float(np.mean(oos_scores)) if oos_scores else 0.0
        return {"windows": results, "avg_oos_score": avg_oos, "num_windows": len(results), "oos_scores": oos_scores}
