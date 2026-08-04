
"""
Experiment Logger – institutional reproducibility, no OOS optimization
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime
import pandas as pd
import getpass
import sys

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    project_name: str
    hypothesis: str
    config: Dict[str, Any]
    data_hash: str
    code_version: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    universe: List[str]
    features_used: List[str]
    model_name: str
    train_period: str
    test_period: str
    notes: str
    user: str
    python_version: str
    platform: str
    def to_dict(self) -> Dict:
        return asdict(self)
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

class ExperimentLogger:
    def __init__(self, log_dir: str = "reports/experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_data(data: Dict[str, pd.DataFrame]) -> str:
        hasher = hashlib.sha256()
        for ticker in sorted(data.keys()):
            df = data[ticker]
            if df.empty:
                continue
            try:
                first = float(df['Close'].iloc[0])
                last = float(df['Close'].iloc[-1])
                length = len(df)
                hasher.update(f"{ticker}:{first}:{last}:{length}".encode())
            except Exception:
                hasher.update(ticker.encode())
        return hasher.hexdigest()[:16]

    @staticmethod
    def _get_code_version() -> str:
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            return commit[:12]
        except Exception:
            return "unknown"

    def log(self, project_name: str, hypothesis: str, config: Dict, data: Dict[str, pd.DataFrame],
            metrics: Dict, parameters: Dict, universe: List[str], features_used: List[str],
            model_name: str, train_period: str, test_period: str, notes: str = ""):
        if not isinstance(train_period, str) or not isinstance(test_period, str):
            raise TypeError("train_period and test_period must be str")
        try:
            train_start_str, train_end_str = [s.strip() for s in train_period.split("to")]
            test_start_str, test_end_str = [s.strip() for s in test_period.split("to")]
            train_end = pd.to_datetime(train_end_str)
            test_start = pd.to_datetime(test_start_str)
            if train_end >= test_start:
                raise ValueError(f"Train period {train_period} overlaps test {test_period} – OOS optimization forbidden")
        except ValueError as ve:
            if "overlaps" in str(ve) or "OOS" in str(ve):
                raise
        if not hypothesis or len(hypothesis.strip()) < 20:
            raise ValueError("hypothesis must have economic justification >=20 chars")
        if not isinstance(features_used, list) or not features_used:
            raise ValueError("features_used must be non-empty list")
        data_hash = self._hash_data(data)
        code_version = self._get_code_version()
        id_hasher = hashlib.sha256()
        id_hasher.update(json.dumps(config, sort_keys=True).encode())
        id_hasher.update(data_hash.encode())
        id_hasher.update(json.dumps(parameters, sort_keys=True).encode())
        id_hasher.update(model_name.encode())
        experiment_id = id_hasher.hexdigest()[:12]
        record = ExperimentRecord(
            experiment_id=experiment_id, timestamp=datetime.utcnow().isoformat(),
            project_name=project_name, hypothesis=hypothesis, config=config,
            data_hash=data_hash, code_version=code_version, metrics=metrics,
            parameters=parameters, universe=sorted(list(dict.fromkeys(universe))),
            features_used=sorted(list(dict.fromkeys(features_used))), model_name=model_name,
            train_period=train_period, test_period=test_period, notes=notes,
            user=getpass.getuser(), python_version=sys.version, platform=platform.platform()
        )
        filename = f"{record.timestamp[:10]}_{experiment_id}_{model_name.replace(' ', '_')}.json"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        path = self.log_dir / filename
        path.write_text(record.to_json(), encoding='utf-8')
        master_csv = self.log_dir / "master_log.csv"
        flat = {"experiment_id": record.experiment_id, "timestamp": record.timestamp,
                "project": record.project_name, "model": record.model_name,
                "train": record.train_period, "test": record.test_period,
                "data_hash": record.data_hash, "code_version": record.code_version,
                "hypothesis": record.hypothesis[:200],
                "total_return": metrics.get("total_return"), "sharpe": metrics.get("sharpe"),
                "max_dd": metrics.get("max_dd"), "psr": metrics.get("psr")}
        df_row = pd.DataFrame([flat])
        if master_csv.exists():
            df_existing = pd.read_csv(master_csv)
            df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        else:
            df_combined = df_row
        df_combined.to_csv(master_csv, index=False)
        return record

    def load_experiment(self, experiment_id: str):
        if not isinstance(experiment_id, str):
            raise TypeError("experiment_id must be str")
        for file in self.log_dir.glob(f"*{experiment_id}*.json"):
            try:
                data = json.loads(file.read_text())
                return ExperimentRecord(**data)
            except Exception:
                continue
        return None

    def list_experiments(self) -> pd.DataFrame:
        master_csv = self.log_dir / "master_log.csv"
        if master_csv.exists():
            df = pd.read_csv(master_csv)
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()
