"""
Lookahead Scanner – automatically detects future data leakage
Checks:
- future prices
- future rolling windows
- future ranking
- future z-score
- future smoothing
- future execution

If lookahead found: stop backtest, print problem, suggest fix – institutional
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import re
import inspect
from pathlib import Path


@dataclass(frozen=True)
class LookaheadIssue:
    type: str  # future_prices, future_rolling, future_ranking, future_zscore, future_smoothing, future_execution
    severity: str  # low, medium, high, critical
    description: str
    location: str  # file or code location
    suggestion: str

    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion
        }


class LookaheadScanner:
    """
    Lookahead Scanner – institutional, deterministic, no randomness
    Scans code and data pipeline for lookahead bias
    """

    def __init__(self, fail_on_critical: bool = True):
        self.fail_on_critical = bool(fail_on_critical)

    @staticmethod
    def _check_execution_delay(execution_config) -> List[LookaheadIssue]:
        issues = []
        try:
            delay = getattr(execution_config, 'execution_delay', 1)
            price_type = getattr(execution_config, 'execution_price', 'next_open')
            if delay == 0:
                issues.append(LookaheadIssue(
                    type="future_execution",
                    severity="critical",
                    description=f"Execution delay is 0 with price {price_type} – signal generated at bar close executed at same bar, requires future price known at time of signal",
                    location="ExecutionConfig.execution_delay",
                    suggestion="Set execution_delay=1 and execution_price='next_open' to ensure no lookahead: signal at close t, execution at open t+1"
                ))
            elif delay < 1:
                issues.append(LookaheadIssue(
                    type="future_execution",
                    severity="high",
                    description=f"Execution delay {delay} <1 – fractional delay may still be lookahead",
                    location="ExecutionConfig.execution_delay",
                    suggestion="Use integer delay >=1"
                ))
        except Exception:
            pass
        return issues

    @staticmethod
    def _scan_code_for_future_patterns(file_path: Path) -> List[LookaheadIssue]:
        """Scan Python file for common lookahead patterns"""
        issues = []
        if not file_path.exists() or not file_path.is_file():
            return issues

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return issues

        # Patterns that indicate potential lookahead
        patterns = [
            (r"\.shift\s*\(\s*-1\s*\)", "future_prices", "Uses shift(-1) – accesses future bar", "Replace shift(-1) with shift(1) for past"),
            (r"center\s*=\s*True", "future_rolling", "Rolling window with center=True uses future data in window", "Use center=False (default) for past-only rolling"),
            (r"Close\[.*\+.*\]", "future_prices", "Direct future indexing Close[i+1]", "Use Close.shift(1) or past only"),
            (r"future.*return", "future_prices", "Variable name contains 'future' and 'return' – possible future returns used", "Ensure returns are past only, use pct_change().shift(1) for signal"),
            (r"z_score.*future|future.*z_score", "future_zscore", "Z-score may use future mean/std if window includes future", "Use rolling(window).mean() with past only, not expanding with future"),
            (r"rank.*future|future.*rank", "future_ranking", "Ranking may use future returns", "Use rank of past returns only"),
            (r"smoothing.*future|future.*smoothing|ewm.*center", "future_smoothing", "Smoothing may use future", "Use ewm(span, adjust=False) causal, not centered"),
        ]

        for pattern, issue_type, desc, sugg in patterns:
            for i, line in enumerate(content.splitlines(), start=1):
                if re.search(pattern, line, re.IGNORECASE):
                    # Ignore if line is in comment explaining lookahead avoidance
                    stripped = line.strip()
                    if stripped.startswith("#") and "avoid" in stripped.lower():
                        continue
                    issues.append(LookaheadIssue(
                        type=issue_type,
                        severity="medium" if "shift(-1)" not in line else "critical",
                        description=f"{desc} found: {line.strip()[:120]}",
                        location=f"{file_path}:{i}",
                        suggestion=sugg
                    ))
        return issues

    @staticmethod
    def _check_feature_store_for_future(feature_store) -> List[LookaheadIssue]:
        """Check FeatureStore for future rolling windows"""
        issues = []
        try:
            # Check if any feature was computed with center=True – we can inspect cache keys?
            # Our FeatureStore uses rolling(window).mean() without center, so safe
            # But we can check if any feature uses future by inspecting source of feature functions
            from ..features.feature_store import FeatureRegistry
            # For each feature, we know its implementation is past-only, but we can still check
            # If FeatureStore has attribute _feature_cache, we can check if any cached df has index that includes future beyond price data?
            # Simplified: if FeatureStore was used with returns that include future (e.g., returns.shift(-1)), we can't detect easily, so we check for common future patterns in feature_store.py file itself
            import pathlib
            fs_path = pathlib.Path(__file__).parent.parent / "features" / "feature_store.py"
            if fs_path.exists():
                content = fs_path.read_text()
                if "shift(-1)" in content:
                    issues.append(LookaheadIssue(
                        type="future_prices",
                        severity="critical",
                        description="feature_store.py contains shift(-1) – future price access",
                        location=str(fs_path),
                        suggestion="Replace shift(-1) with shift(1)"
                    ))
                if "center=True" in content:
                    issues.append(LookaheadIssue(
                        type="future_rolling",
                        severity="high",
                        description="feature_store.py uses center=True in rolling – future window",
                        location=str(fs_path),
                        suggestion="Use center=False"
                    ))
        except Exception:
            pass
        return issues

    @staticmethod
    def _check_signal_uses_future_close(signal_code_path: Optional[Path] = None) -> List[LookaheadIssue]:
        """Check if signal generation uses Close at same bar without shift for execution at same bar"""
        # This is more of a design check: if signal uses Close and execution is same bar, it's lookahead
        # We already check execution_delay, but we can also flag if signal file uses Close directly without shift
        issues = []
        # For TEMACD, signal uses EMA of Close which includes current close – that's okay if execution is next bar
        # If execution is next bar, using current close is okay (close is known at signal time)
        # So we don't flag this as lookahead if delay>=1
        return issues

    def scan_codebase(self, root_dir: str = "src/quant_framework") -> List[LookaheadIssue]:
        """
        Scan entire codebase for lookahead patterns – deterministic, sorted

        Args:
            root_dir: Root directory to scan

        Returns:
            List of LookaheadIssue sorted by severity and location
        """
        import pathlib
        root = pathlib.Path(root_dir)
        if not root.exists():
            return []

        all_issues = []
        for py_file in sorted(root.rglob("*.py")):
            # Skip __pycache__, tests, etc
            if "__pycache__" in str(py_file):
                continue
            issues = self._scan_code_for_future_patterns(py_file)
            all_issues.extend(issues)

        # Sort deterministically
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_issues = sorted(all_issues, key=lambda x: (severity_order.get(x.severity, 4), x.location))
        return all_issues

    def scan_execution(self, execution_config) -> List[LookaheadIssue]:
        """Scan execution config for future execution"""
        return self._check_execution_delay(execution_config)

    def scan_pipeline(self, data: Dict[str, pd.DataFrame], feature_store=None, signal_engine=None, execution_config=None, codebase_root: str = "src/quant_framework") -> Dict:
        """
        Full pipeline scan – checks future prices, rolling windows, ranking, z-score, smoothing, execution

        Args:
            data: Dict ticker -> DataFrame
            feature_store: FeatureStore instance
            signal_engine: SignalEngine instance
            execution_config: ExecutionConfig
            codebase_root: Root dir for code scan

        Returns:
            Dict with issues list, has_critical, summary, suggestions

        Raises:
            RuntimeError if critical lookahead found and fail_on_critical=True
        """
        all_issues: List[LookaheadIssue] = []

        # 1. Execution check
        if execution_config is not None:
            all_issues.extend(self.scan_execution(execution_config))

        # 2. Codebase scan
        all_issues.extend(self.scan_codebase(root_dir=codebase_root))

        # 3. FeatureStore check
        if feature_store is not None:
            all_issues.extend(self._check_feature_store_for_future(feature_store))

        # 4. Data checks – future prices in data itself? Check if any data has future dates beyond today?
        # Also check if returns use future: e.g., if Close contains NaN at end that would be forward filled from future?
        # For simplicity, check if any DataFrame has index in future vs now
        try:
            now = pd.Timestamp.now(tz='UTC')
            for ticker, df in data.items():
                if df.empty:
                    continue
                if df.index.max() > now + pd.Timedelta(days=5):
                    all_issues.append(LookaheadIssue(
                        type="future_prices",
                        severity="high",
                        description=f"Ticker {ticker} has future dates beyond today: last {df.index.max()} > now",
                        location=f"DataManager {ticker}",
                        suggestion="Ensure data only includes past up to today, no future prices"
                    ))
        except Exception:
            pass

        # Separate by severity
        critical = [i for i in all_issues if i.severity == "critical"]
        high = [i for i in all_issues if i.severity == "high"]
        medium = [i for i in all_issues if i.severity == "medium"]

        has_critical = len(critical) > 0

        summary = {
            "total_issues": len(all_issues),
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "low": len([i for i in all_issues if i.severity == "low"]),
            "has_critical": has_critical,
            "issues": [iss.to_dict() for iss in all_issues],
            "suggestions": list(dict.fromkeys([iss.suggestion for iss in all_issues]))  # dedup preserve order
        }

        if has_critical and self.fail_on_critical:
            # Stop backtest – raise with detailed problem
            problems = "\n".join([f"- [{iss.severity.upper()}] {iss.type} at {iss.location}: {iss.description}" for iss in critical[:5]])
            raise RuntimeError(
                f"CRITICAL LOOKAHEAD BIAS DETECTED – Backtest STOPPED\n"
                f"Found {len(critical)} critical issues:\n{problems}\n"
                f"Suggestions:\n" + "\n".join([f"- {s}" for s in summary["suggestions"][:5]]) +
                f"\nFix all critical issues before running backtest. See full report in summary['issues']"
            )

        return summary

    def generate_report(self, scan_result: Dict) -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("="*80)
        lines.append("LOOKAHEAD SCANNER REPORT – Institutional")
        lines.append("="*80)
        lines.append(f"Total issues: {scan_result['total_issues']}, Critical: {scan_result['critical']}, High: {scan_result['high']}, Medium: {scan_result['medium']}")
        lines.append("")
        if scan_result['has_critical']:
            lines.append("⛔ CRITICAL LOOKAHEAD FOUND – BACKTEST MUST STOP")
        else:
            lines.append("✅ No critical lookahead found")

        lines.append("")
        lines.append("Issues:")
        for iss in scan_result['issues'][:20]:
            lines.append(f"  [{iss['severity']}] {iss['type']} at {iss['location']}: {iss['description']}")

        lines.append("")
        lines.append("Suggestions:")
        for sugg in scan_result['suggestions']:
            lines.append(f"  - {sugg}")

        lines.append("="*80)
        return "\n".join(lines)
