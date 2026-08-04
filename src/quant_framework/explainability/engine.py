"""
Explainability Engine – institutional level
Every portfolio decision must be explained:
Date, Signal, Canaries, Votes, Confidence, Regime, Portfolio Weights, Risk Budget, Execution
Output human-readable. Implements Decision Log.

Design principles (quant research engineer perspective):
- Economic justification: Every decision must be explainable in economic terms, not just "model said so"
- No lookahead: Explanation uses only information available at decision time
- Reproducibility: Decision log contains all inputs needed to reproduce decision
- Modularity: Explainability is separate from signal/regime/portfolio engines, consumes their outputs, does not generate signals itself
- Testability: DecisionLogEntry is frozen dataclass, deterministic, can be unit tested
- Extensibility: New canaries, regimes, risk metrics can be added to explanation without modifying engine

Why needed in institutional asset management?
- Compliance and risk oversight require human-readable rationale for each trade
- Portfolio managers need to understand why allocation changed, not just that it changed
- Post-trade attribution: was performance due to market, factor, regime, or execution?
- Model debugging: if performance degrades, decision log shows which component (signal, regime, risk budget) caused it
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime


@dataclass(frozen=True)
class DecisionLogEntry:
    """
    Immutable decision log entry – one row per decision date
    All fields human-readable, no lookahead, deterministic
    """
    date: pd.Timestamp
    signal_summary: str  # e.g., "TQQQ LONG (EMA5>EMA50 Cross, MACD cross), QQQ HOLD"
    canaries: List[Dict]  # List of canary results dicts (name, signal, confidence, value)
    votes: Dict[str, int]  # e.g., {"RiskOn": 2, "RiskOff": 3}
    confidence: float  # overall confidence 0-1
    regime: str  # final regime e.g., RISK_ON_GROWTH
    regime_layers: Dict[str, str]  # layer_name -> regime, e.g., {"MacroRegime": "Expansion", "VolatilityRegime": "LowVol"}
    portfolio_weights: Dict[str, float]  # ticker -> weight
    risk_budget: Dict[str, float]  # ticker -> risk contribution % or deviation
    execution: Dict[str, Any]  # e.g., {"fills": 3, "total_commission": 1.23, "slippage_bps": 2, "cash_after": 98500}
    human_readable: str = ""  # Full narrative explanation

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['date'] = str(self.date)
        return d

    def to_human_text(self) -> str:
        """Return human-readable text, with fallback to stored human_readable"""
        if self.human_readable:
            return self.human_readable
        # Generate basic text if not provided
        canary_str = ", ".join([f"{c.get('name','?')}:{c.get('signal','?')}({c.get('confidence',0):.2f})" for c in self.canaries[:3]])
        weights_str = ", ".join([f"{k} {v*100:.1f}%" for k, v in sorted(self.portfolio_weights.items()) if v > 0.01][:5])
        return (
            f"On {self.date.date()}, Signal {self.signal_summary}. "
            f"Canaries: {canary_str}. Votes {self.votes}. Confidence {self.confidence:.2f}. "
            f"Regime {self.regime} (layers {self.regime_layers}). "
            f"Portfolio: {weights_str}. Risk Budget: {self.risk_budget}. "
            f"Execution: {self.execution}"
        )


class ExplainabilityEngine:
    """
    Explainability Engine – collects and explains portfolio decisions
    - No business logic for signal generation, only explanation
    - Consumes outputs from SignalEngine, RegimeEngine (multi-layer), PortfolioConstructor, RiskEngine, ExecutionSimulator
    - Produces Decision Log – list of DecisionLogEntry, exportable to CSV/JSON/HTML
    - Human-readable output required for institutional oversight

    Economic justification:
    - Every allocation must be traceable to economic factor (e.g., "XLU outperforming SPY 1.2% indicates defensive rotation, money flowing to regulated income, revealed preference for safety")
    - Votes and confidence show uncertainty, not false precision
    - Risk budget shows if portfolio is taking unintended risk concentration
    """

    def __init__(self):
        self._logs: List[DecisionLogEntry] = []

    def log_decision(self,
                     date: pd.Timestamp,
                     signals: Dict[str, Dict],  # ticker -> {signal, confidence} or signal_summary str
                     canaries: List[Dict],
                     votes: Dict[str, int],
                     confidence: float,
                     regime: str,
                     regime_layers: Dict[str, str],
                     portfolio_weights: Dict[str, float],
                     risk_budget: Dict[str, float],
                     execution: Dict[str, Any],
                     human_readable: Optional[str] = None) -> DecisionLogEntry:
        """
        Log a single decision – deterministic, validated, no lookahead

        Args:
            date: Decision date (must be past, not future)
            signals: Dict ticker -> signal info or summary string
            canaries: List of canary result dicts (from RegimeEngine)
            votes: Dict signal -> count (e.g., RiskOn:2, RiskOff:3)
            confidence: Overall confidence 0-1
            regime: Final regime string
            regime_layers: Dict layer_name -> regime string (7 layers)
            portfolio_weights: Dict ticker -> weight
            risk_budget: Dict ticker -> risk contribution deviation or risk budget
            execution: Dict execution info (fills, commission, slippage, cash)
            human_readable: Optional pre-built human explanation, if None will be auto-generated

        Returns:
            DecisionLogEntry immutable

        Raises:
            TypeError, ValueError
        """
        if not isinstance(date, pd.Timestamp):
            try:
                date = pd.to_datetime(date)
            except Exception:
                raise TypeError("date must be Timestamp or parseable string")
        if not isinstance(signals, (dict, str)):
            raise TypeError("signals must be dict or str")
        if isinstance(signals, dict):
            # Convert signals dict to summary string deterministically
            signal_summary = ", ".join([f"{t}:{v.get('signal',v) if isinstance(v, dict) else v}" for t, v in sorted(signals.items())][:10])
        else:
            signal_summary = signals

        if not isinstance(canaries, list):
            raise TypeError("canaries must be list")
        if not isinstance(votes, dict):
            raise TypeError("votes must be dict")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        if not isinstance(regime, str) or not regime:
            raise ValueError("regime must be non-empty str")
        if not isinstance(regime_layers, dict):
            raise TypeError("regime_layers must be dict")
        if not isinstance(portfolio_weights, dict):
            raise TypeError("portfolio_weights must be dict")
        if not isinstance(risk_budget, dict):
            raise TypeError("risk_budget must be dict")
        if not isinstance(execution, dict):
            raise TypeError("execution must be dict")

        # Auto-generate human-readable if not provided – economic justification
        if human_readable is None:
            # Build economic narrative
            canary_econ = []
            for c in canaries[:4]:  # top 4 canaries
                name = c.get('name', 'Unknown')
                sig = c.get('signal', 'Neutral')
                val = c.get('value', 0)
                conf = c.get('confidence', 0)
                # Economic interpretation per canary
                if "XLU" in name:
                    econ = f"XLU vs SPY {val*100:+.2f}% indicates {'defensive rotation (money to regulated income, safety)' if sig=='RiskOff' else 'risk-on, utilities underperforming'}"
                elif "QQQ" in name:
                    econ = f"QQQ vs SPY {val*100:+.2f}% indicates {'tech leadership, growth premium, risk-on' if sig=='RiskOn' else 'tech underperforming, broadening or risk-off'}"
                elif "Credit" in name:
                    econ = f"Credit {val*100:+.2f}% indicates {'credit stress, lenders demanding safety, equity at risk' if sig=='RiskOff' else 'credit risk appetite, plumbing functioning'}"
                elif "VIX" in name:
                    econ = f"VIX {val:.1f} indicates {'fear, tail risk hedging' if sig in ['RiskOff','BlackSwan'] else 'complacency, low fear'}"
                else:
                    econ = f"{name} signal {sig} value {val:.4f}"
                canary_econ.append(f"{name} {sig} (conf {conf:.2f}): {econ}")

            canary_text = "; ".join(canary_econ)
            weights_text = ", ".join([f"{k} {v*100:.1f}%" for k, v in sorted(portfolio_weights.items()) if v > 0.005][:6])
            risk_text = ", ".join([f"{k} {v*100:+.1f}% vs target" if isinstance(v, float) else f"{k} {v}" for k, v in sorted(risk_budget.items()) if abs(v) > 0.001][:5])

            human_readable = (
                f"On {date.date()}, portfolio decision: {signal_summary}. "
                f"Canaries: {canary_text}. "
                f"Votes: {votes} (RiskOn vs RiskOff count). Overall confidence {confidence:.2f}. "
                f"Regime: {regime} with layers {regime_layers}. "
                f"Macro regime indicates {'expansion, capex cycle growing, favors cyclicals' if 'Expansion' in str(regime_layers.values()) else 'contraction or neutral'}. "
                f"Volatility regime indicates {'low vol complacency' if 'LowVol' in str(regime_layers.values()) else 'high vol stress'}. "
                f"Portfolio weights: {weights_text}. "
                f"Risk budget: {risk_text if risk_text else 'within target'}. "
                f"Execution: {execution.get('fills', 0)} fills, commission ${execution.get('total_commission',0):.2f}, slippage {execution.get('slippage_bps',0)} bps, cash after ${execution.get('cash_after',0):.0f}. "
                f"Economic rationale: Allocation tilts to {'risk-on growth (Tech, Mega-cap) when HYG/IEF rising and XLY/XLP rising and VIX<18' if 'RISK_ON' in regime else 'defensive (Utilities, Staples, Treasuries) when XLU/SPY rising and credit weakening' if 'DEFENSIVE' in regime or 'CREDIT_STRESS' in regime else 'inflation hedges (Energy, Materials, Gold) when XLE/XLU rising' if 'INFLATION' in regime else 'cash and long vol when VIX>=30'}."
            )

        entry = DecisionLogEntry(
            date=pd.to_datetime(date),
            signal_summary=signal_summary,
            canaries=canaries,
            votes=votes,
            confidence=float(confidence),
            regime=regime,
            regime_layers=regime_layers,
            portfolio_weights=dict(sorted(portfolio_weights.items())),
            risk_budget=dict(sorted(risk_budget.items())),
            execution=execution,
            human_readable=human_readable
        )

        self._logs.append(entry)
        # Keep logs sorted deterministically by date
        self._logs = sorted(self._logs, key=lambda x: x.date)

        return entry

    def get_decision_log(self) -> pd.DataFrame:
        """Return decision log as DataFrame human-readable, sorted by date"""
        if not self._logs:
            return pd.DataFrame()
        rows = []
        for entry in self._logs:
            rows.append({
                "date": entry.date,
                "signal": entry.signal_summary,
                "regime": entry.regime,
                "confidence": entry.confidence,
                "votes": str(entry.votes),
                "weights": str({k: f"{v*100:.1f}%" for k, v in entry.portfolio_weights.items() if v > 0.01}),
                "risk_budget": str(entry.risk_budget),
                "execution": str(entry.execution),
                "human_readable": entry.human_readable
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df

    def get_full_log(self) -> List[DecisionLogEntry]:
        """Return full list of immutable entries sorted by date"""
        return sorted(self._logs, key=lambda x: x.date)

    def export_csv(self, path: str = "reports/decision_log.csv") -> str:
        """Export decision log to CSV – reproducible"""
        from pathlib import Path
        df = self.get_decision_log()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

    def export_json(self, path: str = "reports/decision_log.json") -> str:
        """Export full log to JSON with all fields"""
        from pathlib import Path
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.get_full_log()]
        with open(p, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return str(p)

    def export_html(self, path: str = "reports/decision_log.html") -> str:
        """Export human-readable HTML report"""
        from pathlib import Path
        df = self.get_decision_log()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Decision Log – Explainability Engine</title>
<style>body{{font-family:sans-serif;background:#f9fafb;padding:20px}} .entry{{background:white;border:1px solid #e5e7eb;padding:16px;border-radius:8px;margin:16px 0}} h2{{color:#1f2937}} .meta{{color:#6b7280;font-size:0.9em}} .human{{background:#f3f4f6;padding:12px;border-radius:6px;margin-top:8px}}</style></head><body>
<h1>Explainability Engine – Decision Log</h1>
<p>Each portfolio decision explained with economic justification – institutional oversight</p>
"""
        for _, row in df.iterrows():
            html += f"""<div class="entry">
<h2>{row['date']} – {row['regime']} (conf {row['confidence']})</h2>
<div class="meta\">Signal: {row['signal']} | Votes: {row['votes']} | Weights: {row['weights']} | Risk: {row['risk_budget']} | Execution: {row['execution']}</div>
<div class="human">{row['human_readable']}</div>
</div>
"""
        html += "</body></html>"
        p.write_text(html, encoding='utf-8')
        return str(p)

    def clear(self) -> None:
        """Clear log"""
        self._logs.clear()
