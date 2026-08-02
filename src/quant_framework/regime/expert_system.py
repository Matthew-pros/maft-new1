"""
Expert System – declarative rules, no if-else inside notebook, knowledge base easily extensible
"""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from .plugins.base import CanaryResult


@dataclass(frozen=True)
class Condition:
    canary_name: str
    field: str
    operator: str
    value: object

    def __post_init__(self):
        if not isinstance(self.canary_name, str) or not self.canary_name:
            raise ValueError("canary_name must be non-empty str")
        if self.operator not in ["==", "!=", ">", "<", ">=", "<=", "in", "not_in"]:
            raise ValueError(f"Invalid operator {self.operator}")

    def evaluate(self, canary_map: Dict[str, CanaryResult]) -> bool:
        if self.canary_name not in canary_map:
            return False
        cr = canary_map[self.canary_name]
        actual = getattr(cr, self.field, None)
        if actual is None:
            actual = cr.to_dict().get(self.field)
        op = self.operator
        exp = self.value
        try:
            if op == "==":
                return actual == exp
            elif op == "!=":
                return actual != exp
            elif op == ">":
                return float(actual) > float(exp)
            elif op == "<":
                return float(actual) < float(exp)
            elif op == ">=":
                return float(actual) >= float(exp)
            elif op == "<=":
                return float(actual) <= float(exp)
            elif op == "in":
                return actual in exp
            elif op == "not_in":
                return actual not in exp
        except Exception:
            return False
        return False


@dataclass(frozen=True)
class Rule:
    name: str
    conditions: List[Condition]
    logic: str
    conclusion_regime: str
    conclusion_confidence: str
    priority: int = 10
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be non-empty str")
        if self.logic not in ["AND", "OR"]:
            raise ValueError("logic must be AND or OR")
        if self.conclusion_confidence not in ["Low", "Medium", "High"]:
            raise ValueError("conclusion_confidence must be Low/Medium/High")
        if not isinstance(self.conditions, list) or len(self.conditions) == 0:
            raise ValueError("conditions must be non-empty list")

    def evaluate(self, canary_map: Dict[str, CanaryResult]) -> bool:
        results = [c.evaluate(canary_map) for c in self.conditions]
        if self.logic == "AND":
            return all(results)
        else:
            return any(results)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "conditions": [{"canary": c.canary_name, "field": c.field, "op": c.operator, "value": c.value} for c in self.conditions],
            "logic": self.logic,
            "conclusion_regime": self.conclusion_regime,
            "conclusion_confidence": self.conclusion_confidence,
            "priority": self.priority,
            "description": self.description
        }


class KnowledgeBase:
    def __init__(self, rules=None):
        self.rules: List[Rule] = sorted(rules or [], key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: Rule) -> None:
        if not isinstance(rule, Rule):
            raise TypeError("rule must be Rule")
        self.rules.append(rule)
        self.rules = sorted(self.rules, key=lambda r: (r.priority, r.name), reverse=True)

    def remove_rule(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be str")
        self.rules = [r for r in self.rules if r.name != name]

    def list_rules(self) -> List[Dict]:
        return [r.to_dict() for r in sorted(self.rules, key=lambda r: r.priority, reverse=True)]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.rules:
            rows.append({
                "name": r.name,
                "conditions_count": len(r.conditions),
                "logic": r.logic,
                "conclusion_regime": r.conclusion_regime,
                "confidence": r.conclusion_confidence,
                "priority": r.priority,
                "description": r.description
            })
        return pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)


def default_knowledge_base() -> KnowledgeBase:
    rules = [
        Rule(
            name="Defensive Rotation",
            conditions=[
                Condition("XLUCanary", "signal", "==", "RiskOff"),
                Condition("QQQCanary", "signal", "==", "RiskOff"),
                Condition("CreditSpreadCanary", "signal", "==", "RiskOff"),
            ],
            logic="AND",
            conclusion_regime="DEFENSIVE_ROTATION",
            conclusion_confidence="High",
            priority=100,
            description="IF XLU outperforming AND QQQ underperforming AND Credit weak THEN Defensive"
        ),
        Rule(
            name="Risk Off High Confidence",
            conditions=[
                Condition("XLUCanary", "signal", "==", "RiskOff"),
                Condition("QQQCanary", "signal", "==", "RiskOff"),
                Condition("CreditSpreadCanary", "signal", "==", "RiskOff"),
            ],
            logic="AND",
            conclusion_regime="CREDIT_STRESS",
            conclusion_confidence="High",
            priority=90,
            description="XLU outperforming + QQQ underperforming + Credit weak = Risk Off"
        ),
        Rule(
            name="Black Swan",
            conditions=[
                Condition("VIXCanary", "signal", "==", "BlackSwan"),
            ],
            logic="OR",
            conclusion_regime="BLACK_SWAN_VOL_SHOCK",
            conclusion_confidence="High",
            priority=110,
            description="VIX >=30 => Black Swan"
        ),
        Rule(
            name="Risk On Growth",
            conditions=[
                Condition("XLUCanary", "signal", "==", "RiskOn"),
                Condition("QQQCanary", "signal", "==", "RiskOn"),
                Condition("CreditSpreadCanary", "signal", "==", "RiskOn"),
            ],
            logic="AND",
            conclusion_regime="RISK_ON_GROWTH",
            conclusion_confidence="High",
            priority=80,
            description="XLU underperforming + QQQ outperforming + Credit strong = Risk On"
        ),
        Rule(
            name="Inflation Growth",
            conditions=[
                Condition("OilCanary", "signal", "==", "Inflation"),
                Condition("GoldCanary", "signal", "==", "SafeHaven"),
            ],
            logic="OR",
            conclusion_regime="INFLATION_GROWTH",
            conclusion_confidence="Medium",
            priority=70,
            description="Oil or Gold outperforming => Inflation"
        ),
        Rule(
            name="Whipsaw Range",
            conditions=[
                Condition("VIXCanary", "value", ">=", 18),
                Condition("VIXCanary", "value", "<", 25),
            ],
            logic="AND",
            conclusion_regime="RANGE_BOUND_WHIPSAW",
            conclusion_confidence="Medium",
            priority=60,
            description="VIX 18-25 => Range bound"
        ),
    ]
    return KnowledgeBase(rules)


class ExpertSystem:
    def __init__(self, knowledge_base: KnowledgeBase):
        if not isinstance(knowledge_base, KnowledgeBase):
            raise TypeError("knowledge_base must be KnowledgeBase")
        self.kb = knowledge_base

    def evaluate(self, canary_results: List[CanaryResult]) -> Dict:
        if not isinstance(canary_results, list):
            raise TypeError("canary_results must be list")
        canary_map = {cr.name: cr for cr in canary_results}
        fired = []
        for rule in sorted(self.kb.rules, key=lambda r: r.priority, reverse=True):
            if rule.evaluate(canary_map):
                fired.append(rule)
        if fired:
            top = fired[0]
            final_regime = top.conclusion_regime
            conf_map = {"Low": 0.4, "Medium": 0.7, "High": 0.9}
            confidence = conf_map.get(top.conclusion_confidence, 0.5)
            explanation = f"Rule '{top.name}' fired: {top.description}"
        else:
            from collections import Counter
            signals = [cr.signal for cr in canary_results]
            if not signals:
                final_regime = "RISK_ON_GROWTH"
                confidence = 0.5
                explanation = "No canaries, default Risk On"
            else:
                cnt = Counter(signals)
                most_common = cnt.most_common(1)[0][0]
                signal_to_regime = {
                    "RiskOn": "RISK_ON_GROWTH",
                    "RiskOff": "DEFENSIVE_ROTATION",
                    "BlackSwan": "BLACK_SWAN_VOL_SHOCK",
                    "Inflation": "INFLATION_GROWTH",
                    "SafeHaven": "DEFENSIVE_ROTATION",
                    "Neutral": "RANGE_BOUND_WHIPSAW"
                }
                final_regime = signal_to_regime.get(most_common, "RISK_ON_GROWTH")
                confidence = 0.5
                explanation = f"No rule fired, majority canary signal {most_common}"
        return {
            "final_regime": final_regime,
            "confidence": float(confidence),
            "fired_rules": [r.to_dict() for r in fired],
            "fired_count": len(fired),
            "explanation": explanation,
            "all_rules_evaluated": len(self.kb.rules)
        }
