"""Deterministic, label-blind risk scheduling for DataModeling tasks.

The first implementation only produces policy artifacts and concise prompt
guidance.  It intentionally does not change model selection or launch parallel
tree routes, so it is safe to validate with unit tests before metric reruns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import DataModelingContract
from .profiler import DataProfile
from .task_router import TaskRoute


@dataclass(frozen=True, slots=True)
class RiskPlan:
    risk_level: str
    reasons: tuple[str, ...]
    prompt_detail_level: str
    recommended_timeout_seconds: int
    max_repair_attempts_hint: int
    early_stop_min_seconds_hint: float
    tree_route_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        reasons = ", ".join(self.reasons) if self.reasons else "no elevated route risk detected"
        return (
            f"risk_level={self.risk_level}; reasons={reasons}; "
            f"prompt_detail_level={self.prompt_detail_level}; "
            f"timeout_hint={self.recommended_timeout_seconds}s; "
            f"repair_attempts_hint={self.max_repair_attempts_hint}; "
            f"tree_route_enabled={self.tree_route_enabled}"
        )


def build_risk_plan(contract: DataModelingContract, profile: DataProfile, route: TaskRoute) -> RiskPlan:
    reasons: list[str] = []

    if route.family in {"text_modeling", "multi_output", "time_series_or_grouped", "simulation_or_special"}:
        reasons.append(f"route_family:{route.family}")
    if route.metric_family in {"probability", "unknown"}:
        reasons.append(f"metric_family:{route.metric_family}")
    if len(contract.submission.target_columns) > 1:
        reasons.append(f"target_count:{len(contract.submission.target_columns)}")
    if contract.text_columns:
        reasons.append(f"text_columns:{min(len(contract.text_columns), 5)}")
    if contract.datetime_columns:
        reasons.append(f"datetime_columns:{min(len(contract.datetime_columns), 5)}")
    if contract.risk_flags:
        reasons.extend(f"contract:{flag}" for flag in contract.risk_flags[:3])
    if profile.warnings:
        reasons.extend(f"profile:{warning}" for warning in profile.warnings[:3])

    risk_score = len(reasons)
    if route.family in {"multi_output", "time_series_or_grouped"}:
        risk_score += 1
    if route.family == "simulation_or_special":
        risk_score += 2

    if risk_score >= 5:
        level = "hard"
        timeout = 3600
        repairs = 2
        early_stop = 90.0
        detail = "full"
        tree = True
    elif risk_score >= 2:
        level = "medium"
        timeout = 2400
        repairs = 2
        early_stop = 60.0
        detail = "standard"
        tree = False
    else:
        level = "easy"
        timeout = 1200
        repairs = 1
        early_stop = 30.0
        detail = "compact"
        tree = False

    return RiskPlan(
        risk_level=level,
        reasons=tuple(reasons[:8]),
        prompt_detail_level=detail,
        recommended_timeout_seconds=timeout,
        max_repair_attempts_hint=repairs,
        early_stop_min_seconds_hint=early_stop,
        tree_route_enabled=tree,
    )
