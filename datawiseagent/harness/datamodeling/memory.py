"""Label-blind outcome memory for DataModeling harness runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from .task_router import TaskRoute


@dataclass(frozen=True, slots=True)
class OutcomeMemoryCard:
    task_id: int | None
    task: str
    route_family: str
    metric_family: str
    skill_card_ids: tuple[str, ...]
    failure: bool
    failure_category: str
    validation_passed: bool
    quality_passed: bool
    repair_action_ids: tuple[str, ...]
    lesson: str
    official_score_status: str | None = None
    official_score: float | None = None
    top_model_family: str | None = None
    feature_strategy: str | None = None
    validation_metric_trust: str | None = None
    performance_tactic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieve_memory_lessons(output_dir: str | Path, route: TaskRoute, *, limit: int = 2) -> list[str]:
    """Return bounded prior lessons for the same route/metric family."""

    store = _store_path(output_dir)
    if not store.exists():
        return []
    lessons: list[str] = []
    with store.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("route_family") != route.family and item.get("metric_family") != route.metric_family:
                continue
            for key in ("performance_tactic", "lesson"):
                lesson = str(item.get(key) or "").strip()
                if lesson and lesson not in lessons:
                    lessons.append(lesson)
                if len(lessons) >= limit:
                    break
            if len(lessons) >= limit:
                break
    return lessons


def build_outcome_memory_card(
    *,
    task_id: int | None,
    task: str,
    route: TaskRoute,
    failure: bool,
    failure_category: str,
    validation_reports: list[dict[str, Any]],
    quality_reports: list[dict[str, Any]],
    events: list[dict[str, Any]],
    official_score: dict[str, Any] | None,
) -> OutcomeMemoryCard:
    validation_passed = bool(validation_reports and validation_reports[-1].get("passed"))
    quality_passed = bool(quality_reports and quality_reports[-1].get("passed"))
    repair_action_ids = tuple(
        action_id
        for event in events
        if event.get("stage") == "repair_planned"
        for action_id in event.get("repair_action_ids", [])
    )
    evidence = _latest_modeling_evidence(quality_reports)
    official_score_value = _maybe_float((official_score or {}).get("score")) if official_score else None
    top_model_family = _top_model_family(evidence)
    feature_strategy = _feature_strategy(evidence)
    validation_metric_trust = _latest_validation_metric_trust(quality_reports)
    performance_tactic = _performance_tactic_for_outcome(
        route=route,
        failure=failure,
        quality_passed=quality_passed,
        evidence=evidence,
        official_score_value=official_score_value,
    )
    return OutcomeMemoryCard(
        task_id=task_id,
        task=task,
        route_family=route.family,
        metric_family=route.metric_family,
        skill_card_ids=tuple(route.skill_card_ids),
        failure=failure,
        failure_category=failure_category,
        validation_passed=validation_passed,
        quality_passed=quality_passed,
        repair_action_ids=repair_action_ids,
        lesson=_lesson_for_outcome(route, failure, failure_category, repair_action_ids),
        official_score_status=(official_score or {}).get("status") if official_score else None,
        official_score=official_score_value,
        top_model_family=top_model_family,
        feature_strategy=feature_strategy,
        validation_metric_trust=validation_metric_trust,
        performance_tactic=performance_tactic,
    )


def write_outcome_memory_card(
    artifact_dir: str | Path,
    output_dir: str | Path,
    card: OutcomeMemoryCard,
    *,
    append_store: bool = True,
) -> tuple[str, str]:
    artifact_path = Path(artifact_dir) / "outcome_memory_card.json"
    payload = card.to_dict()
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    store = _store_path(output_dir)
    store.parent.mkdir(parents=True, exist_ok=True)
    if append_store:
        with store.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(artifact_path), str(store)


def _store_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "memory" / "datamodeling_outcomes.jsonl"


def _lesson_for_outcome(route: TaskRoute, failure: bool, failure_category: str, repair_action_ids: tuple[str, ...]) -> str:
    prefix = f"{route.family}/{route.metric_family}"
    if not failure:
        return f"{prefix}: preserve scoreable candidates, exact schema, scaffold-backed validation evidence, and prediction diversity checks."
    if repair_action_ids:
        return f"{prefix}: prior failure {failure_category}; prioritize repairs {', '.join(repair_action_ids[:3])} before finalization."
    return f"{prefix}: prior failure {failure_category}; validate structure, manifest evidence, and route-specific quality before finalization."


def _latest_modeling_evidence(quality_reports: list[dict[str, Any]]) -> dict[str, Any]:
    for report in reversed(quality_reports):
        evidence = ((report.get("verified_evidence") or {}).get("modeling_evidence") or {})
        if isinstance(evidence, dict) and evidence:
            return evidence
    return {}


def _latest_validation_metric_trust(quality_reports: list[dict[str, Any]]) -> str | None:
    for report in reversed(quality_reports):
        trust = (report.get("summary") or {}).get("validation_metric_trust")
        if trust:
            return str(trust)
    return None


def _top_model_family(evidence: dict[str, Any]) -> str | None:
    name = str(evidence.get("best_candidate_name") or "")
    if not name:
        top = ((evidence.get("summary") or {}).get("top_candidate") or {})
        name = str(top.get("name") or "")
    if not name:
        return None
    return name.split(":", 1)[0]


def _feature_strategy(evidence: dict[str, Any]) -> str | None:
    features = list(evidence.get("features_used") or [])
    summary = evidence.get("summary") or {}
    if not features and isinstance(summary, dict):
        top = summary.get("top_candidate") or {}
        if isinstance(top, dict):
            features = list(top.get("features_used") or [])
    if not features:
        return None
    return ",".join(str(item) for item in features[:8])


def _performance_tactic_for_outcome(
    *,
    route: TaskRoute,
    failure: bool,
    quality_passed: bool,
    evidence: dict[str, Any],
    official_score_value: float | None,
) -> str | None:
    model = _top_model_family(evidence)
    strategy = _feature_strategy(evidence)
    trust = str(evidence.get("validation_metric_trust") or "")
    if failure and not evidence:
        return None
    score_part = f", official_score={official_score_value}" if official_score_value is not None else ""
    quality_part = "quality_passed" if quality_passed else "scoreable_low_confidence"
    model_part = f"model={model}" if model else "model=scaffold_top_candidate"
    feature_part = f", features={strategy}" if strategy else ""
    trust_part = f", trust={trust}" if trust else ""
    return f"{route.family}/{route.metric_family}: prior {quality_part} tactic {model_part}{feature_part}{trust_part}{score_part}; start from similar scaffold evidence before broad tuning."


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
