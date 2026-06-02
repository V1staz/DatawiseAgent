"""Schemas for the DatawiseAgent evaluation harness.

The harness deliberately uses dataclasses instead of Pydantic so that artifact
creation and static validation work even in lightweight reproduction
environments where only the Python standard library is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal
import json

HarnessMode = Literal[
    "baseline",
    "datacard",
    "contract",
    "skills",
    "verify",
    "memory",
    "search",
    "full",
]

SearchPolicy = Literal["off", "hard-only", "all", "validator-fail"]


@dataclass(slots=True)
class TargetAnswer:
    name: str
    type: str = "unknown"
    rounding: int | None = None
    format: str | None = None
    description: str | None = None


@dataclass(slots=True)
class QuestionContract:
    question_id: int | str
    question: str
    level: str | None = None
    concepts: list[str] = field(default_factory=list)
    target_answers: list[TargetAnswer] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    groupby: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    forbidden_ops: list[str] = field(
        default_factory=lambda: ["use_label_file", "manual_ground_truth"]
    )
    final_format: str | None = None
    raw_constraints: str | None = None


@dataclass(slots=True)
class DataCardColumn:
    name: str
    dtype: str
    missing_count: int = 0
    missing_rate: float = 0.0
    unique_count: int | None = None
    sample_values: list[Any] = field(default_factory=list)
    numeric_min: float | None = None
    numeric_max: float | None = None
    date_like: bool = False
    role_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataCard:
    path: str
    row_count: int
    column_count: int
    columns: list[DataCardColumn] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    category_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TraceEvent:
    step_id: int
    stage: str
    skill: str | None = None
    reason_summary: str | None = None
    action: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinalAnswerBlock:
    question_id: int | str
    answers: dict[str, Any] = field(default_factory=dict)
    format_targets: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    validator_report_id: str | None = None


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0
    report_id: str | None = None


@dataclass(slots=True)
class FailureMemory:
    signature: str
    question_features: dict[str, Any] = field(default_factory=dict)
    failure_type: str = "unknown"
    failed_action: str | None = None
    repair: str | None = None
    validator_that_caught_it: str | None = None
    reuse_policy: str = "retrieve_by_concept_and_column_overlap"


@dataclass(slots=True)
class HarnessArtifacts:
    root_dir: str
    data_card_path: str | None = None
    contract_path: str | None = None
    trace_path: str | None = None
    validator_report_path: str | None = None
    final_block_path: str | None = None
    rule_manifest_path: str | None = None
    oracle_plan_path: str | None = None
    semantic_report_path: str | None = None


@dataclass(slots=True)
class HarnessContext:
    question: dict[str, Any]
    table_path: str
    mode: str
    prompt: str
    contract: QuestionContract | None = None
    data_card: DataCard | None = None
    skills: list[str] = field(default_factory=list)
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    oracle_checks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: HarnessArtifacts | None = None


def to_plain(obj: Any) -> Any:
    """Convert dataclasses/Paths recursively to JSON-serializable objects."""
    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_plain(obj), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_plain(obj), ensure_ascii=False, sort_keys=True) + "\n")
