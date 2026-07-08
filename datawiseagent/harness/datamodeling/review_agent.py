"""Read-only review-agent prompt and parsing for DataModeling attempts.

The deterministic validator catches schema and metric-domain violations, while
quality.py catches observable low-information and manifest issues.  This module
adds a deliberately separate reviewer role using the same live model/session: it
asks the model to inspect common Kaggle/DataModeling failure modes and return a
small structured decision that can feed the next repair prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import re

from .contracts import DataModelingContract
from .quality import QualityReport
from .submission import SubmissionValidationResult
from .task_router import TaskRoute
from .verified_evidence import VerifiedModelingEvidence


REVIEW_JSON_TAG = "review_decision_json"
BLOCKING_SEVERITIES = {"blocker", "major"}


@dataclass(slots=True)
class ReviewFinding:
    severity: str
    category: str
    evidence: str
    repair_instruction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewDecision:
    passed: bool
    confidence: str = "unknown"
    summary: str = ""
    findings: list[ReviewFinding] = field(default_factory=list)
    checklist: dict[str, Any] = field(default_factory=dict)
    raw_response_excerpt: str = ""
    parse_error: str | None = None

    @property
    def blocks_finalization(self) -> bool:
        if self.parse_error:
            return False
        return any(finding.severity.lower() in BLOCKING_SEVERITIES for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks_finalization"] = self.blocks_finalization
        return payload


PASSING_REVIEW = ReviewDecision(passed=True, confidence="not_run", summary="review agent disabled or unavailable")


def render_review_prompt(
    contract: DataModelingContract,
    validation: SubmissionValidationResult,
    quality: QualityReport,
    *,
    attempt: int,
    submission_path: str | Path,
    route: TaskRoute,
    protocol_plan: Any | None = None,
    stage_agent_plan: Any | None = None,
    verified_evidence: VerifiedModelingEvidence | None = None,
) -> str:
    """Render a read-only reviewer prompt for one completed attempt."""

    payload = {
        "attempt": attempt,
        "task": contract.name,
        "submission_path": str(submission_path),
        "expected_columns": contract.submission.columns,
        "id_columns": contract.id_columns,
        "target_columns": contract.submission.target_columns,
        "metric": asdict(contract.metric),
        "task_kinds": contract.task_kinds,
        "feature_columns_sample": contract.feature_columns[:50],
        "text_columns": contract.text_columns,
        "datetime_columns": contract.datetime_columns,
        "group_columns": getattr(contract, "group_columns", []),
        "route": route.to_dict(),
        "validator": validation.to_dict(),
        "quality_gate": quality.to_dict(),
        "protocol_plan": protocol_plan.to_dict() if hasattr(protocol_plan, "to_dict") else protocol_plan,
        "stage_agent_plan": _compact_stage_plan(stage_agent_plan),
        "verified_modeling_evidence": _compact_verified_evidence(verified_evidence),
    }
    return f"""You are the DataModeling REVIEW-AGENT for attempt {attempt}.

This is a read-only review pass. Inspect only public uploaded files under ./input/ and the attempt outputs. Do not modify files, do not train new models, and do not use hidden answers/GT/leaderboard data.

Your job is to catch common high-impact mistakes before finalization or before the next repair attempt. Focus on issues that small models often miss:
- final_submission.csv path, exact columns, row count, ID order, target missingness/range/dtype;
- sample_submission target copy, fallback copied as final, or predictions that are constant/near-constant;
- probability metrics receiving hard labels rather than continuous ranking scores;
- manifest missing, falsely marked improved, metric direction mismatch, baseline/validation evidence mismatch;
- task-family mistakes: random split for time/group data, ignored text columns on text tasks, identical columns on multi-output tasks, label encoding not mapped back to submission labels;
- signs that the code crashed after writing only a placeholder or stale file.

Use the validator and quality payload as evidence, but also read back ./input/final_submission.csv and ./input/submission_manifest.json when they exist to verify observable claims.

<attempt_context>
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
</attempt_context>

Return exactly one JSON object inside <{REVIEW_JSON_TAG}> tags with this schema:
{{
  "passed": true/false,
  "confidence": "low|medium|high",
  "summary": "one sentence",
  "checklist": {{
    "schema_ok": true/false/null,
    "not_sample_copy": true/false/null,
    "prediction_informative": true/false/null,
    "manifest_evidence_ok": true/false/null,
    "task_specific_risks_ok": true/false/null
  }},
  "findings": [
    {{
      "severity": "blocker|major|minor",
      "category": "schema|submission_quality|manifest|task_specific|execution",
      "evidence": "concrete observable evidence",
      "repair_instruction": "specific next action"
    }}
  ]
}}
</{REVIEW_JSON_TAG}>
"""


def parse_review_response(response: dict[str, Any] | str) -> ReviewDecision:
    """Parse a review-agent response into a non-throwing ReviewDecision."""

    text = response if isinstance(response, str) else str(response.get("response_content") or response.get("content") or "")
    excerpt = text[:4000]
    try:
        payload = _extract_json_payload(text)
        findings = [
            ReviewFinding(
                severity=str(item.get("severity") or "minor").lower(),
                category=str(item.get("category") or "unknown"),
                evidence=str(item.get("evidence") or ""),
                repair_instruction=str(item.get("repair_instruction") or ""),
            )
            for item in list(payload.get("findings") or [])
            if isinstance(item, dict)
        ]
        passed = bool(payload.get("passed")) and not any(f.severity in BLOCKING_SEVERITIES for f in findings)
        return ReviewDecision(
            passed=passed,
            confidence=str(payload.get("confidence") or "unknown"),
            summary=str(payload.get("summary") or ""),
            findings=findings,
            checklist=dict(payload.get("checklist") or {}),
            raw_response_excerpt="",
        )
    except Exception as exc:
        return ReviewDecision(
            passed=False,
            confidence="parse_failed_inconclusive",
            summary="review response could not be parsed; finalization relies on deterministic validator/quality gates",
            raw_response_excerpt=excerpt,
            parse_error=f"{type(exc).__name__}:{str(exc)[:200]}",
        )


def _compact_stage_plan(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "required_manifest_key": getattr(value, "required_manifest_key", "stage_agent_outputs"),
        "required_agent_ids": list(getattr(value, "required_agent_ids", []) or []),
    }


def _compact_verified_evidence(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value.to_dict() if hasattr(value, "to_dict") else (dict(value) if isinstance(value, dict) else {})
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "status": payload.get("status"),
        "metric_name": payload.get("metric_name"),
        "metric_direction": payload.get("metric_direction"),
        "baseline_metric_value": payload.get("baseline_metric_value"),
        "best_candidate_metric_value": payload.get("best_candidate_metric_value"),
        "best_candidate_name": payload.get("best_candidate_name"),
        "split_policy": payload.get("split_policy"),
        "validation_metric_trust": payload.get("validation_metric_trust"),
        "warnings": list(payload.get("warnings") or [])[:8],
        "summary": {
            "group_column": summary.get("group_column"),
            "group_count": summary.get("group_count"),
            "score_gap_vs_dummy": summary.get("score_gap_vs_dummy"),
        },
    }


def _extract_json_payload(text: str) -> dict[str, Any]:
    tag_match = re.search(rf"<{REVIEW_JSON_TAG}>\s*(.*?)\s*</{REVIEW_JSON_TAG}>", text, flags=re.S)
    candidate = tag_match.group(1) if tag_match else _first_json_object(text)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("review_json_not_object")
    return payload


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("review_json_missing")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("review_json_unclosed")
