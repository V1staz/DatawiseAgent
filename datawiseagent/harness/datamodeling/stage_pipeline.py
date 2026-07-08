"""Executable stage-sub-agent pipeline helpers for DataModeling.

`stage_agents.py` defines the role graph.  This module turns those roles into
independent, auditable sub-agent calls: each stage gets its own prompt/session,
returns structured JSON, and hands its output to downstream stages and the main
modeling prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import re

from .contracts import DataModelingContract
from .data_availability import DataAvailabilityReport
from .profiler import DataProfile
from .protocol import ModelingProtocolPlan
from .risk_scheduler import RiskPlan
from .stage_agents import StageAgent, StageAgentPlan
from .task_router import TaskRoute
from .verified_evidence import VerifiedModelingEvidence


STAGE_OUTPUT_TAG = "stage_agent_output_json"
PRE_MODELING_EXECUTABLE_IDS = {"data_availability_auditor", "data_auditor", "split_guardian", "feature_builder"}
PRE_FINAL_AGENT_IDS = {"verifier_finalizer"}


@dataclass(slots=True)
class StageSubAgentOutput:
    agent_id: str
    role: str
    status: str = "unknown"
    summary: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    handoff_notes: list[str] = field(default_factory=list)
    confidence: str = "unknown"
    raw_response_excerpt: str = ""
    parse_error: str | None = None

    @property
    def blocked(self) -> bool:
        return bool(self.blockers) or self.status in {"blocked", "failed"}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocked"] = self.blocked
        return payload


@dataclass(slots=True)
class StagePipelineResult:
    mode: str
    outputs: list[StageSubAgentOutput] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return any(output.blocked for output in self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "outputs": [output.to_dict() for output in self.outputs],
            "artifact_paths": dict(self.artifact_paths),
            "skipped": self.skipped,
            "reason": self.reason,
            "blocked": self.blocked,
        }

    def prompt_summary(self) -> str:
        payload = {
            "mode": self.mode,
            "blocked": self.blocked,
            "outputs": [
                {
                    "agent_id": output.agent_id,
                    "status": output.status,
                    "summary": output.summary,
                    "outputs": output.outputs,
                    "blockers": output.blockers,
                    "recommendations": output.recommendations,
                    "handoff_notes": output.handoff_notes,
                    "confidence": output.confidence,
                }
                for output in self.outputs
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def executable_pre_modeling_agents(stage_agent_plan: StageAgentPlan) -> list[StageAgent]:
    """Return minimal audit/planning agents that should run before modeling.

    Training/modeling roles remain in the main modeling session so the harness
    does not generate predictions through sub-agents.  Task-specific agents stay
    in the stage plan/manifest contract and are invoked through the main agent
    unless promoted deliberately later.
    """

    return [agent for agent in stage_agent_plan.agents if agent.id in PRE_MODELING_EXECUTABLE_IDS]


def relevant_prior_outputs(agent: StageAgent, stage_agent_plan: StageAgentPlan, prior_outputs: Iterable[StageSubAgentOutput]) -> list[StageSubAgentOutput]:
    """Keep only direct predecessor outputs according to handoff edges."""

    upstream_by_id = {candidate.id: candidate for candidate in stage_agent_plan.agents if agent.id in candidate.handoff_to}
    return [output for output in prior_outputs if output.agent_id in upstream_by_id]


def find_stage_agent(stage_agent_plan: StageAgentPlan, agent_id: str) -> StageAgent | None:
    for agent in stage_agent_plan.agents:
        if agent.id == agent_id:
            return agent
    return None


def compact_stage_outputs(outputs: Iterable[StageSubAgentOutput]) -> list[dict[str, Any]]:
    """Return downstream-safe stage outputs without raw responses or artifact paths."""

    compact: list[dict[str, Any]] = []
    for output in outputs:
        compact.append(
            {
                "agent_id": output.agent_id,
                "status": output.status,
                "summary": output.summary,
                "outputs": output.outputs,
                "blockers": output.blockers,
                "recommendations": output.recommendations,
                "handoff_notes": output.handoff_notes,
                "confidence": output.confidence,
            }
        )
    return compact


def _verified_evidence_compact(value: VerifiedModelingEvidence | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {
        "status": value.status,
        "metric_name": value.metric_name,
        "metric_direction": value.metric_direction,
        "baseline_metric_value": value.baseline_metric_value,
        "best_candidate_metric_value": value.best_candidate_metric_value,
        "best_candidate_name": value.best_candidate_name,
        "split_policy": value.split_policy,
        "features_used": value.features_used[:40],
        "validation_metric_trust": value.validation_metric_trust,
        "warnings": value.warnings[:8],
        "summary": {
            "group_column": value.summary.get("group_column"),
            "group_count": value.summary.get("group_count"),
            "score_gap_vs_dummy": value.summary.get("score_gap_vs_dummy"),
            "recommendations": list(value.summary.get("recommendations") or [])[:5],
        },
    }


def render_stage_subagent_prompt(
    agent: StageAgent,
    contract: DataModelingContract,
    profile: DataProfile,
    route: TaskRoute,
    protocol_plan: ModelingProtocolPlan,
    stage_agent_plan: StageAgentPlan,
    *,
    availability: DataAvailabilityReport | None = None,
    risk_plan: RiskPlan | None = None,
    verified_evidence: VerifiedModelingEvidence | None = None,
    prior_outputs: Iterable[StageSubAgentOutput] | None = None,
    finalization_context: dict[str, Any] | None = None,
) -> str:
    """Render one executable stage-sub-agent prompt."""

    prior = compact_stage_outputs(prior_outputs or [])
    context = {
        "task": contract.name,
        "agent": agent.to_dict(),
        "contract": {
            "metric": asdict(contract.metric),
            "task_kinds": contract.task_kinds,
            "submission_columns": contract.submission.columns,
            "id_columns": contract.id_columns,
            "target_columns": contract.submission.target_columns,
            "feature_columns_sample": contract.feature_columns[:40],
            "text_columns": contract.text_columns,
            "datetime_columns": contract.datetime_columns,
            "group_columns": getattr(contract, "group_columns", []),
            "risk_flags": contract.risk_flags,
            "recipe_hints": contract.recipe_hints,
        },
        "profile_summary": {
            "train_rows_sampled": profile.train_rows_sampled,
            "test_rows_sampled": profile.test_rows_sampled,
            "target_summary": profile.target_summary,
            "warnings": profile.warnings,
        },
        "route": route.to_dict(),
        "protocol_plan": protocol_plan.to_dict(),
        "stage_pipeline_contract": {
            "required_manifest_key": stage_agent_plan.required_manifest_key,
            "stage_ids": stage_agent_plan.required_agent_ids,
        },
        "availability": availability.to_dict() if hasattr(availability, "to_dict") else availability,
        "risk_plan": risk_plan.to_dict() if hasattr(risk_plan, "to_dict") else risk_plan,
        "verified_modeling_evidence": _verified_evidence_compact(verified_evidence),
        "prior_stage_outputs": prior,
        "finalization_context": finalization_context or {},
    }
    mode_note = (
        "This is a pre-modeling planning/audit sub-agent. Do not write final_submission.csv."
        if agent.id != "verifier_finalizer"
        else "This is a post-modeling verifier/finalizer sub-agent. Read final_submission.csv and submission_manifest.json when present; do not train new models."
    )
    return f"""You are an executable DataModeling stage sub-agent: `{agent.id}` ({agent.role}).

{mode_note}
You run as an independent sub-agent call. Inspect only public files under ./input/ and the provided context. Do not use hidden answers, GT files, official scores, or leaderboard data. Keep work bounded and evidence-based.

Responsibilities:
- Purpose: {agent.purpose}
- Owns stages: {', '.join(agent.owns_stages)}
- Expected outputs: {', '.join(agent.outputs)}
- Guardrails: {'; '.join(agent.guardrails)}
- Handoff to: {', '.join(agent.handoff_to) if agent.handoff_to else 'none'}

<stage_context>
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
</stage_context>

Return exactly one JSON object inside <{STAGE_OUTPUT_TAG}> tags. Keep it minimal: include only fields this stage owns, summarize evidence instead of dumping tables, and do not include file paths/session IDs/raw logs unless they are the blocker evidence. Schema:
{{
  "agent_id": "{agent.id}",
  "status": "ok|blocked|failed",
  "confidence": "low|medium|high",
  "summary": "one concise sentence",
  "outputs": {{"field_name": "concrete evidence or decision"}},
  "blockers": ["hard blocker if any"],
  "recommendations": ["specific downstream action"],
  "handoff_notes": ["what the next stage must use or avoid"]
}}
</{STAGE_OUTPUT_TAG}>
"""


def parse_stage_subagent_response(agent: StageAgent, response: dict[str, Any] | str) -> StageSubAgentOutput:
    text = response if isinstance(response, str) else str(response.get("response_content") or response.get("content") or "")
    excerpt = text[:4000]
    try:
        payload = _extract_json_payload(text)
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        return StageSubAgentOutput(
            agent_id=str(payload.get("agent_id") or agent.id),
            role=agent.role,
            status=str(payload.get("status") or "unknown").lower(),
            summary=str(payload.get("summary") or ""),
            outputs=dict(outputs),
            blockers=[str(item) for item in payload.get("blockers") or []],
            recommendations=[str(item) for item in payload.get("recommendations") or []],
            handoff_notes=[str(item) for item in payload.get("handoff_notes") or []],
            confidence=str(payload.get("confidence") or "unknown"),
            raw_response_excerpt="",
        )
    except Exception as exc:
        return StageSubAgentOutput(
            agent_id=agent.id,
            role=agent.role,
            status="parse_failed",
            summary="stage sub-agent response could not be parsed; downstream prompt will fall back to static plan",
            raw_response_excerpt=excerpt,
            parse_error=f"{type(exc).__name__}:{str(exc)[:200]}",
        )


def write_stage_pipeline_result(result: StagePipelineResult, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(target)


def _extract_json_payload(text: str) -> dict[str, Any]:
    tag_match = re.search(rf"<{STAGE_OUTPUT_TAG}>\s*(.*?)\s*</{STAGE_OUTPUT_TAG}>", text, flags=re.S)
    candidate = tag_match.group(1) if tag_match else _first_json_object(text)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("stage_agent_json_not_object")
    return payload


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("stage_agent_json_missing")
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
    raise ValueError("stage_agent_json_unclosed")
