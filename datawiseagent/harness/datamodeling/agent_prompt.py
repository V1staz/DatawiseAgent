"""Prompt and repair-message helpers for the DataModeling agent harness.

This module does not solve modeling tasks.  It turns label-blind contract/profile
facts into concise instructions that constrain and support the LLM-driven
DatawiseAgent execution path.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import json

from .contracts import DataModelingContract
from .data_availability import DataAvailabilityReport, inspect_data_availability
from .guidance import DataModelingGuidancePack, load_guidance_pack
from .profiler import DataProfile
from .protocol import ModelingProtocolPlan, build_protocol_plan
from .quality import QualityReport
from .repair_planner import RepairAction, plan_repair_actions
from .risk_scheduler import RiskPlan
from .stage_agents import StageAgentPlan, build_stage_agent_plan
from .skill_cards import get_skill_cards
from .submission import SubmissionValidationResult, submission_template
from .task_router import TaskRoute, route_task
from .verified_evidence import VerifiedModelingEvidence


def render_agent_prompt(
    contract: DataModelingContract,
    profile: DataProfile,
    description: str,
    *,
    route: TaskRoute | None = None,
    guidance: DataModelingGuidancePack | None = None,
    risk_plan: RiskPlan | None = None,
    availability: DataAvailabilityReport | None = None,
    protocol_plan: ModelingProtocolPlan | None = None,
    stage_agent_plan: StageAgentPlan | None = None,
    verified_evidence: VerifiedModelingEvidence | None = None,
    prior_memory_lessons: list[str] | None = None,
    stage_pipeline_result: Any | None = None,
) -> str:
    """Render the initial LLM-in-the-loop prompt for one DataModeling task."""

    availability = availability or inspect_data_availability(contract)
    route = route or route_task(contract, profile, availability=availability)
    guidance = guidance or load_guidance_pack()
    protocol_plan = protocol_plan or build_protocol_plan(contract, profile, route, availability=availability)
    stage_agent_plan = stage_agent_plan or build_stage_agent_plan(protocol_plan)
    skill_cards = get_skill_cards(route.skill_card_ids)
    prior_memory_lessons = prior_memory_lessons or []
    expected_rows = len(submission_template(contract))
    payload = {
        "task": contract.name,
        "metric": asdict(contract.metric),
        "task_kinds": contract.task_kinds,
        "submission": {
            "columns": contract.submission.columns,
            "id_columns": contract.id_columns,
            "target_columns": contract.submission.target_columns,
            "expected_rows": expected_rows,
        },
        "train_targets": contract.target_columns,
        "feature_columns_sample": contract.feature_columns[:60],
        "text_columns": contract.text_columns,
        "datetime_columns": contract.datetime_columns,
        "group_columns": getattr(contract, "group_columns", []),
        "risk_flags": contract.risk_flags,
        "recipe_hints": contract.recipe_hints,
        "profile_summary": _profile_summary(profile),
    }
    route_payload = route.to_dict()
    return f"""You are solving a Kaggle-style DataModeling task with code execution.

<task_description>
{description.strip()}
</task_description>

<datamodeling_harness_contract>
{json.dumps(payload, ensure_ascii=False, indent=2)}
</datamodeling_harness_contract>

<task_route>
{json.dumps(route_payload, ensure_ascii=False, indent=2)}
</task_route>

<layered_guidance_excerpt>
{guidance.render_excerpt(route.scopes)}
</layered_guidance_excerpt>

<selected_skill_cards>
{_render_skill_cards(skill_cards)}
</selected_skill_cards>

<risk_plan>
{risk_plan.prompt_summary() if risk_plan is not None else "risk_level=unknown; scheduler artifact unavailable"}
</risk_plan>

<modeling_protocol_plan>
{protocol_plan.prompt_summary()}
</modeling_protocol_plan>

<stage_subagent_plan>
{stage_agent_plan.prompt_summary()}
</stage_subagent_plan>

<executable_stage_subagent_outputs>
{stage_pipeline_result.prompt_summary() if hasattr(stage_pipeline_result, "prompt_summary") else (json.dumps(stage_pipeline_result, ensure_ascii=False, indent=2, default=str) if stage_pipeline_result else "status=not_run; executable stage sub-agent pipeline disabled or unavailable.")}
</executable_stage_subagent_outputs>

<verified_modeling_evidence>
{verified_evidence.prompt_summary() if verified_evidence is not None else "status=unavailable; harness did not produce a verified public-data baseline/CV proxy for this prompt."}
</verified_modeling_evidence>

<prior_harness_lessons>
{_render_memory_lessons(prior_memory_lessons)}
</prior_harness_lessons>

Hard harness constraints:
1. The public files are already uploaded under ./input/. Inspect train.csv, test.csv, and sample_submission.csv before modeling.
2. You, the LLM agent, must perform the modeling work. The harness validates and repairs only; it will not generate predictions for you.
3. Do not use hidden answers, GT performance, leaderboard files, or any path outside uploaded public task data.
4. Treat the local contract as authoritative: expected_rows={expected_rows}, exact columns={contract.submission.columns}, and final path ./input/final_submission.csv.
5. Keep fallback and final separated: write recovery fallback to ./input/fallback_submission.csv; write ./input/final_submission.csv only after a modeled or defensible improved candidate is selected.
6. Use <executable_stage_subagent_outputs> and <verified_modeling_evidence> as score-improvement scaffolds: follow concrete audit/split/feature/modeling recommendations from the independent stage sub-agents, inspect verified candidate leaderboard/fold metrics/recommendations when available, and improve from there rather than only satisfying manifest fields. Write ./input/submission_manifest.json next to the final file. Use stage="improved" only when public-data validation/proxy beats fallback. If <verified_modeling_evidence> status is ok, align validation_metric_value and baseline_metric_value with that harness-verified proxy unless you clearly add validation_source="agent_additional_validation" with your own public-data validation details. Include metric_direction, validation_metric_value, baseline_metric_value, trained_rows, selected_reason, prediction_summary, the protocol required_manifest_evidence above, and stage_agent_outputs for every selected stage sub-agent.
7. The quality gate rejects a bare valid CSV, sample-copy targets, fallback-as-final without recovery stage, missing manifest, protocol evidence gaps, and constant low-information finals without validation evidence.
8. Keep execution bounded and observable: small cells, row caps for heavy models, Avoid n_jobs=-1 on large data, and no broad grid search before a valid quality-backed final.
9. Before finishing, read back final_submission.csv and submission_manifest.json and print shape, columns, manifest stage, and prediction diversity.
10. Final response: briefly state modeling approach, validation check, manifest stage, and path ./input/final_submission.csv.
"""


def render_repair_prompt(
    contract: DataModelingContract,
    validation: SubmissionValidationResult,
    *,
    attempt: int,
    submission_path: str,
    clean_retry: bool = False,
    quality: QualityReport | None = None,
    route: TaskRoute | None = None,
    protocol_plan: ModelingProtocolPlan | None = None,
    stage_agent_plan: StageAgentPlan | None = None,
    repair_actions: list[RepairAction] | None = None,
    verified_evidence: VerifiedModelingEvidence | None = None,
    review_decision: Any | None = None,
    stage_pipeline_result: Any | None = None,
) -> str:
    """Render a bounded repair prompt from validator errors only."""

    repair_actions = repair_actions if repair_actions is not None else plan_repair_actions(contract, validation, quality=quality, route=route, protocol_plan=protocol_plan)
    payload = {
        "attempt": attempt,
        "task": contract.name,
        "current_submission_path": submission_path,
        "required_output_path": "./input/final_submission.csv",
        "expected_columns": contract.submission.columns,
        "id_columns": contract.id_columns,
        "target_columns": contract.submission.target_columns,
        "metric": asdict(contract.metric),
        "validator": validation.to_dict(),
        "quality_gate": quality.to_dict() if quality is not None else None,
        "task_route": route.to_dict() if route is not None else None,
        "modeling_protocol_plan": protocol_plan.to_dict() if protocol_plan is not None else None,
        "stage_subagent_plan": _compact_stage_plan(stage_agent_plan),
        "verified_modeling_evidence": _compact_verified_evidence(verified_evidence),
        "review_agent_decision": _compact_review_decision(review_decision),
        "executable_stage_subagent_outputs": _compact_stage_pipeline(stage_pipeline_result),
        "selected_skill_cards": [card.to_dict() for card in get_skill_cards(route.skill_card_ids)] if route is not None else None,
        "repair_actions": [action.to_dict() for action in repair_actions],
    }
    problem = "validator or quality gate found that your current DataModeling submission is not acceptable"
    context_note = (
        "This repair is running in a fresh session; the prior failed CSV may not exist in the current ./input/. Use the validator/quality payload as the source of truth and rerun the minimal necessary public-data modeling steps."
        if clean_retry
        else "The prior files should still be available in the current session; inspect the existing submission before making the smallest targeted repair."
    )
    return f"""The harness {problem}.

<validator_feedback>
{json.dumps(payload, ensure_ascii=False, indent=2)}
</validator_feedback>

<targeted_repair_actions>
{_render_repair_actions(repair_actions)}
</targeted_repair_actions>

<repair_context>
{context_note}
</repair_context>

Repair instructions:
1. Do not restart the task from scratch unless necessary. Inspect existing files when available; in fresh-session repair, rely on the validator/quality payload and rerun only the minimal needed public-data modeling.
2. If the final CSV is missing but you have a modeled candidate under ./input/, copy that candidate to ./input/final_submission.csv. Do not copy fallback_submission.csv as final unless every improved attempt failed and the manifest uses stage="fallback_recovery".
3. Fix only reported issues: columns, row count, ID/order, dtype, missing values, metric/range constraints, manifest fields, fallback misuse, constant low-information predictions, or review-agent blocker/major findings.
4. Do not use hidden answers or scoring outputs.
5. If the failed file copied the sample submission targets or contains invalid ranges, rerun/repair your own public-data modeling or fallback logic rather than copying sample targets.
6. Save the repaired final file exactly to ./input/final_submission.csv and write/update ./input/submission_manifest.json.
7. After saving, read back both files and print final shape, columns, manifest stage, and per-target prediction diversity so the harness can verify the repair.
"""


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
        "features_used": list(payload.get("features_used") or [])[:40],
        "validation_metric_trust": payload.get("validation_metric_trust"),
        "warnings": list(payload.get("warnings") or [])[:8],
        "summary": {
            "group_column": summary.get("group_column"),
            "group_count": summary.get("group_count"),
            "score_gap_vs_dummy": summary.get("score_gap_vs_dummy"),
        },
    }


def _compact_review_decision(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value.to_dict() if hasattr(value, "to_dict") else (dict(value) if isinstance(value, dict) else {})
    return {
        "passed": payload.get("passed"),
        "confidence": payload.get("confidence"),
        "summary": payload.get("summary"),
        "blocks_finalization": payload.get("blocks_finalization"),
        "findings": list(payload.get("findings") or [])[:5],
    }


def _compact_stage_pipeline(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    outputs = getattr(value, "outputs", None)
    if outputs is None and isinstance(value, dict):
        outputs = value.get("outputs")
    compact_outputs = []
    for item in list(outputs or []):
        payload = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        compact_outputs.append({
            "agent_id": payload.get("agent_id"),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "outputs": payload.get("outputs"),
            "blockers": payload.get("blockers"),
            "recommendations": payload.get("recommendations"),
            "handoff_notes": payload.get("handoff_notes"),
            "confidence": payload.get("confidence"),
        })
    return {
        "mode": getattr(value, "mode", None) if not isinstance(value, dict) else value.get("mode"),
        "blocked": getattr(value, "blocked", None) if not isinstance(value, dict) else value.get("blocked"),
        "outputs": compact_outputs,
    }


def _render_skill_cards(skill_cards: list[Any]) -> str:
    if not skill_cards:
        return "- No machine-readable skill cards selected."
    return "\n".join(card.prompt_summary() for card in skill_cards)


def _render_repair_actions(actions: list[RepairAction]) -> str:
    if not actions:
        return "- No targeted repair actions selected."
    return "\n".join(action.prompt_summary() for action in actions)


def _render_memory_lessons(lessons: list[str]) -> str:
    if not lessons:
        return "- No prior same-route lessons selected."
    return "\n".join(f"- {lesson}" for lesson in lessons[:3])


def _profile_summary(profile: DataProfile) -> dict[str, Any]:
    target_summary = profile.target_summary
    columns = []
    for col in profile.columns[:80]:
        columns.append(
            {
                "name": col.name,
                "role": col.role,
                "dtype": col.dtype,
                "missing_fraction": col.missing_fraction,
                "unique_count": col.unique_count,
                "sample_values": col.sample_values[:3],
            }
        )
    return {
        "train_rows_sampled": profile.train_rows_sampled,
        "test_rows_sampled": profile.test_rows_sampled,
        "train_columns": profile.train_columns,
        "test_columns": profile.test_columns,
        "columns_sample": columns,
        "target_summary": target_summary,
        "warnings": profile.warnings,
    }
