"""Stage-level role planning for DataModeling harness prompts.

The stage plan is both a prompt contract and, when enabled by the live runner,
an executable sub-agent graph.  Each stage has one accountable role persona,
concrete inputs, outputs, guardrails, and handoffs.  Runtime stage sub-agents
produce structured artifacts that the main modeling agent, verifier, and repair
loop can consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .protocol import ModelingProtocolPlan


@dataclass(frozen=True, slots=True)
class StageAgent:
    id: str
    role: str
    purpose: str
    owns_stages: list[str]
    inputs: list[str]
    outputs: list[str]
    guardrails: list[str]
    handoff_to: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        outputs = "; ".join(self.outputs[:3])
        guardrails = "; ".join(self.guardrails[:2])
        return f"- `{self.id}` ({self.role}): {self.purpose}. Outputs: {outputs}. Guardrails: {guardrails}."


@dataclass(frozen=True, slots=True)
class StageAgentPlan:
    orchestration_mode: str
    agents: list[StageAgent]
    required_manifest_key: str = "stage_agent_outputs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "orchestration_mode": self.orchestration_mode,
            "required_manifest_key": self.required_manifest_key,
            "agents": [agent.to_dict() for agent in self.agents],
        }

    def prompt_summary(self) -> str:
        lines = [
            f"orchestration_mode={self.orchestration_mode}",
            "runtime_semantics=executable harness sub-agents when enabled; otherwise single-session internal stage roles",
            f"manifest_must_include={self.required_manifest_key}",
        ]
        lines.extend(agent.prompt_summary() for agent in self.agents)
        return "\n".join(lines)

    @property
    def required_agent_ids(self) -> list[str]:
        return [agent.id for agent in self.agents]


def build_stage_agent_plan(protocol_plan: ModelingProtocolPlan) -> StageAgentPlan:
    """Build a compact internal stage-role plan for the selected protocol."""

    if protocol_plan.target_type == "data_unavailable_or_pointer":
        return StageAgentPlan(
            orchestration_mode="blocked_public_data_triage",
            agents=[
                StageAgent(
                    id="data_availability_auditor",
                    role="public-data auditor",
                    purpose="Confirm whether public CSV files are usable before any modeling.",
                    owns_stages=["validate_public_data_availability", "inspect_schema_and_target_contract"],
                    inputs=["data_availability.json", "contract.json", "uploaded ./input files"],
                    outputs=["blocked_files", "restore_required_before_modeling"],
                    guardrails=["do not model on Git LFS pointer bytes", "do not treat pointer text as a text feature"],
                    handoff_to=[],
                )
            ],
        )

    agents = [
        StageAgent(
            id="data_auditor",
            role="schema and data-quality auditor",
            purpose="Read public train/test/sample schemas, target contract, missingness, dtypes, and obvious leakage risks.",
            owns_stages=["validate_public_data_availability", "inspect_schema_and_target_contract", "clean_missing_and_malformed_values"],
            inputs=["contract.json", "profile.json", "data_availability.json", "./input/train.csv", "./input/test.csv", "./input/sample_submission.csv"],
            outputs=["schema_summary", "cleaning_plan", "leakage_risk_notes"],
            guardrails=["label-blind public files only", "preserve sample submission schema exactly"],
            handoff_to=["split_guardian", "feature_builder"],
        ),
        StageAgent(
            id="split_guardian",
            role="validation and leakage guardian",
            purpose="Choose split policy from target/data structure and prevent random, temporal, group, or target-encoding leakage.",
            owns_stages=["eda_hypotheses_from_public_train_only", "choose_leakage_aware_split_policy"],
            inputs=["schema_summary", "protocol split_constraint", "target distribution summary"],
            outputs=["split_policy", "validation_strategy", "leakage_controls"],
            guardrails=["never validate on future/group-leaked rows", "stratify classification when feasible"],
            handoff_to=["feature_builder", "modeling_specialist"],
        ),
        StageAgent(
            id="feature_builder",
            role="feature engineering and encoding specialist",
            purpose="Build bounded preprocessing that matches the data structure and avoids high-cardinality or text/time leakage traps.",
            owns_stages=["build_feature_encoding_pipeline"],
            inputs=["cleaning_plan", "split_policy", "data_structure"],
            outputs=["features_used", "encoding_strategy", "feature_risk_notes"],
            guardrails=["fit encoders on train folds only", "avoid raw object columns in numeric estimators"],
            handoff_to=["modeling_specialist"],
        ),
        StageAgent(
            id="modeling_specialist",
            role="baseline/model/tuning specialist",
            purpose="Train a defensive fallback and at least one bounded candidate aligned with the metric and compute budget.",
            owns_stages=["train_bounded_baseline_and_candidate_models", "compare_against_defensive_fallback"],
            inputs=["features_used", "validation_strategy", "business_objective"],
            outputs=["baseline_metric_value", "validation_metric_value", "modeling_method", "selected_reason"],
            guardrails=["compare to fallback before claiming improved", "avoid broad grid search before a valid final exists"],
            handoff_to=["verifier_finalizer"],
        ),
        StageAgent(
            id="verifier_finalizer",
            role="submission verifier and manifest finalizer",
            purpose="Write/read back final_submission.csv and complete the evidence-rich manifest from all upstream stage outputs.",
            owns_stages=["finalize_submission_and_manifest"],
            inputs=["selected candidate predictions", "sample submission schema", "all prior stage outputs"],
            outputs=["final_submission.csv", "submission_manifest.json", "prediction_summary", "stage_agent_outputs"],
            guardrails=["final path must be ./input/final_submission.csv", "manifest stage improved only when validation beats baseline"],
            handoff_to=[],
        ),
    ]
    agents.extend(_task_specific_agents(protocol_plan))
    return StageAgentPlan(orchestration_mode="executable_stage_subagent_pipeline", agents=agents)


def _task_specific_agents(protocol_plan: ModelingProtocolPlan) -> list[StageAgent]:
    target_type = protocol_plan.target_type
    agents: list[StageAgent] = []
    if target_type in {"binary_probability", "multilabel_probability"}:
        agents.append(
            StageAgent(
                id="probability_calibrator",
                role="probability and ranking-quality specialist",
                purpose="Ensure probability metrics receive continuous, range-valid, calibrated or ranking-useful scores.",
                owns_stages=["emit_probabilities_not_hard_labels", "calibrate_or_rank_probability_scores"],
                inputs=["candidate model scores", "validation folds", "metric direction"],
                outputs=["probability_range", "calibration_or_ranking_check"],
                guardrails=["do not output hard labels for AUC/log-loss-like metrics", "keep probabilities within metric range"],
                handoff_to=["verifier_finalizer"],
            )
        )
    if target_type.startswith("time_series"):
        agents.append(
            StageAgent(
                id="time_leakage_guardian",
                role="time/group leakage specialist",
                purpose="Validate ordered/group split and lag/rolling features without using future information.",
                owns_stages=["sort_by_time_before_validation", "create_lag_rolling_calendar_features_without_future_leakage"],
                inputs=["datetime/group columns", "feature plan", "validation_strategy"],
                outputs=["time_or_group_columns", "future_leakage_checks"],
                guardrails=["no future rows in lag/rolling aggregates", "no random split when time ordering is required"],
                handoff_to=["verifier_finalizer"],
            )
        )
    if target_type == "text_modeling":
        agents.append(
            StageAgent(
                id="text_signal_specialist",
                role="text feature/modeling specialist",
                purpose="Extract lightweight text signal and document text preprocessing for text-heavy tasks.",
                owns_stages=["normalize_text_without_target_leakage", "select_lightweight_text_vectorizer_or_embedding"],
                inputs=["text columns", "metric prediction kind", "tabular side features"],
                outputs=["text_columns_used", "text_preprocessing"],
                guardrails=["avoid target leakage through text-derived labels", "handle empty/long text robustly"],
                handoff_to=["feature_builder", "verifier_finalizer"],
            )
        )
    if "multi_output" in target_type:
        agents.append(
            StageAgent(
                id="multi_output_coordinator",
                role="multi-output coordinator",
                purpose="Coordinate per-target modeling, diversity checks, and target-specific evidence.",
                owns_stages=["validate_each_target_column_separately", "avoid_constant_columns_in_multi_output_submission"],
                inputs=["target columns", "per-target validation results", "prediction summary"],
                outputs=["per_target_prediction_summary", "multi_output_strategy"],
                guardrails=["do not copy one vector across all targets", "explain any defensible constant target separately"],
                handoff_to=["verifier_finalizer"],
            )
        )
    if target_type == "ranking_or_topk_multiclass":
        agents.append(
            StageAgent(
                id="ranking_formatter",
                role="top-k/ranking output specialist",
                purpose="Ensure ranked class outputs preserve top-k format and class labels.",
                owns_stages=["construct_top_k_label_output_format", "avoid_collapsing_ranked_choices_to_single_label"],
                inputs=["class probabilities", "sample output format", "label mapping"],
                outputs=["ranking_output_format", "class_label_mapping"],
                guardrails=["do not collapse top-k output to one class", "map encoded labels back to original labels"],
                handoff_to=["verifier_finalizer"],
            )
        )
    return agents
