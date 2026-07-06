"""DataModeling LLM-in-the-loop harness public API.

The package-level exports intentionally prefer the official agent harness and
shared label-blind contract/profile/validator helpers.  Deterministic recipe
baselines live behind explicit imports from ``.runner`` / ``.recipes`` so they
are not mistaken for model-capability evaluation.
"""

from .contracts import DataModelingContract, MetricSpec, SubmissionSpec, build_contract, load_tasks
from .data_availability import DataAvailabilityReport, inspect_data_availability
from .profiler import DataProfile, build_profile
from .protocol import ModelingProtocolPlan, build_protocol_plan
from .stage_agents import StageAgent, StageAgentPlan, build_stage_agent_plan
from .agent_prompt import render_agent_prompt, render_repair_prompt
from .agent_runner import DataModelingAgentHarnessConfig, DataModelingAgentHarnessRunner, FunctionChatClient, summarize_agent_harness_results
from .quality import QualityReport, evaluate_submission_quality
from .submission import SubmissionValidationResult, submission_template, validate_submission
from .task_router import TaskRoute, route_task
from .verified_evidence import VerifiedModelingEvidence, build_verified_modeling_evidence

__all__ = [
    "DataModelingContract",
    "MetricSpec",
    "SubmissionSpec",
    "DataAvailabilityReport",
    "inspect_data_availability",
    "build_contract",
    "load_tasks",
    "DataProfile",
    "build_profile",
    "ModelingProtocolPlan",
    "build_protocol_plan",
    "StageAgent",
    "StageAgentPlan",
    "build_stage_agent_plan",
    "render_agent_prompt",
    "render_repair_prompt",
    "DataModelingAgentHarnessConfig",
    "DataModelingAgentHarnessRunner",
    "FunctionChatClient",
    "summarize_agent_harness_results",
    "QualityReport",
    "evaluate_submission_quality",
    "TaskRoute",
    "route_task",
    "VerifiedModelingEvidence",
    "build_verified_modeling_evidence",
    "SubmissionValidationResult",
    "submission_template",
    "validate_submission",
]
