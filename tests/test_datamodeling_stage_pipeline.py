from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from datawiseagent.harness.datamodeling.agent_runner import DataModelingAgentHarnessConfig, DataModelingAgentHarnessRunner
from datawiseagent.harness.datamodeling.contracts import DataModelingContract, MetricSpec, SubmissionSpec
from datawiseagent.harness.datamodeling.protocol import ModelingProtocolPlan
from datawiseagent.harness.datamodeling.stage_agents import StageAgent, StageAgentPlan
from datawiseagent.harness.datamodeling.stage_pipeline import (
    executable_pre_modeling_agents,
    parse_stage_subagent_response,
    render_stage_subagent_prompt,
)
from datawiseagent.harness.datamodeling.task_router import route_task


def _contract(tmp_path: Path) -> DataModelingContract:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": [1, 2], "target": [0.5, 0.5]}).to_csv(data / "sample_submission.csv", index=False)
    pd.DataFrame({"id": [1, 2], "x": [10, 20]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": [3, 4], "x": [1, 2], "target": [0, 1]}).to_csv(data / "train.csv", index=False)
    return DataModelingContract(
        task_id=0,
        name="toy",
        benchmark_root=str(tmp_path),
        task_description_path=str(tmp_path / "task.txt"),
        data_dir=str(data),
        train_path=str(data / "train.csv"),
        test_path=str(data / "test.csv"),
        sample_submission_path=str(data / "sample_submission.csv"),
        eval_script_path=None,
        metric=MetricSpec("roc_auc", "maximize", prediction_kind="probability", probability_output=True, bounded_range=(0.0, 1.0)),
        submission=SubmissionSpec(str(data / "sample_submission.csv"), ["id", "target"], ["id"], ["target"], 2, {"target": "float64"}),
        task_kinds=["tabular_classification"],
        target_columns=["target"],
        id_columns=["id"],
        feature_columns=["x"],
        text_columns=[],
        datetime_columns=[],
        group_columns=[],
        categorical_columns=[],
        numeric_columns=["x"],
        risk_flags=[],
        recipe_hints=[],
        description_excerpt="toy",
    )


def _protocol() -> ModelingProtocolPlan:
    return ModelingProtocolPlan(
        target_type="binary_probability",
        data_structure=["tabular", "numeric"],
        split_constraint="stratified",
        business_objective="probability_ranking_quality_with_calibration_awareness:roc_auc",
        common_stages=[],
        task_specific_stages=[],
        required_manifest_evidence=["split_policy", "features_used"],
    )


def _agent(agent_id: str = "data_auditor") -> StageAgent:
    return StageAgent(
        id=agent_id,
        role="test role",
        purpose="test purpose",
        owns_stages=["inspect"],
        inputs=["train.csv"],
        outputs=["schema_summary"],
        guardrails=["label-blind"],
        handoff_to=[],
    )


def test_parse_stage_subagent_response_extracts_structured_output() -> None:
    agent = _agent()
    output = parse_stage_subagent_response(
        agent,
        {"response_content": '<stage_agent_output_json>{"agent_id":"data_auditor","status":"ok","confidence":"high","summary":"schema ok","outputs":{"schema_summary":"2 rows"},"blockers":[],"recommendations":["use x"],"handoff_notes":["preserve id"]}</stage_agent_output_json>'},
    )

    assert output.agent_id == "data_auditor"
    assert output.status == "ok"
    assert not output.blocked
    assert output.outputs["schema_summary"] == "2 rows"


def test_stage_pipeline_filters_non_audit_agents_from_pre_modeling() -> None:
    plan = StageAgentPlan("test", [_agent("data_auditor"), _agent("modeling_specialist"), _agent("probability_calibrator"), _agent("verifier_finalizer")])

    assert [agent.id for agent in executable_pre_modeling_agents(plan)] == ["data_auditor"]


def test_render_stage_subagent_prompt_has_independent_agent_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    profile = SimpleNamespace(train_rows_sampled=2, test_rows_sampled=2, target_summary={"target": {"mean": 0.5}}, warnings=[])
    route = route_task(contract, None)
    protocol = _protocol()
    plan = StageAgentPlan("test", [_agent()])

    prompt = render_stage_subagent_prompt(_agent(), contract, profile, route, protocol, plan)

    assert "independent sub-agent call" in prompt
    assert "stage_agent_output_json" in prompt
    assert "Do not use hidden answers" in prompt


class _FakeClient:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def create_user(self, username: str) -> str:
        return "user"

    def create_session(self, user_id: str, session_name: str, tool_mode: str = "datamodeling") -> str:
        session = f"session-{len(self.sessions)}-{session_name}"
        self.sessions.append(session)
        return session

    def upload_file(self, user_id: str, session_id: str, file_path: str, filename_to_save: str | None = None) -> None:
        self.uploads.append((session_id, filename_to_save or Path(file_path).name))

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        agent_id = "data_auditor" if "`data_auditor`" in query else "split_guardian"
        return {
            "response_content": f'<stage_agent_output_json>{{"agent_id":"{agent_id}","status":"ok","confidence":"medium","summary":"done","outputs":{{"evidence":"ok"}},"blockers":[],"recommendations":[],"handoff_notes":[]}}</stage_agent_output_json>'
        }

    def stop_session(self, user_id: str, session_id: str) -> None:
        self.stopped.append(session_id)


def test_runner_executes_pre_modeling_stage_subagents(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    profile = SimpleNamespace(train_rows_sampled=2, test_rows_sampled=2, target_summary={"target": {"mean": 0.5}}, warnings=[])
    route = route_task(contract, None)
    protocol = _protocol()
    plan = StageAgentPlan("test", [_agent("data_auditor"), _agent("split_guardian"), _agent("verifier_finalizer")])
    client = _FakeClient()
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(output_dir=tmp_path / "out", workspace_root=tmp_path / "ws", stage_subagent_timeout_seconds=5),
        client,
    )

    result = runner._run_stage_subagent_pipeline("user", contract, profile, route, protocol, plan, tmp_path / "artifacts")

    assert [output.agent_id for output in result.outputs] == ["data_auditor", "split_guardian"]
    assert len(client.sessions) == 2
    assert len(client.stopped) == 2
    assert (tmp_path / "artifacts" / "stage_subagents" / "00_data_auditor_prompt.txt").exists()
    uploaded_by_session = {session: [] for session in client.sessions}
    for session, filename in client.uploads:
        uploaded_by_session.setdefault(session, []).append(filename)
    assert uploaded_by_session[client.sessions[0]] == ["train.csv", "test.csv", "sample_submission.csv"]
    assert uploaded_by_session[client.sessions[1]] == ["train.csv"]
