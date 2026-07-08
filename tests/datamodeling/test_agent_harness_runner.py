from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import httpx
import pandas as pd

from datawiseagent.harness.datamodeling.agent_prompt import render_agent_prompt, render_repair_prompt
from datawiseagent.harness.datamodeling.agent_runner import (
    DataModelingAgentHarnessConfig,
    DataModelingAgentHarnessRunner,
    FunctionChatClient,
)
from datawiseagent.harness.datamodeling.contracts import build_contract
from datawiseagent.harness.datamodeling.profiler import build_profile
from datawiseagent.harness.datamodeling.submission import validate_submission
from tests.datamodeling.test_contracts_and_submission import make_benchmark


class FakeChatClient:
    def __init__(
        self,
        workspace_root: Path,
        *,
        repair_needed: bool = False,
        write_submission: bool = True,
        sample_copy: bool = False,
        out_of_range: bool = False,
        fallback_only: bool = False,
    ) -> None:
        self.workspace_root = workspace_root
        self.repair_needed = repair_needed
        self.write_submission = write_submission
        self.sample_copy = sample_copy
        self.out_of_range = out_of_range
        self.fallback_only = fallback_only
        self.chat_calls: list[str] = []
        self.uploaded: list[str] = []
        self.stopped: list[tuple[str, str]] = []
        self._user_counter = 0
        self._session_counter = 0

    def create_user(self, username: str) -> str:
        self._user_counter += 1
        return str(uuid.UUID(int=self._user_counter))

    def create_session(self, user_id: str, session_name: str, tool_mode: str = "datamodeling") -> str:
        assert tool_mode == "datamodeling"
        self._session_counter += 1
        session_id = str(uuid.UUID(int=1000 + self._session_counter))
        (self.workspace_root / user_id / session_id / "input").mkdir(parents=True, exist_ok=True)
        (self.workspace_root / user_id / session_id / "working").mkdir(parents=True, exist_ok=True)
        return session_id

    def upload_file(self, user_id: str, session_id: str, file_path: str, filename_to_save: str | None = None) -> None:
        target = self.workspace_root / user_id / session_id / "input" / (filename_to_save or Path(file_path).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, target)
        self.uploaded.append(str(target))

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        assert work_mode == "jupyter+script"
        assert timeout_seconds is not None
        self.chat_calls.append(query)
        output = self.workspace_root / user_id / session_id / "input" / "final_submission.csv"
        if self.fallback_only:
            pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output.parent / "fallback_submission.csv", index=False)
            self._write_manifest(output.parent, stage="fallback_recovery")
            return {"user_content": query, "response_content": "wrote fallback_submission.csv only"}
        if not self.write_submission:
            return {"user_content": query, "response_content": "did not write final submission"}
        if self.sample_copy:
            shutil.copy(output.parent / "sample_submission.csv", output)
            self._write_manifest(output.parent)
            return {"user_content": query, "response_content": "copied sample submission"}
        if self.out_of_range:
            pd.DataFrame({"id": [5, 6], "target": [-0.2, 1.4]}).to_csv(output, index=False)
            self._write_manifest(output.parent)
            return {"user_content": query, "response_content": "wrote out-of-range probabilities"}
        if self.repair_needed and len(self.chat_calls) == 1:
            pd.DataFrame({"wrong": [1, 2]}).to_csv(output, index=False)
            self._write_manifest(output.parent)
            return {"user_content": query, "response_content": "wrote invalid submission"}
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        self._write_manifest(output.parent)
        return {"user_content": query, "response_content": "wrote valid final_submission.csv"}

    def stop_session(self, user_id: str, session_id: str) -> None:
        self.stopped.append((user_id, session_id))

    @staticmethod
    def _write_manifest(input_dir: Path, *, stage: str = "improved") -> None:
        payload = {
            "stage": stage,
            "modeling_method": "toy logistic baseline" if stage == "improved" else "fallback recovery",
            "selected_reason": "validation proxy beats constant fallback" if stage == "improved" else "no improved candidate available",
            "validation_metric_name": "roc_auc_proxy",
            "validation_source": "agent_additional_validation",
            "metric_direction": "maximize",
            "validation_metric_value": 0.75 if stage == "improved" else 0.5,
            "baseline_metric_value": 0.5,
            "trained_rows": 4 if stage == "improved" else 0,
            "split_policy": "stratified holdout",
            "features_used": ["num", "cat"],
            "leakage_controls": ["public train only", "fit preprocessing on train folds"],
            "probability_range": [0.2, 0.8],
            "calibration_or_ranking_check": "roc_auc proxy beats constant fallback",
            "prediction_summary": {"target": {"nunique": 2, "min": 0.2, "max": 0.8}},
            "stage_agent_outputs": {
                "data_auditor": {"schema_summary": "toy binary schema checked"},
                "split_guardian": {"split_policy": "stratified holdout"},
                "feature_builder": {"features_used": ["num", "cat"]},
                "modeling_specialist": {"validation_metric_value": 0.75},
                "verifier_finalizer": {"final_submission": "read back"},
                "probability_calibrator": {"probability_range": [0.2, 0.8]},
            },
        }
        (input_dir / "submission_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


class BlockingAfterFinalClient(FakeChatClient):
    def __init__(self, workspace_root: Path) -> None:
        super().__init__(workspace_root)
        self.stop_event = threading.Event()

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        output = self.workspace_root / user_id / session_id / "input" / "final_submission.csv"
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        self._write_manifest(output.parent)
        self.chat_calls.append(query)
        self.stop_event.wait(timeout=5.0)
        return {"user_content": query, "response_content": "eventually returned after final submission"}

    def stop_session(self, user_id: str, session_id: str) -> None:
        super().stop_session(user_id, session_id)
        self.stop_event.set()


class BlockingAfterCandidateClient(BlockingAfterFinalClient):
    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        output = self.workspace_root / user_id / session_id / "input" / "fallback_submission.csv"
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        self._write_manifest(output.parent, stage="fallback_recovery")
        self.chat_calls.append(query)
        self.stop_event.wait(timeout=5.0)
        return {"user_content": query, "response_content": "eventually returned after fallback submission"}


class RaisesAfterFinalClient(FakeChatClient):
    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        output = self.workspace_root / user_id / session_id / "input" / "final_submission.csv"
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        FakeChatClient._write_manifest(output.parent)
        self.chat_calls.append(query)
        raise RuntimeError("simulated chat endpoint 500")


class TimeoutThenRepairClient(FakeChatClient):
    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        self.chat_calls.append(query)
        output = self.workspace_root / user_id / session_id / "input" / "final_submission.csv"
        if len(self.chat_calls) == 1:
            raise httpx.TimeoutException("simulated timeout")
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        FakeChatClient._write_manifest(output.parent)
        return {"user_content": query, "response_content": "repair wrote valid final_submission.csv"}


class NoManifestClient(FakeChatClient):
    @staticmethod
    def _write_manifest(input_dir: Path, *, stage: str = "improved") -> None:
        return None


class LegacyNoTimeoutClient:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.chat_calls = 0

    def create_user(self, username: str) -> str:
        return "legacy-user"

    def create_session(self, user_id: str, session_name: str, tool_mode: str = "datamodeling") -> str:
        session_id = "legacy-session"
        (self.workspace_root / user_id / session_id / "input").mkdir(parents=True, exist_ok=True)
        return session_id

    def upload_file(self, user_id: str, session_id: str, file_path: str, filename_to_save: str | None = None) -> None:
        target = self.workspace_root / user_id / session_id / "input" / (filename_to_save or Path(file_path).name)
        shutil.copy(file_path, target)

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script") -> dict[str, str]:
        self.chat_calls += 1
        output = self.workspace_root / user_id / session_id / "input" / "final_submission.csv"
        pd.DataFrame({"id": [5, 6], "target": [0.2, 0.8]}).to_csv(output, index=False)
        FakeChatClient._write_manifest(output.parent)
        return {"user_content": query, "response_content": "legacy client wrote valid final_submission.csv"}

    def stop_session(self, user_id: str, session_id: str) -> None:
        return None


def test_agent_prompt_states_llm_must_model_and_save_final_submission(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    profile = build_profile(contract)
    description = Path(contract.task_description_path).read_text()

    prompt = render_agent_prompt(contract, profile, description)

    assert "You, the LLM agent, must perform the modeling work" in prompt
    assert "./input/final_submission.csv" in prompt
    assert "./input/submission_manifest.json" in prompt
    assert "Do not use hidden answers" in prompt
    assert "expected_rows=2" in prompt
    assert "<task_route>" in prompt
    assert "tabular_binary_classification" in prompt
    assert "Guidance surfaces available" in prompt
    assert "AGENTS.md" in prompt
    assert "SKILLS.md" in prompt
    assert "MCP_TOOLS.md" in prompt
    assert "<selected_skill_cards>" in prompt
    assert "classification_probability_modeling" in prompt
    assert "<risk_plan>" in prompt
    assert "risk_level=" in prompt
    assert "<modeling_protocol_plan>" in prompt
    assert "binary_probability" in prompt
    assert "choose_leakage_aware_split_policy" in prompt
    assert "<stage_subagent_plan>" in prompt
    assert "split_guardian" in prompt
    assert "probability_calibrator" in prompt
    assert "<verified_modeling_evidence>" in prompt
    assert "<prior_harness_lessons>" in prompt
    assert "fallback_submission.csv" in prompt
    assert "quality gate rejects a bare valid CSV" in prompt
    assert "Avoid n_jobs=-1" in prompt
    assert "Once a valid ./input/final_submission.csv has been created" not in prompt
    assert '"roc_auc"' in prompt


def test_agent_prompt_direct_render_uses_availability_for_lfs_pointer(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path, "toy-lfs")
    data_dir = root / "data" / "data_resplit" / "toy-lfs"
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 123456\n"
    (data_dir / "train.csv").write_text(pointer, encoding="utf-8")
    (data_dir / "test.csv").write_text(pointer, encoding="utf-8")
    contract = build_contract(root, "toy-lfs")
    profile = build_profile(contract)
    description = Path(contract.task_description_path).read_text()

    prompt = render_agent_prompt(contract, profile, description)

    assert '"family": "data_unavailable_or_pointer"' in prompt
    assert "public_data_availability_guard" in prompt
    assert "data_unavailable_or_pointer" in prompt


def test_repair_prompt_contains_validator_errors_without_answers(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    contract = build_contract(root, "toy-binary")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1, 2]}).to_csv(bad, index=False)
    validation = validate_submission(bad, contract)

    prompt = render_repair_prompt(contract, validation, attempt=1, submission_path=str(bad))

    assert "columns_mismatch" in prompt
    assert "./input/final_submission.csv" in prompt
    assert "Do not use hidden answers" in prompt
    assert "modeled candidate" in prompt
    assert "submission_manifest.json" in prompt
    assert "rather than copying sample targets" in prompt
    assert "<targeted_repair_actions>" in prompt
    assert "repair_submission_schema" in prompt
    assert "test_answer" not in prompt


def test_agent_harness_repairs_invalid_submission_and_writes_artifacts(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = FakeChatClient(workspace, repair_needed=True)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=1,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert not record.failure
    assert record.harness["mode"] == "llm_in_the_loop_agent_harness"
    assert record.harness["agent_does_modeling"] is True
    assert record.harness["deterministic_recipe_used"] is False
    assert record.harness["repair_count"] == 1
    assert len(client.chat_calls) == 2
    assert "validator or quality gate found" in client.chat_calls[1]
    assert Path(record.result_path).exists()
    assert Path(record.result_path).name == "final_submission.csv"
    artifact_dir = Path(record.harness["artifact_dir"])
    assert Path(record.result_path).parent == artifact_dir
    assert (out / "results.jsonl").exists()
    assert (artifact_dir / "initial_prompt.txt").exists()
    assert (artifact_dir / "validation_attempt_0.json").exists()
    assert (artifact_dir / "validation_attempt_1.json").exists()
    assert (artifact_dir / "runtime.json").exists()
    assert (artifact_dir / "final_submission.csv").exists()
    assert (artifact_dir / "route.json").exists()
    assert (artifact_dir / "data_availability.json").exists()
    assert (artifact_dir / "protocol_plan.json").exists()
    assert (artifact_dir / "stage_agent_plan.json").exists()
    assert (artifact_dir / "stage_artifacts" / "data_audit.json").exists()
    assert (artifact_dir / "stage_artifacts" / "split_plan.json").exists()
    assert (artifact_dir / "stage_artifacts" / "feature_plan.json").exists()
    assert (artifact_dir / "stage_artifacts" / "modeling_leaderboard.json").exists()
    assert (artifact_dir / "stage_artifacts" / "finalization_check.json").exists()
    assert (artifact_dir / "verified_evidence" / "verified_modeling_evidence.json").exists()
    assert (artifact_dir / "selected_skill_cards.json").exists()
    assert (artifact_dir / "risk_plan.json").exists()
    assert (artifact_dir / "prior_memory_lessons.json").exists()
    assert (artifact_dir / "run_state.json").exists()
    assert (artifact_dir / "event_timeline.jsonl").exists()
    assert (artifact_dir / "outcome_memory_card.json").exists()
    assert (artifact_dir / "guidance_manifest.json").exists()
    assert (artifact_dir / "quality_attempt_1.json").exists()
    assert (artifact_dir / "repair_plan_attempt_1.json").exists()
    assert (artifact_dir / "submission_attempt_0.csv").exists()
    assert (artifact_dir / "submission_attempt_1.csv").exists()
    assert (artifact_dir / "chat_response_attempt_0.json").exists()
    assert (artifact_dir / "chat_response_attempt_1.json").exists()

    row = json.loads((out / "results.jsonl").read_text().strip())
    assert row["harness"]["validation_reports"][-1]["passed"] is True
    assert row["harness"]["quality_reports"][-1]["passed"] is True
    assert Path(row["harness"]["selected_skill_cards_path"]).exists()
    assert Path(row["harness"]["data_availability_path"]).exists()
    assert Path(row["harness"]["protocol_plan_path"]).exists()
    assert Path(row["harness"]["stage_agent_plan_path"]).exists()
    assert Path(row["harness"]["stage_artifact_paths"]["data_audit"]).exists()
    assert Path(row["harness"]["stage_artifact_paths"]["finalization_check"]).exists()
    assert Path(row["harness"]["verified_modeling_evidence_path"]).exists()
    assert Path(row["harness"]["risk_plan_path"]).exists()
    assert Path(row["harness"]["prior_memory_lessons_path"]).exists()
    assert Path(row["harness"]["outcome_memory_card_path"]).exists()
    assert Path(row["harness"]["outcome_memory_store_path"]).exists()
    assert Path(row["harness"]["repair_plan_paths"][0]).exists()
    repair_plan = json.loads(Path(row["harness"]["repair_plan_paths"][0]).read_text())
    assert repair_plan[0]["id"] == "repair_submission_schema"
    run_state = json.loads(Path(row["harness"]["run_state_path"]).read_text())
    assert run_state["route_family"] == "tabular_binary_classification"
    assert "classification_probability_modeling" in run_state["skill_card_ids"]
    assert "repair_planned" in run_state["completed_stages"]
    risk_plan = json.loads(Path(row["harness"]["risk_plan_path"]).read_text())
    assert risk_plan["risk_level"] in {"easy", "medium", "hard"}
    protocol_plan = json.loads(Path(row["harness"]["protocol_plan_path"]).read_text())
    assert protocol_plan["target_type"] == "binary_probability"
    assert "validate_public_data_availability" in protocol_plan["common_stages"]
    assert "emit_probabilities_not_hard_labels" in protocol_plan["task_specific_stages"]
    stage_agent_plan = json.loads(Path(row["harness"]["stage_agent_plan_path"]).read_text())
    assert "stage_agent_outputs" == stage_agent_plan["required_manifest_key"]
    assert any(agent["id"] == "probability_calibrator" for agent in stage_agent_plan["agents"])
    memory_card = json.loads(Path(row["harness"]["outcome_memory_card_path"]).read_text())
    assert memory_card["route_family"] == "tabular_binary_classification"
    assert memory_card["quality_passed"] is True
    assert memory_card["top_model_family"]
    assert memory_card["validation_metric_trust"] in {"agent_attested", "harness_verified"}
    assert memory_card["performance_tactic"]
    assert "test_answer" not in json.dumps(memory_card)
    event_lines = Path(row["harness"]["event_timeline_path"]).read_text().strip().splitlines()
    assert any(json.loads(line)["stage"] == "quality_checked" for line in event_lines)


def test_agent_harness_does_not_accept_uploaded_sample_submission_as_result(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = FakeChatClient(workspace, write_submission=False)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert record.failure
    assert record.harness["failure_category"] == "missing_submission_or_column"
    assert record.harness["validation_reports"][-1]["errors"] == ["submission_file_missing"]


def test_agent_harness_rejects_fallback_only_candidate_as_quality_final(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = FakeChatClient(workspace, fallback_only=True)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert record.failure
    assert record.harness["failure_category"] == "quality_low_information_submission"
    assert any(
        check == "adopted_agent_candidate:fallback_submission.csv"
        for check in record.harness["validation_reports"][-1]["checks"]
    )
    assert record.harness["validation_reports"][-1]["summary"]["adopted_from"].endswith("fallback_submission.csv")
    assert record.harness["quality_reports"][-1]["passed"] is False


def test_agent_harness_preserves_scoreable_low_confidence_submission(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = NoManifestClient(workspace)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert not record.failure
    assert record.harness["quality_confidence"] == "low"
    assert record.harness["quality_reports"][-1]["passed"] is False
    assert "manifest_missing" in record.harness["quality_reports"][-1]["errors"]
    assert Path(record.result_path).exists()


def test_agent_harness_early_stops_after_valid_final_submission(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = BlockingAfterFinalClient(workspace)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            chat_timeout_seconds=30,
            early_stop_min_seconds=0,
            early_stop_poll_seconds=0.01,
            early_stop_grace_seconds=0.5,
            score=False,
        ),
        client,
    )

    start = time.time()
    records = runner.run_tasks([0], resume=False)

    assert time.time() - start < 2.0
    assert len(records) == 1
    assert not records[0].failure
    assert client.stopped
    assert records[0].harness["validation_reports"][-1]["passed"] is True
    assert records[0].harness["early_stop_count"] >= 1


def test_agent_harness_does_not_early_stop_by_adopting_candidate(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = BlockingAfterCandidateClient(workspace)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            chat_timeout_seconds=1,
            early_stop_min_seconds=0,
            early_stop_poll_seconds=0.01,
            early_stop_grace_seconds=0.01,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    assert records[0].failure
    assert records[0].harness["early_stop_count"] == 0
    assert any(
        check == "adopted_agent_candidate:fallback_submission.csv"
        for check in records[0].harness["validation_reports"][-1]["checks"]
    )


def test_agent_harness_recreates_session_for_repair_after_timeout(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = TimeoutThenRepairClient(workspace)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=1,
            chat_timeout_seconds=30,
            early_stop_min_seconds=0,
            early_stop_poll_seconds=0.01,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    assert not records[0].failure
    assert len(client.chat_calls) == 2
    assert client._session_counter == 2
    assert client.stopped


def test_agent_harness_validates_submission_after_chat_exception(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = RaisesAfterFinalClient(workspace)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    assert not records[0].failure
    chat_response = json.loads(Path(records[0].harness["chat_response_paths"][0]).read_text())
    assert chat_response["chat_exception"]["type"] == "RuntimeError"


def test_function_chat_client_does_not_duplicate_internal_type_errors(tmp_path: Path) -> None:
    calls = {"count": 0}

    def bad_chat(user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, str]:
        calls["count"] += 1
        raise TypeError("internal bug after side effect")

    client = FunctionChatClient(
        create_user_func=lambda username: "u",
        create_session_func=lambda user_id, session_name, tool_mode="datamodeling": "s",
        upload_file_func=lambda user_id, session_id, file_path, filename_to_save=None: None,
        chat_func=bad_chat,
        stop_session_func=lambda user_id, session_id: None,
    )

    try:
        client.chat("u", "s", "q", timeout_seconds=1)
    except TypeError as exc:
        assert "internal bug" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected TypeError")
    assert calls["count"] == 1


def test_function_chat_client_supports_legacy_no_timeout_signature(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    legacy = LegacyNoTimeoutClient(workspace)
    client = FunctionChatClient(
        legacy.create_user,
        legacy.create_session,
        legacy.upload_file,
        legacy.chat,
        legacy.stop_session,
    )
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    assert not records[0].failure
    assert legacy.chat_calls == 1


def test_agent_harness_rejects_sample_submission_copy(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = FakeChatClient(workspace, sample_copy=True)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert record.failure
    assert record.harness["failure_category"] == "sample_submission_copy"
    assert any("target_identical_to_sample_submission" in err for err in record.harness["validation_reports"][-1]["errors"])


def test_agent_harness_rejects_probability_range_violations(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    client = FakeChatClient(workspace, out_of_range=True)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=False)

    assert len(records) == 1
    record = records[0]
    assert record.failure
    assert record.harness["failure_category"] == "prediction_range_error"
    assert any("prediction_outside_metric_range" in err for err in record.harness["validation_reports"][-1]["errors"])


def test_agent_harness_resume_ignores_failed_rows(tmp_path: Path) -> None:
    root = make_benchmark(tmp_path)
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    (out).mkdir(parents=True)
    failed_row = {
        "id": 0,
        "name": "toy-binary",
        "failure": True,
        "harness": {"validation_reports": [{"passed": False}]},
    }
    (out / "results.jsonl").write_text(json.dumps(failed_row) + "\n")
    client = FakeChatClient(workspace, write_submission=True)
    runner = DataModelingAgentHarnessRunner(
        DataModelingAgentHarnessConfig(
            benchmark_root=root,
            output_dir=out,
            workspace_root=workspace,
            model_name="fake-agent",
            max_repair_attempts=0,
            score=False,
        ),
        client,
    )

    records = runner.run_tasks([0], resume=True)

    assert len(records) == 1
    assert not records[0].failure
