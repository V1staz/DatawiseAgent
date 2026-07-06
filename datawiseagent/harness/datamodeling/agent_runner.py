"""LLM-in-the-loop DataModeling harness runner.

This runner preserves the original DatawiseAgent/FastAPI execution path: the LLM
still reads the task, writes/runs code, trains models, and creates the final CSV.
The harness only injects label-blind task structure, validates outputs, asks the
same session for bounded repairs, and records auditable artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Protocol
import queue
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

try:  # pragma: no cover - only needed for live HTTP runs
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from .agent_prompt import render_agent_prompt, render_repair_prompt
from .contracts import build_contract, load_tasks
from .data_availability import inspect_data_availability
from .guidance import load_guidance_pack
from .memory import build_outcome_memory_card, retrieve_memory_lessons, write_outcome_memory_card
from .profiler import build_profile, write_profile
from .protocol import build_protocol_plan
from .quality import QualityReport, evaluate_submission_quality, quality_failure_category
from .repair_planner import actions_to_dict, plan_repair_actions
from .risk_scheduler import build_risk_plan
from .stage_agents import build_stage_agent_plan
from .stage_artifacts import write_finalization_stage_artifact, write_pre_execution_stage_artifacts
from .skill_cards import get_skill_cards
from .submission import SubmissionValidationResult, validate_submission
from .task_router import TaskRoute, route_task
from .verified_evidence import VerifiedModelingEvidence, build_verified_modeling_evidence


class DataModelingChatClient(Protocol):
    def create_user(self, username: str) -> str: ...

    def create_session(self, user_id: str, session_name: str, tool_mode: str = "datamodeling") -> str: ...

    def upload_file(self, user_id: str, session_id: str, file_path: str, filename_to_save: str | None = None) -> None: ...

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, Any]: ...

    def stop_session(self, user_id: str, session_id: str) -> Any: ...


@dataclass(slots=True)
class FunctionChatClient:
    create_user_func: Callable[..., Any]
    create_session_func: Callable[..., Any]
    upload_file_func: Callable[..., Any]
    chat_func: Callable[..., Any]
    stop_session_func: Callable[..., Any]

    def create_user(self, username: str) -> str:
        return str(self.create_user_func(username=username))

    def create_session(self, user_id: str, session_name: str, tool_mode: str = "datamodeling") -> str:
        return str(self.create_session_func(user_id=user_id, session_name=session_name, tool_mode=tool_mode))

    def upload_file(self, user_id: str, session_id: str, file_path: str, filename_to_save: str | None = None) -> None:
        self.upload_file_func(user_id, session_id, file_path=file_path, filename_to_save=filename_to_save)

    def chat(self, user_id: str, session_id: str, query: str, work_mode: str = "jupyter+script", timeout_seconds: int | None = None) -> dict[str, Any]:
        if _callable_accepts_kwarg(self.chat_func, "timeout_seconds"):
            response = self.chat_func(user_id, session_id, query=query, work_mode=work_mode, timeout_seconds=timeout_seconds)
        else:
            response = self.chat_func(user_id, session_id, query=query, work_mode=work_mode)
        return dict(response)

    def stop_session(self, user_id: str, session_id: str) -> Any:
        return self.stop_session_func(user_id, session_id)


@dataclass(slots=True)
class DataModelingAgentHarnessConfig:
    benchmark_root: str | Path = "evaluation/DataModeling"
    output_dir: str | Path = "evaluation/experimental_results/Datamodeling-DSBench/agent-harness"
    workspace_root: str | Path = "log/workspace/users"
    model_name: str = "agent-harness"
    max_repair_attempts: int = 2
    chat_timeout_seconds: int = 3600
    early_stop_on_valid_submission: bool = True
    early_stop_min_seconds: float = 60.0
    early_stop_poll_seconds: float = 10.0
    early_stop_grace_seconds: float = 20.0
    enforce_quality_gate: bool = True
    fresh_session_per_repair: bool = True
    score: bool = False
    stop_session: bool = True
    num_workers: int = 1
    verified_evidence: bool = True
    verified_evidence_budget_seconds: int = 90
    verified_evidence_max_candidates: int = 3
    preserve_scoreable_low_confidence: bool = True


@dataclass(slots=True)
class AgentHarnessTaskRecord:
    id: int | None
    name: str
    input_text: str
    response: str
    session_id: str | None
    user_id: str | None
    time: float
    cost: None
    model: str
    input: None
    output: None
    result_path: str
    harness: dict[str, Any]
    failure: bool = False

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class AgentHarnessTaskResult:
    record: AgentHarnessTaskRecord
    artifact_dir: Path


class DataModelingAgentHarnessRunner:
    def __init__(self, config: DataModelingAgentHarnessConfig, client: DataModelingChatClient) -> None:
        self.config = config
        self.client = client
        self.benchmark_root = Path(config.benchmark_root).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.workspace_root = Path(config.workspace_root).resolve()
        self.artifact_root = self.output_dir / "artifacts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()

    def run_tasks(self, tasks: Iterable[str | int] | None = None, *, resume: bool = True) -> list[AgentHarnessTaskRecord]:
        task_items = list(tasks) if tasks is not None else [i for i, _ in enumerate(load_tasks(self.benchmark_root))]
        completed_ids, completed_names = self._completed_keys() if resume else (set(), set())
        pending = [task for task in task_items if not self._is_completed(task, completed_ids, completed_names)]
        if self.config.num_workers <= 1:
            records = []
            for task in pending:
                result = self.run_task(task)
                self._append_result(result.record)
                records.append(result.record)
            return records

        from concurrent.futures import ThreadPoolExecutor, as_completed

        records: list[AgentHarnessTaskRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_task = {executor.submit(self.run_task, task): task for task in pending}
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                except Exception as exc:  # Defensive: run_task should normally capture task failures.
                    result = self._setup_failure_result(future_to_task[future], time.time(), exc)
                self._append_result(result.record)
                records.append(result.record)
        return records

    def run_task(self, task: str | int) -> AgentHarnessTaskResult:
        start = time.time()
        events: list[dict[str, Any]] = []
        try:
            contract = build_contract(self.benchmark_root, task)
            artifact_dir = self._artifact_dir(contract.task_id, contract.name)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            def mark(stage: str, **payload: Any) -> None:
                event = {
                    "stage": stage,
                    "elapsed_seconds": round(time.time() - start, 6),
                }
                event.update(payload)
                events.append(event)

            mark("contract_built", task_id=contract.task_id, task=contract.name)
            description = Path(contract.task_description_path).read_text(encoding="utf-8", errors="replace")
            availability = inspect_data_availability(contract)
            data_availability_path = artifact_dir / "data_availability.json"
            data_availability_path.write_text(json.dumps(availability.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            mark(
                "data_availability_checked",
                usable_for_modeling=availability.usable_for_modeling,
                lfs_pointer_files=availability.lfs_pointer_files,
            )
            profile = build_profile(contract)
            mark("profiled", train_rows_sampled=profile.train_rows_sampled, test_rows_sampled=profile.test_rows_sampled)
            route = route_task(contract, profile, availability=availability)
            selected_skill_cards = get_skill_cards(route.skill_card_ids)
            selected_skill_cards_path = artifact_dir / "selected_skill_cards.json"
            selected_skill_cards_path.write_text(
                json.dumps([card.to_dict() for card in selected_skill_cards], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mark("routed", family=route.family, metric_family=route.metric_family, skill_card_ids=route.skill_card_ids)
            risk_plan = build_risk_plan(contract, profile, route)
            risk_plan_path = artifact_dir / "risk_plan.json"
            risk_plan_path.write_text(json.dumps(risk_plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            prior_memory_lessons = retrieve_memory_lessons(self.output_dir, route)
            prior_memory_lessons_path = artifact_dir / "prior_memory_lessons.json"
            prior_memory_lessons_path.write_text(json.dumps(prior_memory_lessons, ensure_ascii=False, indent=2), encoding="utf-8")
            mark("risk_scheduled", risk_level=risk_plan.risk_level, reasons=list(risk_plan.reasons))
            protocol_plan = build_protocol_plan(contract, profile, route, availability=availability)
            protocol_plan_path = artifact_dir / "protocol_plan.json"
            protocol_plan_path.write_text(json.dumps(protocol_plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            stage_agent_plan = build_stage_agent_plan(protocol_plan)
            stage_agent_plan_path = artifact_dir / "stage_agent_plan.json"
            stage_agent_plan_path.write_text(json.dumps(stage_agent_plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            mark(
                "protocol_planned",
                target_type=protocol_plan.target_type,
                split_constraint=protocol_plan.split_constraint,
                stage_agents=stage_agent_plan.required_agent_ids,
            )
            verified_evidence = (
                build_verified_modeling_evidence(
                    contract,
                    profile,
                    route,
                    protocol_plan,
                    availability,
                    artifact_dir / "verified_evidence",
                    budget_seconds=self.config.verified_evidence_budget_seconds,
                    max_candidates=self.config.verified_evidence_max_candidates,
                )
                if self.config.verified_evidence
                else None
            )
            verified_evidence_path = (
                verified_evidence.artifact_paths.get("verified_modeling_evidence")
                if verified_evidence is not None
                else None
            )
            mark(
                "verified_evidence_built" if verified_evidence is not None else "verified_evidence_disabled",
                status=getattr(verified_evidence, "status", None),
                trust=getattr(verified_evidence, "validation_metric_trust", None),
                best_candidate=getattr(verified_evidence, "best_candidate_name", None),
            )
            stage_artifact_paths = write_pre_execution_stage_artifacts(
                artifact_dir,
                contract=contract,
                profile=profile,
                availability=availability,
                route=route,
                protocol_plan=protocol_plan,
                verified_evidence=verified_evidence,
            )
            mark("stage_artifacts_written", artifact_count=len(stage_artifact_paths))
            guidance = load_guidance_pack()
            prompt = render_agent_prompt(
                contract,
                profile,
                description,
                route=route,
                guidance=guidance,
                risk_plan=risk_plan,
                availability=availability,
                protocol_plan=protocol_plan,
                stage_agent_plan=stage_agent_plan,
                prior_memory_lessons=prior_memory_lessons,
                verified_evidence=verified_evidence,
            )
            (artifact_dir / "contract.json").write_text(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            write_profile(profile, artifact_dir / "profile.json")
            (artifact_dir / "route.json").write_text(json.dumps(route.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            guidance_artifacts = guidance.write_artifacts(artifact_dir, route.scopes)
            (artifact_dir / "initial_prompt.txt").write_text(prompt, encoding="utf-8")
            mark("prompted", prompt_path=str(artifact_dir / "initial_prompt.txt"))
        except Exception as exc:
            return self._setup_failure_result(task, start, exc)

        user_id: str | None = None
        session_id: str | None = None
        response_content = ""
        result_path = ""
        validation_payloads: list[dict[str, Any]] = []
        quality_payloads: list[dict[str, Any]] = []
        submission_artifacts: list[dict[str, Any]] = []
        repair_prompts: list[str] = []
        repair_plan_paths: list[str] = []
        chat_response_paths: list[str] = []
        early_stop_count = 0
        failure = False
        failure_category = "none"
        score_result: dict[str, Any] | None = None
        quality_confidence = "unavailable"
        current_session_stopped = False

        try:
            user_name = f"{self.config.model_name}-{contract.task_id}-{contract.name}"
            user_id = self.client.create_user(user_name)
            session_id = self._create_task_session(user_id, contract)
            mark("session_created", session_id=session_id)

            chat_response = self._chat_with_timeout(
                user_id,
                session_id,
                prompt,
                contract=contract,
                route=route,
                protocol_plan=protocol_plan,
                stage_agent_plan=stage_agent_plan,
                verified_evidence=verified_evidence,
                artifact_dir=artifact_dir,
                attempt=0,
            )
            if chat_response.get("early_stop_valid_submission"):
                early_stop_count += 1
            current_session_stopped = bool(chat_response.get("session_stopped"))
            chat_response_paths.append(self._write_json_artifact(artifact_dir / "chat_response_attempt_0.json", chat_response))
            response_content = str(chat_response.get("response_content", ""))
            mark(
                "executed",
                attempt=0,
                early_stop_valid_submission=bool(chat_response.get("early_stop_valid_submission")),
                timeout=bool(chat_response.get("timeout")),
            )
            result_path = self._canonical_submission_path(user_id, session_id)
            validation, quality = self._validate_attempt(
                result_path,
                contract,
                route,
                protocol_plan,
                stage_agent_plan,
                verified_evidence,
                artifact_dir,
                attempt=0,
                submission_artifacts=submission_artifacts,
            )
            validation_payloads.append(validation.to_dict())
            quality_payloads.append(quality.to_dict())
            mark("validated", attempt=0, passed=validation.passed, errors=validation.errors[:3])
            mark("quality_checked", attempt=0, passed=quality.passed, errors=quality.errors[:3])

            attempt = 0
            while self._attempt_needs_repair(validation, quality) and attempt < self.config.max_repair_attempts:
                attempt += 1
                if self.config.fresh_session_per_repair and not current_session_stopped and user_id is not None and session_id is not None:
                    self._stop_session_quietly(user_id, session_id)
                    current_session_stopped = True
                if current_session_stopped:
                    session_id = self._create_task_session(user_id, contract)
                    current_session_stopped = False
                    result_path = self._canonical_submission_path(user_id, session_id)
                    mark("session_created", attempt=attempt, session_id=session_id)
                repair_actions = plan_repair_actions(contract, validation, quality=quality, route=route, protocol_plan=protocol_plan)
                repair_plan_path = artifact_dir / f"repair_plan_attempt_{attempt}.json"
                repair_plan_path.write_text(json.dumps(actions_to_dict(repair_actions), ensure_ascii=False, indent=2), encoding="utf-8")
                repair_plan_paths.append(str(repair_plan_path))
                repair_prompt = render_repair_prompt(
                    contract,
                    validation,
                    attempt=attempt,
                    submission_path=result_path,
                    clean_retry=bool(self.config.fresh_session_per_repair),
                    quality=quality,
                    route=route,
                    protocol_plan=protocol_plan,
                    stage_agent_plan=stage_agent_plan,
                    repair_actions=repair_actions,
                    verified_evidence=verified_evidence,
                )
                repair_prompts.append(repair_prompt)
                (artifact_dir / f"repair_prompt_attempt_{attempt}.txt").write_text(repair_prompt, encoding="utf-8")
                mark(
                    "repair_planned",
                    attempt=attempt,
                    validation_passed=validation.passed,
                    quality_passed=quality.passed,
                    failure_category=_failure_category(validation) if not validation.passed else quality_failure_category(quality),
                    repair_action_ids=[action.id for action in repair_actions],
                )
                repair_response = self._chat_with_timeout(
                    user_id,
                    session_id,
                    repair_prompt,
                    contract=contract,
                    route=route,
                    protocol_plan=protocol_plan,
                    stage_agent_plan=stage_agent_plan,
                    verified_evidence=verified_evidence,
                    artifact_dir=artifact_dir,
                    attempt=attempt,
                )
                if repair_response.get("early_stop_valid_submission"):
                    early_stop_count += 1
                current_session_stopped = bool(repair_response.get("session_stopped"))
                chat_response_paths.append(self._write_json_artifact(artifact_dir / f"chat_response_attempt_{attempt}.json", repair_response))
                response_content += "\n\n--- REPAIR ATTEMPT %d ---\n%s" % (attempt, repair_response.get("response_content", ""))
                mark(
                    "executed",
                    attempt=attempt,
                    early_stop_valid_submission=bool(repair_response.get("early_stop_valid_submission")),
                    timeout=bool(repair_response.get("timeout")),
                )
                result_path = self._canonical_submission_path(user_id, session_id)
                validation, quality = self._validate_attempt(
                    result_path,
                    contract,
                    route,
                    protocol_plan,
                    stage_agent_plan,
                    verified_evidence,
                    artifact_dir,
                    attempt=attempt,
                    submission_artifacts=submission_artifacts,
                )
                validation_payloads.append(validation.to_dict())
                quality_payloads.append(quality.to_dict())
                mark("validated", attempt=attempt, passed=validation.passed, errors=validation.errors[:3])
                mark("quality_checked", attempt=attempt, passed=quality.passed, errors=quality.errors[:3])

            quality_confidence = _quality_confidence(validation, quality, enforce_quality_gate=self.config.enforce_quality_gate)
            scoreable_low_confidence = (
                self.config.preserve_scoreable_low_confidence
                and validation.passed
                and self.config.enforce_quality_gate
                and not quality.passed
                and not _quality_blocks_scoring(quality)
            )
            if validation.passed and (not self.config.enforce_quality_gate or quality.passed or scoreable_low_confidence):
                final_artifact = self._copy_final_submission(result_path, artifact_dir)
                result_path = str(final_artifact)
                if scoreable_low_confidence:
                    failure_category = quality_failure_category(quality)
                mark("finalized", result_path=result_path, quality_confidence=quality_confidence, scoreable_low_confidence=scoreable_low_confidence)
                stage_artifact_paths["finalization_check"] = write_finalization_stage_artifact(
                    artifact_dir,
                    result_path=result_path,
                    validation=validation,
                    quality=quality,
                    quality_confidence=quality_confidence,
                )
                if self.config.score:
                    score_result = self._score_task(contract.name, result_path)
                    mark("scored", status=score_result.get("status"), score=score_result.get("score"))
                    if score_result.get("status") != "ok":
                        failure = True
                        failure_category = "score_failed"
            else:
                failure = True
                failure_category = _failure_category(validation) if not validation.passed else quality_failure_category(quality)
                mark("failed", failure_category=failure_category)

        except Exception as exc:
            failure = True
            failure_category = f"runner_exception:{type(exc).__name__}"
            response_content += f"\nHARNESS_RUNNER_EXCEPTION:{type(exc).__name__}:{str(exc)[:500]}"
            (artifact_dir / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
            mark("runner_exception", failure_category=failure_category)
        finally:
            if session_id is not None and user_id is not None and self.config.stop_session:
                try:
                    self.client.stop_session(user_id, session_id)
                    mark("session_stop_requested", session_id=session_id)
                except Exception:
                    (artifact_dir / "stop_session_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")

        seconds = time.time() - start
        runtime = {
            "seconds": seconds,
            "user_id": user_id,
            "session_id": session_id,
            "result_path": result_path,
            "repair_count": len(repair_prompts),
            "failure_category": failure_category,
            "score": score_result,
            "fresh_session_per_repair": self.config.fresh_session_per_repair,
            "quality_confidence": quality_confidence,
        }
        (artifact_dir / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        run_state_path, event_timeline_path = self._write_run_state_artifacts(
            artifact_dir,
            events,
            task_id=contract.task_id,
            task_name=contract.name,
            failure=failure,
            failure_category=failure_category,
            repair_count=len(repair_prompts),
            route=route,
        )
        outcome_memory_card = build_outcome_memory_card(
            task_id=contract.task_id,
            task=contract.name,
            route=route,
            failure=failure,
            failure_category=failure_category,
            validation_reports=validation_payloads,
            quality_reports=quality_payloads,
            events=events,
            official_score=score_result,
        )
        with self._write_lock:
            outcome_memory_card_path, outcome_memory_store_path = write_outcome_memory_card(
                artifact_dir,
                self.output_dir,
                outcome_memory_card,
            )

        record = AgentHarnessTaskRecord(
            id=contract.task_id,
            name=contract.name,
            input_text=prompt,
            response=response_content,
            session_id=session_id,
            user_id=user_id,
            time=seconds,
            cost=None,
            model=self.config.model_name,
            input=None,
            output=None,
            result_path=result_path,
            harness={
                "mode": "llm_in_the_loop_agent_harness",
                "agent_does_modeling": True,
                "deterministic_recipe_used": False,
                "label_blind_inputs": True,
                "answers_or_gt_used_for_training": False,
                "contract_path": str(artifact_dir / "contract.json"),
                "profile_path": str(artifact_dir / "profile.json"),
                "route_path": str(artifact_dir / "route.json"),
                "data_availability_path": str(data_availability_path),
                "protocol_plan_path": str(protocol_plan_path),
                "stage_agent_plan_path": str(stage_agent_plan_path),
                "stage_artifact_paths": stage_artifact_paths,
                "verified_modeling_evidence_path": str(verified_evidence_path) if verified_evidence_path else None,
                "selected_skill_cards_path": str(selected_skill_cards_path),
                "risk_plan_path": str(risk_plan_path),
                "prior_memory_lessons_path": str(prior_memory_lessons_path),
                "run_state_path": run_state_path,
                "event_timeline_path": event_timeline_path,
                "outcome_memory_card_path": outcome_memory_card_path,
                "outcome_memory_store_path": outcome_memory_store_path,
                "guidance_artifacts": guidance_artifacts,
                "initial_prompt_path": str(artifact_dir / "initial_prompt.txt"),
                "chat_response_paths": chat_response_paths,
                "validation_reports": validation_payloads,
                "quality_reports": quality_payloads,
                "submission_artifacts": submission_artifacts,
                "repair_plan_paths": repair_plan_paths,
                "repair_count": len(repair_prompts),
                "early_stop_count": early_stop_count,
                "fresh_session_per_repair": self.config.fresh_session_per_repair,
                "quality_confidence": quality_confidence,
                "preserve_scoreable_low_confidence": self.config.preserve_scoreable_low_confidence,
                "artifact_dir": str(artifact_dir),
                "failure_category": failure_category,
                "official_score": score_result,
            },
            failure=failure,
        )
        return AgentHarnessTaskResult(record=record, artifact_dir=artifact_dir)

    def _artifact_dir(self, task_id: int | None, name: str) -> Path:
        return self.artifact_root / f"{task_id:03d}_{name}" if task_id is not None else self.artifact_root / name

    def _setup_failure_result(self, task: str | int, start: float, exc: Exception) -> AgentHarnessTaskResult:
        safe_task = str(task).replace(os.sep, "_")
        artifact_dir = self.artifact_root / f"setup_failure_{safe_task}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        seconds = time.time() - start
        record = AgentHarnessTaskRecord(
            id=int(task) if str(task).isdigit() else None,
            name=str(task),
            input_text="",
            response=f"HARNESS_SETUP_FAILURE:{type(exc).__name__}:{str(exc)[:500]}",
            session_id=None,
            user_id=None,
            time=seconds,
            cost=None,
            model=self.config.model_name,
            input=None,
            output=None,
            result_path="",
            harness={
                "mode": "llm_in_the_loop_agent_harness",
                "agent_does_modeling": False,
                "deterministic_recipe_used": False,
                "label_blind_inputs": True,
                "answers_or_gt_used_for_training": False,
                "artifact_dir": str(artifact_dir),
                "failure_category": f"setup_exception:{type(exc).__name__}",
            },
            failure=True,
        )
        return AgentHarnessTaskResult(record=record, artifact_dir=artifact_dir)

    def _upload_public_csvs(self, user_id: str, session_id: str, contract: Any) -> None:
        data_dir = Path(contract.data_dir)
        for root, _dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                full_path = Path(root) / file
                relative_path = os.path.relpath(full_path, data_dir)
                self.client.upload_file(user_id, session_id, file_path=str(full_path), filename_to_save=relative_path)

    def _create_task_session(self, user_id: str, contract: Any) -> str:
        session_id = self.client.create_session(user_id, contract.name, tool_mode="datamodeling")
        self._upload_public_csvs(user_id, session_id, contract)
        return session_id

    def _chat_with_timeout(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        *,
        contract: Any | None = None,
        route: TaskRoute | None = None,
        protocol_plan: Any | None = None,
        stage_agent_plan: Any | None = None,
        verified_evidence: VerifiedModelingEvidence | None = None,
        artifact_dir: Path | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        if not self.config.early_stop_on_valid_submission or contract is None or artifact_dir is None or attempt is None:
            return self._blocking_chat_with_timeout(user_id, session_id, prompt)

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def run_chat() -> None:
            try:
                result_queue.put(("result", self.client.chat(user_id, session_id, query=prompt, work_mode="jupyter+script", timeout_seconds=self.config.chat_timeout_seconds)))
            except Exception as exc:  # pragma: no cover - exercised by live HTTP timeouts
                result_queue.put(("exception", exc))

        import threading

        thread = threading.Thread(target=run_chat, daemon=True)
        thread.start()
        start = time.time()
        deadline = start + self.config.chat_timeout_seconds
        poll_seconds = max(0.1, float(self.config.early_stop_poll_seconds))
        min_seconds = max(0.0, float(self.config.early_stop_min_seconds))

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._stop_session_quietly(user_id, session_id)
                return {"user_content": prompt, "response_content": f"TIMEOUT={self.config.chat_timeout_seconds}", "timeout": True, "session_stopped": True}

            try:
                kind, payload = result_queue.get(timeout=min(poll_seconds, remaining))
                response = self._chat_queue_payload_to_response(kind, payload, prompt)
                if response.get("timeout"):
                    self._stop_session_quietly(user_id, session_id)
                    response["session_stopped"] = True
                return response
            except queue.Empty:
                pass

            if time.time() - start < min_seconds:
                continue
            if not self._current_final_submission_valid(
                user_id,
                session_id,
                contract,
                route=route,
                protocol_plan=protocol_plan,
                stage_agent_plan=stage_agent_plan,
                verified_evidence=verified_evidence,
            ):
                continue

            self._stop_session_quietly(user_id, session_id)
            try:
                kind, payload = result_queue.get(timeout=max(0.0, float(self.config.early_stop_grace_seconds)))
                response = self._chat_queue_payload_to_response(kind, payload, prompt)
                response["early_stop_valid_submission"] = True
                response["session_stopped"] = True
                return response
            except queue.Empty:
                return {
                    "user_content": prompt,
                    "response_content": "EARLY_STOP_VALID_SUBMISSION: strict validator accepted ./input/final_submission.csv before chat completed.",
                    "early_stop_valid_submission": True,
                    "session_stopped": True,
                }

    def _blocking_chat_with_timeout(self, user_id: str, session_id: str, prompt: str) -> dict[str, Any]:
        try:
            return self.client.chat(user_id, session_id, query=prompt, work_mode="jupyter+script", timeout_seconds=self.config.chat_timeout_seconds)
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.TimeoutException):
                self._stop_session_quietly(user_id, session_id)
                return {"user_content": prompt, "response_content": f"TIMEOUT={self.config.chat_timeout_seconds}", "timeout": True, "session_stopped": True}
            return _chat_exception_response(prompt, exc)

    def _chat_queue_payload_to_response(self, kind: str, payload: Any, prompt: str) -> dict[str, Any]:
        if kind == "result":
            return dict(payload)
        exc = payload
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return {"user_content": prompt, "response_content": f"TIMEOUT={self.config.chat_timeout_seconds}", "timeout": True}
        return _chat_exception_response(prompt, exc)

    def _stop_session_quietly(self, user_id: str, session_id: str) -> None:
        try:
            self.client.stop_session(user_id, session_id)
        except Exception:
            pass

    def _current_final_submission_valid(
        self,
        user_id: str,
        session_id: str,
        contract: Any,
        *,
        route: TaskRoute | None = None,
        protocol_plan: Any | None = None,
        stage_agent_plan: Any | None = None,
        verified_evidence: VerifiedModelingEvidence | None = None,
    ) -> bool:
        result_path = self._canonical_submission_path(user_id, session_id)
        validation = validate_submission(
            result_path,
            contract,
            strict=True,
            reject_sample_target_copy=True,
        )
        if not validation.passed:
            return False
        if not self.config.enforce_quality_gate:
            return True
        route = route or route_task(contract, None)
        quality = evaluate_submission_quality(
            result_path,
            contract,
            route,
            validation,
            require_manifest=True,
            protocol_plan=protocol_plan,
            stage_agent_plan=stage_agent_plan,
            verified_evidence=verified_evidence,
        )
        return quality.passed

    def _canonical_submission_path(self, user_id: str, session_id: str) -> str:
        return str(self.workspace_root / str(user_id) / str(session_id) / "input" / "final_submission.csv")

    def _validate_attempt(
        self,
        result_path: str,
        contract: Any,
        route: TaskRoute,
        protocol_plan: Any | None,
        stage_agent_plan: Any | None,
        verified_evidence: VerifiedModelingEvidence | None,
        artifact_dir: Path,
        *,
        attempt: int,
        submission_artifacts: list[dict[str, Any]],
    ) -> tuple[SubmissionValidationResult, QualityReport]:
        validation = validate_submission(
            result_path,
            contract,
            strict=True,
            reject_sample_target_copy=True,
            write_path=artifact_dir / f"validation_attempt_{attempt}.json",
        )
        if validation.errors == ["submission_file_missing"]:
            adopted = self._adopt_valid_candidate_submission(result_path, contract, artifact_dir, attempt)
            if adopted is not None:
                validation = adopted
        snapshot = self._snapshot_submission(result_path, artifact_dir, attempt)
        if snapshot is not None:
            submission_artifacts.append(snapshot)
        quality = evaluate_submission_quality(
            result_path,
            contract,
            route,
            validation,
            write_path=artifact_dir / f"quality_attempt_{attempt}.json",
            require_manifest=self.config.enforce_quality_gate,
            protocol_plan=protocol_plan,
            stage_agent_plan=stage_agent_plan,
            verified_evidence=verified_evidence,
        )
        return validation, quality

    def _attempt_needs_repair(self, validation: SubmissionValidationResult, quality: QualityReport) -> bool:
        if not validation.passed:
            return True
        return bool(self.config.enforce_quality_gate and not quality.passed)

    def _adopt_valid_candidate_submission(
        self,
        result_path: str,
        contract: Any,
        artifact_dir: Path,
        attempt: int,
    ) -> SubmissionValidationResult | None:
        """Copy an agent-created candidate to the canonical final path when safe.

        This is a file-placement repair, not modeling: the LLM still creates the
        predictions.  The harness only accepts explicit candidate filenames and
        only after the strict, sample-copy-rejecting validator passes.
        """

        final_path = Path(result_path)
        input_dir = final_path.parent
        candidate_names = (
            "fallback_submission.csv",
            "submission.csv",
            "predictions.csv",
            "candidate_submission.csv",
        )
        for name in candidate_names:
            candidate = input_dir / name
            if not candidate.exists() or candidate.resolve() == final_path.resolve():
                continue
            candidate_validation = validate_submission(
                candidate,
                contract,
                strict=True,
                reject_sample_target_copy=True,
            )
            if not candidate_validation.passed:
                continue
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, final_path)
            adopted = validate_submission(
                final_path,
                contract,
                strict=True,
                reject_sample_target_copy=True,
                write_path=artifact_dir / f"validation_attempt_{attempt}.json",
            )
            adopted.checks.append(f"adopted_agent_candidate:{name}")
            adopted.summary["adopted_from"] = str(candidate)
            (artifact_dir / f"validation_attempt_{attempt}.json").write_text(
                json.dumps(adopted.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return adopted
        return None

    def _snapshot_submission(self, submission_path: str, artifact_dir: Path, attempt: int) -> dict[str, Any] | None:
        source = Path(submission_path)
        if not source.exists():
            return None
        target = artifact_dir / f"submission_attempt_{attempt}.csv"
        shutil.copy2(source, target)
        meta = _file_summary(target)
        meta["attempt"] = attempt
        meta["path"] = str(target)
        (artifact_dir / f"submission_attempt_{attempt}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def _copy_final_submission(self, submission_path: str, artifact_dir: Path) -> Path:
        source = Path(submission_path)
        target = artifact_dir / "final_submission.csv"
        shutil.copy2(source, target)
        meta = _file_summary(target)
        meta["path"] = str(target)
        (artifact_dir / "final_submission.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _write_run_state_artifacts(
        self,
        artifact_dir: Path,
        events: list[dict[str, Any]],
        *,
        task_id: int | None,
        task_name: str,
        failure: bool,
        failure_category: str,
        repair_count: int,
        route: TaskRoute,
    ) -> tuple[str, str]:
        timeline_path = artifact_dir / "event_timeline.jsonl"
        timeline_path.write_text(
            "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in events),
            encoding="utf-8",
        )
        state = {
            "task_id": task_id,
            "task": task_name,
            "failure": failure,
            "failure_category": failure_category,
            "route_family": route.family,
            "metric_family": route.metric_family,
            "skill_card_ids": route.skill_card_ids,
            "repair_count": repair_count,
            "current_stage": events[-1]["stage"] if events else "unknown",
            "completed_stages": [event["stage"] for event in events],
        }
        state_path = artifact_dir / "run_state.json"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(state_path), str(timeline_path)

    @staticmethod
    def _write_json_artifact(path: Path, payload: Any) -> str:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(path)

    def _score_task(self, name: str, prediction_path: str) -> dict[str, Any]:
        eval_script = (self.benchmark_root / "evaluation" / f"{name}_eval.py").resolve()
        answer_file = (self.benchmark_root / "data" / "answers" / name / "test_answer.csv").resolve()
        prediction_file = Path(prediction_path).resolve()
        output_dir = (self.output_dir / "performances").resolve()
        if not eval_script.exists() or not answer_file.exists() or not prediction_file.exists():
            return {
                "status": "missing_inputs",
                "eval_script": str(eval_script),
                "answer_file_available": answer_file.exists(),
                "prediction_file": str(prediction_file),
            }
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(eval_script),
                "--answer_file",
                str(answer_file),
                "--predict_file",
                str(prediction_file),
                "--path",
                str(output_dir),
                "--name",
                name,
            ],
            cwd=str(self.benchmark_root.parent.resolve()),
            capture_output=True,
            text=True,
            check=False,
        )
        result_path = output_dir / name / "result.txt"
        score_text = result_path.read_text(encoding="utf-8").strip() if result_path.exists() else None
        score_value = None
        if score_text is not None:
            try:
                score_value = float(score_text)
            except ValueError:
                pass
        return {
            "status": "ok" if completed.returncode == 0 and result_path.exists() else "failed",
            "returncode": completed.returncode,
            "score": score_value,
            "score_text": score_text,
            "result_path": str(result_path),
            "stderr": completed.stderr[-1000:],
            "stdout": completed.stdout[-1000:],
        }

    def _append_result(self, record: AgentHarnessTaskRecord) -> None:
        with self._write_lock:
            with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(record.to_json_line() + "\n")

    def _completed_keys(self) -> tuple[set[int], set[str]]:
        path = self.output_dir / "results.jsonl"
        if not path.exists():
            return set(), set()
        completed_ids: set[int] = set()
        completed_names: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not _row_is_resume_complete(item, require_score=self.config.score):
                    continue
                if item.get("id") is not None:
                    completed_ids.add(int(item["id"]))
                if item.get("name"):
                    completed_names.add(str(item["name"]))
        return completed_ids, completed_names

    @staticmethod
    def _is_completed(task: str | int, completed_ids: set[int], completed_names: set[str]) -> bool:
        if isinstance(task, int) or str(task).isdigit():
            return int(task) in completed_ids
        return str(task) in completed_names


def _file_summary(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    summary: dict[str, Any] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "columns": [],
        "rows": None,
    }
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        summary["columns"] = header
        summary["rows"] = rows
    except Exception as exc:
        summary["csv_summary_error"] = f"{type(exc).__name__}:{str(exc)[:160]}"
    return summary


def _callable_accepts_kwarg(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def _failure_category(validation: SubmissionValidationResult) -> str:
    errors = validation.errors
    if not errors:
        return "unknown_validation_failure"
    first = errors[0]
    if "missing" in first:
        return "missing_submission_or_column"
    if "row_count" in first:
        return "row_count_mismatch"
    if "id_" in first:
        return "id_mismatch"
    if "sample_submission" in first:
        return "sample_submission_copy"
    if "outside_metric_range" in first:
        return "prediction_range_error"
    if "target" in first or "numeric" in first or "text" in first or "class" in first:
        return "target_format_error"
    return first.split(":", 1)[0]


def _quality_confidence(validation: SubmissionValidationResult, quality: QualityReport, *, enforce_quality_gate: bool) -> str:
    if not validation.passed:
        return "invalid"
    if not enforce_quality_gate:
        return "not_enforced"
    if quality.passed:
        return "high"
    if _quality_blocks_scoring(quality):
        return "blocked"
    return "low"


def _quality_blocks_scoring(quality: QualityReport) -> bool:
    """Return true when quality failure should not be score-preserved.

    Most low-information but structurally valid submissions can still receive a
    nonzero official score and should be preserved for score-first evaluation.
    This hard-block list is reserved for cases that are invalid, recovery-only,
    unavailable, or likely benchmark-contaminating.
    """

    hard_prefixes = (
        "protocol_public_data_unavailable_for_modeling",
        "final_identical_to_fallback_without_recovery_stage",
        "fallback_recovery_is_not_quality_final",
        "manifest_stage_failed",
    )
    hard_exact = {
        "structural_validation_failed",
    }
    return any(error in hard_exact or error.startswith(hard_prefixes) for error in quality.errors)


def _chat_exception_response(prompt: str, exc: Exception) -> dict[str, Any]:
    return {
        "user_content": prompt,
        "response_content": f"CHAT_EXCEPTION:{type(exc).__name__}:{str(exc)[:500]}",
        "chat_exception": {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        },
    }


def summarize_agent_harness_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    results_path = output_dir / "results.jsonl"
    rows = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    score_statuses: dict[str, int] = {}
    summary = {
        "rows": len(rows),
        "failures": sum(1 for row in rows if row.get("failure")),
        "validator_passed": sum(1 for row in rows if _row_validator_passed(row)),
        "quality_passed": sum(1 for row in rows if _row_quality_passed(row)),
        "average_time": sum(float(row.get("time") or 0) for row in rows) / len(rows) if rows else 0,
        "repair_attempts": sum(int((row.get("harness") or {}).get("repair_count") or 0) for row in rows),
        "score_ok": 0,
        "score_failed": 0,
        "score_missing": 0,
        "failure_categories": {},
    }
    for row in rows:
        category = (row.get("harness") or {}).get("failure_category") or "none"
        summary["failure_categories"][category] = summary["failure_categories"].get(category, 0) + 1
        status = (((row.get("harness") or {}).get("official_score") or {}).get("status") or "missing")
        score_statuses[status] = score_statuses.get(status, 0) + 1
    summary["score_ok"] = score_statuses.get("ok", 0)
    summary["score_missing"] = score_statuses.get("missing", 0)
    summary["score_failed"] = sum(v for k, v in score_statuses.items() if k not in {"ok", "missing"})
    (output_dir / "agent_harness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _row_validator_passed(row: dict[str, Any]) -> bool:
    reports = (row.get("harness") or {}).get("validation_reports") or []
    return bool(reports and reports[-1].get("passed"))


def _row_quality_passed(row: dict[str, Any]) -> bool:
    reports = (row.get("harness") or {}).get("quality_reports") or []
    return bool(reports and reports[-1].get("passed"))


def _row_is_resume_complete(row: dict[str, Any], *, require_score: bool) -> bool:
    if row.get("failure"):
        return False
    if not _row_validator_passed(row):
        return False
    quality_reports = (row.get("harness") or {}).get("quality_reports") or []
    if quality_reports and not _row_quality_passed(row):
        return False
    if require_score:
        score = (row.get("harness") or {}).get("official_score") or {}
        return score.get("status") == "ok"
    return True
