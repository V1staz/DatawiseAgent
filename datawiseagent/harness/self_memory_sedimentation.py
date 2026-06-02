"""Write-side sedimentation for local self-updated harness memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .schemas import FailureMemory, FinalAnswerBlock, QuestionContract, ValidationResult, to_plain
from .self_memory_schema import with_self_memory_defaults


class SelfMemoryWriter:
    """Append self-updated memories derived from harness/agent artifacts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None

    def add(self, memory: FailureMemory | dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = with_self_memory_defaults(to_plain(memory))
        self._append_payload(payload)

    def _append_payload(self, payload: dict[str, Any]) -> None:
        assert self.path is not None
        target = self.path / "self_updated_memory.jsonl" if self.path.exists() and self.path.is_dir() else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def record_validator_failure(
        self,
        contract: QuestionContract | None,
        failure_type: str,
        errors: list[str],
    ) -> None:
        if contract is None:
            return
        signature = _signature(contract, failure_type)
        self.add(
            {
                "signature": signature,
                "question_features": _question_features(contract),
                "failure_type": failure_type,
                "failed_action": "finalization",
                "repair": _repair_for_validator_errors(errors),
                "validator_that_caught_it": ",".join(errors[:5]),
                "source": "validator_report",
                "scope_level": "task",
                "severity": "hard" if errors else "medium",
                "origin_question_id": contract.question_id,
                "contains_benchmark_label": False,
            }
        )

    def record_runtime_failure(
        self,
        contract: QuestionContract | None,
        runtime: dict[str, Any] | None,
    ) -> None:
        if contract is None or not runtime or not runtime.get("error"):
            return
        error = str(runtime.get("error"))
        self.add(
            {
                "signature": _signature(contract, "runtime_failure"),
                "question_features": _question_features(contract),
                "failure_type": "runtime_failure",
                "failed_action": "agent_runtime",
                "repair": "Treat runtime/UI/session errors as recovery triggers: preserve computed evidence when present, retry only the failing operation or finalization path, and avoid changing statistical definitions.",
                "validator_that_caught_it": error[:240],
                "source": "runtime_error",
                "scope_level": "episode",
                "severity": "hard",
                "origin_question_id": contract.question_id,
                "contains_benchmark_label": False,
            }
        )

    def record_semantic_failure(
        self,
        contract: QuestionContract | None,
        semantic_validation: ValidationResult | dict[str, Any] | None,
    ) -> None:
        if contract is None or semantic_validation is None:
            return
        result = _validation_to_dict(semantic_validation)
        errors = result.get("errors", []) or []
        warnings = result.get("warnings", []) or []
        if not errors and not warnings:
            return
        self.add(
            {
                "signature": _signature(contract, "semantic_oracle_gap"),
                "question_features": _question_features(contract),
                "failure_type": "semantic_oracle_gap",
                "failed_action": "method_verification",
                "repair": _repair_for_semantic_messages(errors or warnings),
                "validator_that_caught_it": ",".join((errors or warnings)[:5]),
                "source": "semantic_oracle_report",
                "scope_level": "task",
                "severity": "hard" if errors else "medium",
                "origin_question_id": contract.question_id,
                "contains_benchmark_label": False,
            }
        )

    def record_branch_outcome(
        self,
        contract: QuestionContract | None,
        branch_name: str,
        validation: ValidationResult | dict[str, Any],
        semantic_validation: ValidationResult | dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        final_block: FinalAnswerBlock | None = None,
    ) -> None:
        if contract is None:
            return
        validation_dict = _validation_to_dict(validation)
        semantic_dict = _validation_to_dict(semantic_validation) if semantic_validation else {}
        runtime = runtime or {}
        if (
            validation_dict.get("passed")
            and semantic_dict.get("passed", True)
            and not semantic_dict.get("warnings")
            and not runtime.get("error")
        ):
            # Successful branches become compact positive process memories only
            # when they contain method evidence, not answer labels.
            evidence = final_block.evidence if final_block else {}
            if evidence:
                self.add(
                    {
                        "signature": _signature(contract, "successful_method_trace"),
                        "question_features": _question_features(contract),
                        "failure_type": "positive_process_trace",
                        "failed_action": None,
                        "repair": _positive_lesson_from_evidence(evidence),
                        "source": "agent_final_block_evidence",
                        "scope_level": "domain",
                        "severity": "medium",
                        "origin_question_id": contract.question_id,
                        "branch_name": branch_name,
                        "contains_benchmark_label": False,
                    }
                )
            return
        if validation_dict.get("errors"):
            self.record_validator_failure(contract, "final_validator_failure", list(validation_dict.get("errors") or []))
        if semantic_dict and (semantic_dict.get("errors") or semantic_dict.get("warnings")):
            self.record_semantic_failure(contract, semantic_dict)
        if runtime.get("error"):
            self.record_runtime_failure(contract, runtime)


def _question_features(contract: QuestionContract) -> dict[str, Any]:
    return {
        "concepts": contract.concepts,
        "methods": contract.methods,
        "columns": contract.required_columns,
        "level": contract.level,
    }


def _signature(contract: QuestionContract, suffix: str) -> str:
    parts = [
        str(contract.level or "unknown"),
        "+".join(sorted(contract.concepts)) or "no_concept",
        "+".join(sorted(contract.methods)) or "no_method",
        suffix,
    ]
    return "|".join(parts)


def _repair_for_validator_errors(errors: list[str]) -> str:
    text = " ".join(errors)
    if "final_block_missing" in text:
        return "Redo finalization only: emit one fenced JSON FinalAnswerBlock with contract answer names, answers, format_targets, evidence, and verified=true."
    if "format_target_answer_mismatch" in text:
        return "Rebuild format_targets from FinalAnswerBlock.answers using the target-specific format, type, and rounding; do not recompute unrelated analysis."
    if "answer_names_missing" in text or "format_targets_missing" in text:
        return "Use exact contract answer names for both answers and format_targets; do not invent aliases."
    return "Re-run finalization with contract answer names and format_targets only."


def _repair_for_semantic_messages(messages: list[str]) -> str:
    text = " ".join(messages)
    if "std_expected_ddof1" in text or "std_ddof1" in text:
        return "For unspecified standard deviation, recompute or verify with sample std/ddof=1; only use population std when explicitly requested."
    if "year_2000" in text:
        return "Re-check temporal boundary wording; '2000 and beyond' must include 2000 (>= 2000) unless a stricter explicit constraint says otherwise."
    if "engineered_feature_baseline" in text or "all_numeric_baseline" in text:
        return "For engineered-feature model comparisons, use all non-target numeric predictors as baseline and add only the engineered feature for the augmented model, preserving the split."
    if "source_columns" in text:
        return "Verify required source columns against the Data Card and write them into FinalAnswerBlock.evidence.source_columns."
    return "Before finalization, fill the semantic oracle evidence fields and resolve any method-policy mismatch."


def _positive_lesson_from_evidence(evidence: dict[str, Any]) -> str:
    method = str(evidence.get("method") or "").strip()
    columns = evidence.get("source_columns") or evidence.get("feature_columns") or []
    pieces = []
    if method:
        pieces.append(f"Successful method trace: {method[:180]}")
    if columns:
        pieces.append(f"Columns/features were explicitly recorded: {columns}")
    return "; ".join(pieces) or "Successful branch included explicit method evidence before finalization."


def _validation_to_dict(validation: ValidationResult | dict[str, Any] | None) -> dict[str, Any]:
    if validation is None:
        return {}
    if isinstance(validation, dict):
        return validation
    return to_plain(validation)
