"""Compatibility facade for harness self-updated memory.

The memory subsystem is split into two explicit modules:

- ``self_memory_sedimentation`` writes local self-updated memories from
  validator/runtime/oracle/final-block artifacts.
- ``self_memory_retrieval`` retrieves only those local self-updated memories.

Curated rules live in ``rules.py`` and are not visible to this retrieval path.
The historic ``FailureMemoryStore`` name is kept for existing controller/tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .schemas import FailureMemory, FinalAnswerBlock, QuestionContract, ValidationResult
from .self_memory_retrieval import SelfMemoryRetriever
from .self_memory_sedimentation import SelfMemoryWriter


class FailureMemoryStore:
    """Facade preserving the previous API while separating write/read roles."""

    def __init__(self, path: str | Path | None = None, *, allow_question_scope: bool = False) -> None:
        self.path = Path(path) if path else None
        self.allow_question_scope = allow_question_scope
        self.writer = SelfMemoryWriter(self.path)
        self.retriever = SelfMemoryRetriever(self.path, allow_question_scope=allow_question_scope)

    def add(self, memory: FailureMemory | dict[str, Any]) -> None:
        self.writer.add(memory)

    def retrieve(
        self,
        contract: QuestionContract | None,
        columns: list[str] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(contract, columns=columns, limit=limit)

    # Private compatibility hooks for older tests/tools that may have reached
    # through the old monolithic implementation.
    def _append_payload(self, payload: dict[str, Any]) -> None:
        self.writer.add(payload)

    def _iter_records(self) -> Iterable[dict[str, Any]]:
        return self.retriever._iter_records()

    def record_validator_failure(
        self,
        contract: QuestionContract | None,
        failure_type: str,
        errors: list[str],
    ) -> None:
        self.writer.record_validator_failure(contract, failure_type, errors)

    def record_runtime_failure(
        self,
        contract: QuestionContract | None,
        runtime: dict[str, Any] | None,
    ) -> None:
        self.writer.record_runtime_failure(contract, runtime)

    def record_semantic_failure(
        self,
        contract: QuestionContract | None,
        semantic_validation: ValidationResult | dict[str, Any] | None,
    ) -> None:
        self.writer.record_semantic_failure(contract, semantic_validation)

    def record_branch_outcome(
        self,
        contract: QuestionContract | None,
        branch_name: str,
        validation: ValidationResult | dict[str, Any],
        semantic_validation: ValidationResult | dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        final_block: FinalAnswerBlock | None = None,
    ) -> None:
        self.writer.record_branch_outcome(
            contract,
            branch_name=branch_name,
            validation=validation,
            semantic_validation=semantic_validation,
            runtime=runtime,
            final_block=final_block,
        )
