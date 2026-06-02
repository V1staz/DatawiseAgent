"""Read-side retrieval for local self-updated harness memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re

from .schemas import QuestionContract
from .self_memory_schema import retrieval_view

_SCOPE_WEIGHTS = {
    "global": 0.5,
    "domain": 1.0,
    "task": 1.5,
    "episode": 1.25,
    "question": 0.25,  # question-local memories are stored for audit but not promoted by default.
}


class SelfMemoryRetriever:
    """Retrieve self-updated non-label memories by task-feature overlap."""

    def __init__(self, path: str | Path | None = None, *, allow_question_scope: bool = False) -> None:
        self.path = Path(path) if path else None
        self.allow_question_scope = allow_question_scope

    def retrieve(
        self,
        contract: QuestionContract | None,
        columns: list[str] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if self.path is None or contract is None or limit <= 0:
            return []
        query_concepts = set(contract.concepts or [])
        query_methods = set(contract.methods or [])
        query_columns = {_norm_column(col) for col in (columns or contract.required_columns or [])}
        query_text = _contract_text(contract)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._iter_records():
            scope = str(item.get("scope_level", "episode"))
            if scope == "question" and not self.allow_question_scope:
                # Store question-local experiences for audit, but do not inject them
                # unless explicitly allowed; this prevents benchmark-id overfitting.
                continue
            features = item.get("question_features", {}) or {}
            concepts = set(features.get("concepts", []) or item.get("concepts", []) or [])
            methods = set(features.get("methods", []) or item.get("methods", []) or [])
            mem_columns = {_norm_column(col) for col in (features.get("columns", []) or item.get("columns", []) or [])}
            keywords = set(str(k).lower() for k in (item.get("keywords", []) or []))

            concept_overlap = len(query_concepts & concepts)
            method_overlap = len(query_methods & methods)
            column_overlap = len(query_columns & mem_columns)
            keyword_hits = sum(1 for keyword in keywords if keyword and keyword in query_text)
            score = concept_overlap * 3 + method_overlap * 2 + column_overlap + keyword_hits
            score += _SCOPE_WEIGHTS.get(scope, 0.5)
            score += float(item.get("priority", 0) or 0) / 100.0
            if _severity(item) == "hard":
                score += 0.75
            if score > 0:
                item = dict(item)
                item["retrieval_score"] = round(score, 4)
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return _dedupe_memories([item for _, item in scored], limit=limit)

    def _iter_records(self) -> Iterable[dict[str, Any]]:
        if self.path is None:
            return []
        paths: list[Path] = []
        if self.path.is_dir():
            paths.extend(sorted(self.path.rglob("*.jsonl")))
        elif self.path.exists():
            paths.append(self.path)
        else:
            return []
        records: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    view = retrieval_view(item)
                    if view is not None:
                        records.append(view)
        return records


def _contract_text(contract: QuestionContract) -> str:
    parts = [contract.question or "", contract.raw_constraints or "", contract.final_format or ""]
    parts.extend(contract.filters or [])
    return "\n".join(parts).lower()


def _norm_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _severity(item: dict[str, Any]) -> str:
    return str(item.get("severity") or item.get("priority_severity") or "medium").lower()


def _dedupe_memories(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("memory_id") or item.get("signature") or item.get("repair"))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output
