"""Policy schema for agent self-updated harness memory.

This module is intentionally separate from ``rules.py``.  Rule cards are
curated, versioned project configuration; self memory is local runtime sediment
written by the harness from agent artifacts.  Retrieval must only see the latter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from typing import Any
import json

SELF_UPDATED_MEMORY_KIND = "self_updated"
SELF_MEMORY_PROVENANCE = "agent_harness_artifact"
SELF_MEMORY_COPY_POLICY = "local_self_update_only_not_for_rulebook_or_project_transfer"
SELF_MEMORY_REUSE_POLICY = "retrieve_self_updated_by_concept_method_column_overlap_no_labels"

# Sources the harness itself can create from one run's execution artifacts.  They
# are not human reports, paper summaries, benchmark labels, or curated rule cards.
SELF_MEMORY_SOURCES = frozenset(
    {
        "harness_execution",
        "validator_report",
        "runtime_error",
        "semantic_oracle_report",
        "agent_final_block_evidence",
        "agent_trace",
    }
)

BLOCKED_MEMORY_KINDS = frozenset(
    {
        "curated_rule",
        "human_report",
        "copy_transferred",
        "external_transfer",
        "benchmark_label",
    }
)
BLOCKED_MEMORY_SOURCES = frozenset(
    {
        "curated_harness_rulebook",
        "human_report",
        "manual_report",
        "analysis_report",
        "paper_report",
        "copied_memory",
        "transferred_memory",
    }
)


def with_self_memory_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a write-ready self-memory record with non-transfer metadata.

    The writer path is only for model/harness self-updated memory.  Even if a
    caller passes sparse or legacy-shaped data, the persisted record is marked as
    local, non-transferable, and distinct from curated rulebook entries.
    """

    ensure_self_memory_write_allowed(payload)
    result = dict(payload)
    result.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    result.setdefault("source", "harness_execution")
    result.setdefault("scope_level", infer_scope_level(result))
    result.setdefault("memory_id", memory_id(result))
    result["memory_kind"] = SELF_UPDATED_MEMORY_KIND
    result["provenance"] = SELF_MEMORY_PROVENANCE
    result["contains_benchmark_label"] = False
    result["human_report_derived"] = False
    result["curated_rule"] = False
    result["transferable"] = False
    result["copy_policy"] = SELF_MEMORY_COPY_POLICY
    result["reuse_policy"] = SELF_MEMORY_REUSE_POLICY
    return result


def ensure_self_memory_write_allowed(payload: dict[str, Any]) -> None:
    """Reject records that belong to rule/report/transfer/label surfaces."""

    if not isinstance(payload, dict):
        raise TypeError("self memory payload must be a mapping")
    blocked_flags = [
        name
        for name in ("contains_benchmark_label", "human_report_derived", "curated_rule", "transferable")
        if truthy(payload.get(name))
    ]
    if blocked_flags:
        raise ValueError(f"self memory write blocked by flags: {','.join(blocked_flags)}")

    kind = payload.get("memory_kind")
    if kind is not None and kind != SELF_UPDATED_MEMORY_KIND:
        raise ValueError(f"self memory write blocked by memory_kind={kind!r}")

    source = str(payload.get("source") or "harness_execution")
    if source in BLOCKED_MEMORY_SOURCES:
        raise ValueError(f"self memory write blocked by source={source!r}")

    copy_policy = payload.get("copy_policy")
    if copy_policy is not None and str(copy_policy) != SELF_MEMORY_COPY_POLICY:
        raise ValueError(f"self memory write blocked by copy_policy={copy_policy!r}")


def retrieval_view(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized retrieval view, or ``None`` if policy blocks it."""

    if not is_retrievable_self_memory(payload):
        return None
    result = dict(payload)
    result.setdefault("source", "harness_execution")
    result.setdefault("memory_kind", SELF_UPDATED_MEMORY_KIND)
    result.setdefault("provenance", SELF_MEMORY_PROVENANCE)
    result.setdefault("transferable", False)
    result.setdefault("copy_policy", SELF_MEMORY_COPY_POLICY)
    result.setdefault("reuse_policy", SELF_MEMORY_REUSE_POLICY)
    result.setdefault("scope_level", infer_scope_level(result))
    result.setdefault("memory_id", memory_id(result))
    return result


def is_retrievable_self_memory(payload: dict[str, Any]) -> bool:
    """Strict gate for memory retrieval.

    Retrieval is for self-updated local memories only.  Curated rules, human
    analysis, transferred memories, and benchmark-label-bearing artifacts are
    intentionally invisible to this path.  Legacy records without
    ``memory_kind`` are accepted only when their source is one of the harness
    self-memory artifact sources and no transfer/report flags are present.
    """

    if not isinstance(payload, dict):
        return False
    if truthy(payload.get("contains_benchmark_label")):
        return False
    if truthy(payload.get("human_report_derived")) or truthy(payload.get("curated_rule")):
        return False
    if truthy(payload.get("transferable")):
        return False

    copy_policy = payload.get("copy_policy")
    if copy_policy is not None and str(copy_policy) != SELF_MEMORY_COPY_POLICY:
        return False

    kind = payload.get("memory_kind")
    if kind in BLOCKED_MEMORY_KINDS:
        return False

    source = str(payload.get("source") or "harness_execution")
    if source in BLOCKED_MEMORY_SOURCES:
        return False

    if kind == SELF_UPDATED_MEMORY_KIND:
        return source in SELF_MEMORY_SOURCES
    if kind is None:
        # Backward compatibility for earlier harness experiments that wrote
        # agent-derived records before memory_kind/copy_policy existed.
        return source in SELF_MEMORY_SOURCES
    return False


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def infer_scope_level(payload: dict[str, Any]) -> str:
    if payload.get("scope_level"):
        return str(payload["scope_level"])
    if payload.get("origin_question_id") and not payload.get("repair"):
        return "question"
    if payload.get("question_features", {}).get("methods"):
        return "task"
    return "episode"


def memory_id(payload: dict[str, Any]) -> str:
    basis = json.dumps(
        {
            "signature": payload.get("signature"),
            "failure_type": payload.get("failure_type"),
            "repair": payload.get("repair"),
            "features": payload.get("question_features"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "mem-" + sha1(basis.encode("utf-8")).hexdigest()[:12]
