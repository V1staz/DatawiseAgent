"""Hierarchical hard-question branch-search policy for the harness controller."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from .schemas import HarnessContext, SearchPolicy, ValidationResult
from .verifier import Verifier


@dataclass(slots=True)
class SearchBranch:
    index: int
    name: str
    prompt_suffix: str
    parent_index: int | None = None
    depth: int = 0
    strategy: str = "deterministic"
    oracle_focus: list[str] = field(default_factory=list)


class SearchController:
    def __init__(self, policy: SearchPolicy = "hard-only", max_branches: int = 3) -> None:
        self.policy = policy
        self.max_branches = max(1, min(max_branches, 3))
        self.verifier = Verifier()

    def pretrigger(self, context: HarnessContext) -> bool:
        if self.policy == "off":
            return False
        if self.policy == "all":
            return True
        if self.policy == "hard-only":
            q = context.question
            return str(q.get("level", "")).lower() == "hard"
        return False

    def posttrigger(self, validation: ValidationResult) -> bool:
        return self.policy == "validator-fail" and not validation.passed

    def branches(self, context: HarnessContext | None = None) -> list[SearchBranch]:
        oracle_ids = [str(item.get("id", "")) for item in (getattr(context, "oracle_checks", []) or []) if item.get("id")]
        focus = _focus_from_oracles(oracle_ids)
        branches = [
            SearchBranch(
                0,
                "deterministic_skill_branch",
                "\n# Branch Policy\nFollow the routed persistent rules deterministically. Emit hypothesis -> computation -> oracle check -> FinalAnswerBlock. Avoid alternative interpretations unless a validator or oracle check fails.\n",
                parent_index=None,
                depth=0,
                strategy="deterministic_rule_path",
                oracle_focus=oracle_ids[:2],
            ),
            SearchBranch(
                1,
                "alternative_interpretation_branch",
                "\n# Branch Policy\nTreat this branch as a sibling hypothesis: before computing, test plausible alternatives for column aliases, statistical口径, grouping grain, boundary inclusivity, and missing-value handling. Choose the interpretation best supported by the contract, Data Card, and semantic oracle plan.\n",
                parent_index=0,
                depth=1,
                strategy="alternative_hypothesis",
                oracle_focus=focus.get("interpretation", []),
            ),
            SearchBranch(
                2,
                "oracle_guarded_backtrack_branch",
                "\n# Branch Policy\nTreat this branch as a backtracking verifier: recompute from a minimal independent path, explicitly satisfy every semantic oracle evidence field, reject narrow ML baselines, wrong ddof, and boundary mistakes before finalizing.\n",
                parent_index=0,
                depth=1,
                strategy="oracle_guarded_backtrack",
                oracle_focus=focus.get("semantic", oracle_ids),
            ),
        ]
        return branches[: self.max_branches]

    def summarize(
        self,
        branch_results: list[dict[str, Any]],
        selected_index: int,
    ) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "branch_count": len(branch_results),
            "selected_branch_index": selected_index,
            "selection_basis": "format_validation + semantic_oracle_score + runtime_success + branch_cost",
            "branches": [
                {
                    "index": item.get("branch_index"),
                    "name": item.get("branch_name"),
                    "score": item.get("branch_score"),
                    "validator_passed": (item.get("validator_report") or {}).get("passed"),
                    "semantic_passed": (item.get("semantic_report") or {}).get("passed"),
                    "semantic_score": (item.get("semantic_report") or {}).get("score"),
                }
                for item in branch_results
            ],
        }

    def select(self, branch_results: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        if not branch_results:
            raise ValueError("branch_results must not be empty")
        best_pos = 0
        best_score = -1.0
        for pos, item in enumerate(branch_results):
            score = float(item.get("branch_score", 0.0))
            if score > best_score:
                best_pos = pos
                best_score = score
        return best_pos, branch_results[best_pos]


def _focus_from_oracles(oracle_ids: list[str]) -> dict[str, list[str]]:
    interpretation = [item for item in oracle_ids if "filter" in item or "std" in item or "data" in item]
    semantic = [item for item in oracle_ids if "ml" in item or "std" in item or "filter" in item]
    return {"interpretation": interpretation, "semantic": semantic}


def branch_to_dict(branch: SearchBranch) -> dict[str, Any]:
    return asdict(branch)
