"""Recovery metadata and prompt instructions.

The first harness version keeps the existing DatawiseAgent execution engine as the
action runner. Runtime rollback is therefore represented as explicit instructions
and structured recovery events, while kernel restart/replay remains delegated to
Session.safe_execute_code_blocks/rerun_cells.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(slots=True)
class RecoveryEvent:
    trigger: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    def prompt_instructions(self) -> str:
        return """# Recovery Policy
- After each successful computation step, treat the executed cells and key dataframe shape/columns as a checkpoint.
- If a runtime error occurs, fix only the failing step first; do not silently change statistical definitions.
- If a column is missing, resolve aliases against the Data Card before proceeding.
- If a filter creates an empty sample, re-check the filter condition and report the corrected sample size.
- If final formatting is invalid, redo finalization only; do not recompute unrelated steps.
"""

    def validator_failure_event(self, errors: list[str]) -> RecoveryEvent:
        return RecoveryEvent(
            trigger="validator_failure",
            action="rerun_finalization_or_search",
            details={"errors": errors},
        )

    def to_dict(self, event: RecoveryEvent) -> dict[str, Any]:
        return asdict(event)
