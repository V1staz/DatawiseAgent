"""Trace writer for compact, auditable harness events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import TraceEvent, append_jsonl


class TraceWriter:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._step_id = 0

    def event(
        self,
        stage: str,
        skill: str | None = None,
        reason_summary: str | None = None,
        action: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> TraceEvent:
        self._step_id += 1
        event = TraceEvent(
            step_id=self._step_id,
            stage=stage,
            skill=skill,
            reason_summary=reason_summary,
            action=action or {},
            observation=observation or {},
            validation=validation or {},
        )
        if self.path is not None:
            append_jsonl(self.path, event)
        return event
