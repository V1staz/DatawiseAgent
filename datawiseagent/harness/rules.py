"""Persistent rulebook for compact harness guidance.

The rulebook turns formerly monolithic prompt snippets into reusable, auditable
rule cards.  Rules are loaded from JSONL so they can be versioned, selected, and
persisted separately from the transient prompt assembled for one benchmark item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import re

from .schemas import DataCard, QuestionContract, TargetAnswer, to_plain

DEFAULT_RULEBOOK_PATH = Path(__file__).with_name("default_rules.jsonl")


@dataclass(slots=True)
class HarnessRule:
    id: str
    directive: str
    skill: str = "general"
    layer: str = "domain"
    priority: int = 50
    severity: str = "medium"
    concepts: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    source: str = "curated_harness_rulebook"
    provenance: str = "unknown"
    enabled: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "HarnessRule":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {key: payload[key] for key in payload.keys() & allowed}
        for key in ("concepts", "methods", "keywords", "target_types", "checks"):
            value = data.get(key, [])
            if isinstance(value, str):
                data[key] = [value]
            elif value is None:
                data[key] = []
            else:
                data[key] = list(value)
        return cls(**data)

    def matches_skill(self, skill_name: str) -> bool:
        return self.enabled and self.skill == skill_name

    def match_score(
        self,
        contract: QuestionContract | None,
        question: dict[str, Any] | None = None,
        data_card: DataCard | None = None,
    ) -> float:
        if not self.enabled:
            return -1.0
        if self.layer in {"global", "format"}:
            score = 1.0
        else:
            score = 0.0
        question_text = _question_text(contract, question)
        concepts = set((contract.concepts if contract else []) or [])
        methods = set((contract.methods if contract else []) or [])
        target_types = {target.type for target in (contract.target_answers if contract else [])}

        if self.concepts:
            overlap = len(concepts & set(self.concepts))
            if overlap:
                score += 4.0 * overlap
            elif self.layer != "format":
                score -= 2.0
        if self.methods:
            overlap = len(methods & set(self.methods))
            if overlap:
                score += 3.0 * overlap
        if self.target_types:
            overlap = len(target_types & set(self.target_types))
            if overlap:
                score += 2.0 * overlap
        for keyword in self.keywords:
            if keyword and keyword.lower() in question_text:
                score += 2.0
        if data_card is not None and self.skill == "machine_learning" and data_card.numeric_columns:
            score += 0.25
        if self.skill == "final_format":
            score += 6.0
        if score <= 0 and self.concepts:
            return -1.0
        return score + self.priority / 100.0


def _question_text(contract: QuestionContract | None, question: dict[str, Any] | None = None) -> str:
    pieces: list[str] = []
    if question:
        pieces.extend(str(question.get(key, "")) for key in ("question", "constraints", "format"))
    if contract:
        pieces.extend([contract.question or "", contract.raw_constraints or "", contract.final_format or ""])
        pieces.extend(contract.filters or [])
        pieces.extend(_target_text(contract.target_answers))
    return "\n".join(pieces).lower()


def _target_text(targets: Iterable[TargetAnswer]) -> list[str]:
    return [" ".join(str(getattr(target, attr, "") or "") for attr in ("name", "format", "description")) for target in targets]


class RuleBook:
    """Load, select, and render persistent harness rules."""

    def __init__(self, paths: str | Path | list[str | Path] | None = None, include_defaults: bool = True) -> None:
        self.paths: list[Path] = []
        if include_defaults:
            self.paths.append(DEFAULT_RULEBOOK_PATH)
        if paths:
            if isinstance(paths, (str, Path)):
                self.paths.append(Path(paths))
            else:
                self.paths.extend(Path(path) for path in paths)
        self.rules = self._load_rules(self.paths)

    def _load_rules(self, paths: Iterable[Path]) -> list[HarnessRule]:
        rules: list[HarnessRule] = []
        seen: set[str] = set()
        for path in paths:
            if not path.exists():
                continue
            candidates = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
            for candidate in candidates:
                with candidate.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        try:
                            payload = json.loads(line)
                            rule = HarnessRule.from_mapping(payload)
                        except Exception:
                            continue
                        # Later files override earlier defaults while preserving order.
                        if rule.id in seen:
                            rules = [existing for existing in rules if existing.id != rule.id]
                        seen.add(rule.id)
                        rules.append(rule)
        return rules

    def select(
        self,
        contract: QuestionContract | None,
        question: dict[str, Any] | None = None,
        data_card: DataCard | None = None,
        limit: int = 12,
    ) -> list[HarnessRule]:
        scored: list[tuple[float, HarnessRule]] = []
        for rule in self.rules:
            score = rule.match_score(contract, question=question, data_card=data_card)
            if score > 0:
                scored.append((score, rule))
        scored.sort(key=lambda item: (item[0], item[1].priority), reverse=True)
        selected: list[HarnessRule] = []
        seen: set[str] = set()
        # Always include final formatting rules if present, then highest-scoring task rules.
        for _, rule in scored:
            if rule.id not in seen and rule.skill == "final_format":
                selected.append(rule)
                seen.add(rule.id)
        for _, rule in scored:
            if rule.id not in seen:
                selected.append(rule)
                seen.add(rule.id)
            if len(selected) >= limit:
                break
        return selected

    def select_for_skills(self, skill_names: Iterable[str], limit: int | None = None) -> list[HarnessRule]:
        wanted = set(skill_names)
        selected = [rule for rule in self.rules if rule.enabled and rule.skill in wanted]
        selected.sort(key=lambda rule: rule.priority, reverse=True)
        if limit is not None:
            selected = selected[:limit]
        return selected

    def to_prompt(self, rules: list[HarnessRule], title: str = "# Routed Persistent Rule Memory") -> str:
        if not rules:
            return ""
        lines = [title, "These reusable rules are selected by task features; they are not benchmark labels or human report text."]
        for rule in rules:
            checks = ", ".join(rule.checks[:5])
            suffix = f" checks={checks}" if checks else ""
            lines.append(f"- [{rule.severity}] {rule.id}: {rule.directive}{suffix}")
        return "\n".join(lines)

    def manifest(self, rules: list[HarnessRule]) -> list[dict[str, Any]]:
        return [to_plain(rule) for rule in rules]

    def append_rule(self, rule: HarnessRule, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_plain(rule), ensure_ascii=False, sort_keys=True) + "\n")


def normalize_rule_id(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", ".", text.lower()).strip(".")
    return value[:80] or "rule"
