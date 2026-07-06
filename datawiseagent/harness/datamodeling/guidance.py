"""Layered guidance loading for DataModeling harness prompts.

Guidance lives in markdown/JSONL artifacts beside the harness rather than in one
large code string.  The prompt renderer injects a concise route-specific excerpt
and writes the complete artifacts for audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


GUIDANCE_DIR = Path(__file__).with_name("guidance")


@dataclass(slots=True)
class GuidanceRule:
    id: str
    scope: str
    priority: int
    text: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuidanceRule":
        return cls(
            id=str(payload.get("id") or "unknown"),
            scope=str(payload.get("scope") or "all"),
            priority=int(payload.get("priority") or 0),
            text=str(payload.get("text") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataModelingGuidancePack:
    agents_md: str
    skills_md: str
    mcp_tools_md: str
    rules: list[GuidanceRule] = field(default_factory=list)

    def select_rules(self, scopes: list[str], *, limit: int = 8) -> list[GuidanceRule]:
        wanted = {"all", *scopes}
        selected = [rule for rule in self.rules if rule.scope in wanted or any(scope in rule.scope for scope in scopes)]
        return sorted(selected, key=lambda rule: (-rule.priority, rule.id))[:limit]

    def render_excerpt(self, scopes: list[str], *, limit_rules: int = 8) -> str:
        rules = self.select_rules(scopes, limit=limit_rules)
        rule_lines = "\n".join(f"- `{rule.id}` ({rule.scope}): {rule.text}" for rule in rules)
        return (
            "Guidance surfaces available in this session:\n"
            "- AGENTS.md: lifecycle, boundaries, manifest contract, finalization policy.\n"
            "- SKILLS.md: route-specific modeling skill tree.\n"
            "- MCP_TOOLS.md: contract/profile/route/validator/quality-gate tool semantics.\n\n"
            "Selected route rules:\n"
            f"{rule_lines or '- No route-specific rules selected.'}"
        )

    def manifest(self, scopes: list[str]) -> dict[str, Any]:
        return {
            "surfaces": {
                "AGENTS.md": "lifecycle/boundaries/manifest/finalization",
                "SKILLS.md": "task-family skill tree",
                "MCP_TOOLS.md": "contract/profile/route/validator/quality semantics",
                "rules.jsonl": "compact curated rule cards",
            },
            "selected_scopes": scopes,
            "selected_rules": [rule.to_dict() for rule in self.select_rules(scopes)],
        }

    def write_artifacts(self, target_dir: str | Path, scopes: list[str]) -> dict[str, str]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        files = {
            "guidance_AGENTS.md": self.agents_md,
            "guidance_SKILLS.md": self.skills_md,
            "guidance_MCP_TOOLS.md": self.mcp_tools_md,
            "guidance_manifest.json": json.dumps(self.manifest(scopes), ensure_ascii=False, indent=2),
        }
        written: dict[str, str] = {}
        for name, content in files.items():
            path = target / name
            path.write_text(content, encoding="utf-8")
            written[name] = str(path)
        rules_path = target / "guidance_rules.jsonl"
        rules_path.write_text("\n".join(json.dumps(rule.to_dict(), ensure_ascii=False) for rule in self.rules) + "\n", encoding="utf-8")
        written["guidance_rules.jsonl"] = str(rules_path)
        return written


def load_guidance_pack(root: str | Path | None = None) -> DataModelingGuidancePack:
    base = Path(root) if root is not None else GUIDANCE_DIR
    return DataModelingGuidancePack(
        agents_md=_read_text(base / "AGENTS.md"),
        skills_md=_read_text(base / "SKILLS.md"),
        mcp_tools_md=_read_text(base / "MCP_TOOLS.md"),
        rules=_read_rules(base / "rules.jsonl"),
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_rules(path: Path) -> list[GuidanceRule]:
    if not path.exists():
        return []
    rules: list[GuidanceRule] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rules.append(GuidanceRule.from_dict(json.loads(line)))
    return rules
