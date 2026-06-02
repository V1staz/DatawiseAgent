"""Harness controller that wraps the existing DatawiseAgent/FastAPI chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import time

from .contract import ContractParser
from .datacard import DataCardBuilder
from .finalizer import (
    extract_final_answer_block,
    final_block_from_legacy_targets,
    normalize_final_block_to_contract,
    render_reformat_response,
)
from .memory import FailureMemoryStore
from .oracles import SemanticOracleCheck, SemanticOracleRegistry
from .recovery import RecoveryManager
from .schemas import (
    HarnessArtifacts,
    HarnessContext,
    HarnessMode,
    SearchPolicy,
    ValidationResult,
    to_plain,
    write_json,
)
from .rules import RuleBook
from .search import SearchController
from .skills import SkillRegistry
from .trace import TraceWriter
from .verifier import Verifier

_MODE_ORDER = {
    "baseline": 0,
    "datacard": 1,
    "contract": 2,
    "skills": 3,
    "verify": 4,
    "memory": 5,
    "search": 6,
    "full": 7,
}


@dataclass(slots=True)
class HarnessConfig:
    mode: HarnessMode = "full"
    search_policy: SearchPolicy = "hard-only"
    artifact_root: str | Path = "evaluation/experimental_results/InfiAgent-Bench/harness_artifacts/default"
    memory_path: str | Path | None = None
    rulebook_path: str | Path | None = None
    max_search_branches: int = 3
    semantic_oracles: bool = True
    memory_retrieval: bool = True
    memory_recording: bool = True


class HarnessController:
    """Prepare prompts, persist artifacts, validate final blocks, and score branches."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.data_card_builder = DataCardBuilder()
        self.contract_parser = ContractParser()
        self.skill_registry = SkillRegistry()
        self.verifier = Verifier()
        self.recovery = RecoveryManager()
        self.memory = FailureMemoryStore(config.memory_path)
        self.rulebook = RuleBook(config.rulebook_path)
        self.oracles = SemanticOracleRegistry()
        self.search = SearchController(config.search_policy, config.max_search_branches)

    def enabled(self, capability: str) -> bool:
        mode = self.config.mode
        if mode == "full":
            return True
        rank = _MODE_ORDER.get(mode, 0)
        required = _MODE_ORDER.get(capability, 0)
        return rank >= required

    def prepare(self, question: dict[str, Any], table_root: str | Path) -> HarnessContext:
        qid = str(question.get("id", "unknown"))
        artifact_dir = Path(self.config.artifact_root) / f"q_{qid}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts = HarnessArtifacts(root_dir=str(artifact_dir))
        trace_path = artifact_dir / "trace.jsonl"
        trace_path.write_text("", encoding="utf-8")
        artifacts.trace_path = str(trace_path)
        trace = TraceWriter(trace_path)

        file_name = question.get("file_name", "")
        table_path = Path(table_root) / str(file_name)
        base_prompt = self._base_prompt(question)
        prompt_parts = [base_prompt]

        data_card = None
        if self.enabled("datacard"):
            data_card = self.data_card_builder.build(table_path)
            artifacts.data_card_path = str(artifact_dir / "data_card.json")
            write_json(artifacts.data_card_path, data_card)
            prompt_parts.append(self.data_card_builder.to_prompt(data_card))
            trace.event(
                stage="datacard",
                reason_summary="Pre-analyze uploaded table schema to reduce repeated head/info/describe calls.",
                action={"type": "build_data_card", "file": str(table_path)},
                observation={"shape": [data_card.row_count, data_card.column_count]},
                validation={"passed": not data_card.warnings, "warnings": data_card.warnings},
            )

        contract = None
        if self.enabled("contract"):
            contract = self.contract_parser.parse(question, data_card)
            artifacts.contract_path = str(artifact_dir / "contract.json")
            write_json(artifacts.contract_path, contract)
            contract_validation = self.verifier.validate_contract(contract, data_card)
            prompt_parts.append(self.contract_parser.to_prompt(contract))
            trace.event(
                stage="contract",
                reason_summary="Parse question constraints and output targets before execution.",
                action={"type": "parse_contract"},
                observation={"target_answers": [t.name for t in contract.target_answers]},
                validation=to_plain(contract_validation),
            )

        skills = []
        rules = []
        if self.enabled("skills"):
            skills = self.skill_registry.route(question)
            rules = self.rulebook.select(contract, question=question, data_card=data_card)
            artifacts.rule_manifest_path = str(artifact_dir / "rule_manifest.json")
            write_json(artifacts.rule_manifest_path, self.rulebook.manifest(rules))
            prompt_parts.append(self.skill_registry.to_prompt(skills, rules=rules))
            prompt_parts.append(self._toolkit_prompt())
            trace.event(
                stage="routing",
                reason_summary="Route concepts/keywords to persistent rule cards instead of one-off prompt sprawl.",
                action={"type": "route_skills_and_rules"},
                observation={"skills": [skill.name for skill in skills], "rule_ids": [rule.id for rule in rules]},
                validation={"passed": True, "checks": ["skills_routed", "persistent_rules_selected"]},
            )

        memory_hits: list[dict[str, Any]] = []
        if self.enabled("memory"):
            columns = [col.name for col in data_card.columns] if data_card else []
            if self.config.memory_retrieval:
                memory_hits = self.memory.retrieve(contract, columns=columns, limit=3)
                if memory_hits:
                    prompt_parts.append(self._memory_prompt(memory_hits))
                action_type = "retrieve_failure_memory"
            else:
                action_type = "memory_retrieval_disabled"
            trace.event(
                stage="memory",
                reason_summary="Retrieve local self-updated non-label memories by concept/column overlap, unless disabled for write-only control runs.",
                action={"type": action_type},
                observation={"hit_count": len(memory_hits), "retrieval_enabled": self.config.memory_retrieval},
                validation={"passed": True},
            )

        oracle_checks = []
        if self.enabled("verify") and self.config.semantic_oracles:
            oracle_checks = self.oracles.plan(contract, data_card)
            if oracle_checks:
                artifacts.oracle_plan_path = str(artifact_dir / "oracle_plan.json")
                write_json(artifacts.oracle_plan_path, self.oracles.manifest(oracle_checks))
                prompt_parts.append(self.oracles.to_prompt(oracle_checks))
            trace.event(
                stage="semantic_oracle_plan",
                reason_summary="Derive method-level checks from contract and Data Card for branch scoring.",
                action={"type": "plan_semantic_oracles"},
                observation={"oracle_ids": [check.id for check in oracle_checks]},
                validation={"passed": True},
            )

        if self.enabled("verify"):
            prompt_parts.append(self._trace_and_verification_prompt())
            prompt_parts.append(self.recovery.prompt_instructions())
            prompt_parts.append(self._final_block_prompt(question))

        context = HarnessContext(
            question=question,
            table_path=str(table_path),
            mode=self.config.mode,
            prompt="\n\n".join(prompt_parts),
            contract=contract,
            data_card=data_card,
            skills=[skill.name for skill in skills],
            memory_hits=memory_hits,
            rules=self.rulebook.manifest(rules),
            oracle_checks=self.oracles.manifest(oracle_checks),
            artifacts=artifacts,
        )
        return context

    def postprocess(
        self,
        context: HarnessContext,
        response_text: str,
        branch_index: int = 0,
        branch_name: str = "deterministic_skill_branch",
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = context.artifacts
        trace = TraceWriter(artifacts.trace_path if artifacts else None)
        final_block = extract_final_answer_block(response_text)
        recovered_from_legacy = False
        if final_block is None:
            final_block = final_block_from_legacy_targets(response_text, context.contract)
            recovered_from_legacy = final_block is not None
        final_block = normalize_final_block_to_contract(final_block, context.contract)
        validation = self.verifier.validate_final_block(
            final_block, context.contract, response_text=response_text
        )
        if recovered_from_legacy:
            validation.warnings.append("legacy_at_answer_recovered_as_final_block")
        if final_block and validation.report_id:
            final_block.validator_report_id = validation.report_id

        semantic_checks = [SemanticOracleCheck(**item) for item in (context.oracle_checks or [])]
        semantic_validation = self.oracles.validate(final_block, semantic_checks, response_text=response_text)

        if artifacts:
            artifacts.validator_report_path = str(
                Path(artifacts.root_dir) / f"validator_report_branch_{branch_index}.json"
            )
            write_json(artifacts.validator_report_path, validation)
            artifacts.semantic_report_path = str(
                Path(artifacts.root_dir) / f"semantic_report_branch_{branch_index}.json"
            )
            write_json(artifacts.semantic_report_path, semantic_validation)
            artifacts.final_block_path = str(Path(artifacts.root_dir) / f"final_block_branch_{branch_index}.json")
            if final_block:
                write_json(artifacts.final_block_path, final_block)

        recovery_events = []
        if not validation.passed:
            event = self.recovery.validator_failure_event(validation.errors)
            recovery_events.append(self.recovery.to_dict(event))
        if not semantic_validation.passed:
            event = self.recovery.validator_failure_event(semantic_validation.errors)
            recovery_events.append(self.recovery.to_dict(event))

        if self.enabled("memory") and self.config.memory_recording:
            self.memory.record_branch_outcome(
                context.contract,
                branch_name=branch_name,
                validation=validation,
                semantic_validation=semantic_validation,
                runtime=runtime or {},
                final_block=final_block,
            )

        trace.event(
            stage="finalization",
            skill="final_format",
            reason_summary="Validate the response FinalAnswerBlock before allowing deterministic reformat.",
            action={"type": "extract_and_validate_final_block", "branch": branch_name},
            observation={
                "final_block_present": final_block is not None,
                "recovered_from_legacy": recovered_from_legacy,
            },
            validation=to_plain(validation),
        )
        trace.event(
            stage="semantic_validation",
            reason_summary="Score branch method evidence against contract-derived semantic oracle checks.",
            action={"type": "validate_semantic_oracles", "branch": branch_name},
            observation={"oracle_count": len(semantic_checks)},
            validation=to_plain(semantic_validation),
        )

        branch_score = self.verifier.branch_score(
            validation,
            runtime_success=not ((runtime or {}).get("error")),
            branch_index=branch_index,
            semantic_validation=semantic_validation,
        )
        reformat_response = (
            render_reformat_response(final_block, context.question.get("format"))
            if final_block
            else ""
        )
        return {
            "branch_index": branch_index,
            "branch_name": branch_name,
            "branch_score": branch_score,
            "response": response_text,
            "final_block": to_plain(final_block) if final_block else {},
            "reformat_response": reformat_response,
            "validator_report": to_plain(validation),
            "semantic_report": to_plain(semantic_validation),
            "validator_report_path": artifacts.validator_report_path if artifacts else None,
            "semantic_report_path": artifacts.semantic_report_path if artifacts else None,
            "final_block_path": artifacts.final_block_path if artifacts else None,
            "recovery_events": recovery_events,
            "runtime": runtime or {},
        }

    def should_search_before_primary(self, context: HarnessContext) -> bool:
        return self.enabled("search") and self.search.pretrigger(context)

    def should_search_after_primary(self, validation: ValidationResult | dict[str, Any]) -> bool:
        if not self.enabled("search"):
            return False
        if isinstance(validation, dict):
            validation = ValidationResult(**validation)
        return self.search.posttrigger(validation)

    def branch_prompts(self, context: HarnessContext) -> list[dict[str, Any]]:
        branches = []
        for branch in self.search.branches(context):
            branches.append(
                {
                    "index": branch.index,
                    "name": branch.name,
                    "prompt": context.prompt + branch.prompt_suffix,
                    "strategy": branch.strategy,
                    "depth": branch.depth,
                    "parent_index": branch.parent_index,
                    "oracle_focus": branch.oracle_focus,
                }
            )
        return branches

    def select_branch(self, branch_results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        selected_pos, selected = self.search.select(branch_results)
        summary = self.search.summarize(branch_results, selected.get("branch_index", selected_pos))
        return selected, summary

    def _base_prompt(self, question: dict[str, Any]) -> str:
        file_name = question.get("file_name", "")
        return (
            f"Question: {question.get('question', '')}\n"
            f"{question.get('constraints', '')}\n\n"
            f"Expected output format: {question.get('format', '')}\n"
            f"{file_name} has been uploaded."
        )

    def _toolkit_prompt(self) -> str:
        return """# Optional Harness Tools
A lightweight helper module is uploaded as `./input/harness_tools.py`. You may use it for deterministic cross-checks:
```python
import sys
sys.path.append('./input')
from harness_tools import read_csv, summary_stats, grouped_mean, pearson_correlation, iqr_outliers, value_counts
rows = read_csv('./input/<uploaded_csv_name>')
```
Use these helpers only when they match the contract; pandas remains allowed.
"""

    def _trace_and_verification_prompt(self) -> str:
        return """# Structured ReAct Trace Requirement
For each meaningful step, keep a short auditable note with:
- reason_summary: one sentence, no hidden chain-of-thought.
- action: code/tool/dataframe operation performed.
- observation: concise output summary, shape/sample size/thresholds where relevant.
- validation: checks passed/failed.
Do not finalize until required columns, sample sizes, formulas/statistical methods, and format targets have been checked.
"""

    def _final_block_prompt(self, question: dict[str, Any]) -> str:
        qid = json.dumps(question.get("id", ""), ensure_ascii=False)
        return f"""# FinalAnswerBlock Requirement
When the task is complete, end with exactly one fenced JSON object using this schema and no extra @answer lines outside the JSON:
```json
{{
  "final_answer_block": {{
    "question_id": {qid},
    "answers": {{"answer_name": "computed_value"}},
    "format_targets": {{"answer_name": "@answer_name[computed_value]"}},
    "evidence": {{
      "source_columns": [],
      "sample_size": 0,
      "method": "short method description",
      "oracle_evidence": {{"field_name": "value checked when semantic oracle asks for it"}}
    }},
    "verified": true,
    "validator_report_id": null
  }}
}}
```
Use the exact answer names from the expected output format: {question.get('format', '')}
If a required field is unknown, continue verification instead of finalizing.
"""

    def _memory_prompt(self, hits: list[dict[str, Any]]) -> str:
        lines = [
            "# Retrieved Self-Updated Persistent Memories",
            "Use these as cautionary non-label repair hints from prior local agent/harness artifacts only; never use benchmark labels, human reports, transferred memories, or curated rule cards.",
        ]
        for hit in hits:
            scope = hit.get("scope_level", "episode")
            source = hit.get("source", "unknown")
            score = hit.get("retrieval_score", "")
            lines.append(
                f"- scope={scope}; source={source}; score={score}; signature={hit.get('signature')}: failure={hit.get('failure_type')}; repair={hit.get('repair')}"
            )
        return "\n".join(lines)


def build_runtime(start_time: float, **extra: Any) -> dict[str, Any]:
    runtime = {"tokens": 0, "llm_calls": 1, "seconds": round(time.time() - start_time, 3)}
    runtime.update(extra)
    return runtime
