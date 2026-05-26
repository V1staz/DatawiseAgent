from __future__ import annotations

from typing import Any

from .io_utils import compact_json


FINAL_BLOCK_INSTRUCTION = """
Before giving the final @answer output, write one machine-readable JSON block labeled FINAL_BLOCK.
The block must have this shape:
{
  "task_id": <task id>,
  "answers": [
    {
      "name": "<exact answer name>",
      "value": "<value as final string>",
      "raw_value": <raw value or null>,
      "format_checked": true,
      "source_variable": "<variable name>"
    }
  ],
  "verification": {"passed": true, "warnings": []}
}
Do not use placeholders. Do not let a later reformat step infer answers from the notebook log.
"""


def build_original_prompt(question_record: dict[str, Any]) -> str:
    return (
        f"Question: {question_record.get('question', '')}\n"
        f"{question_record.get('constraints', '')}\n\n"
        f"{question_record.get('file_name', '')} has been uploaded."
    )


def build_improved_prompt(
    question_record: dict[str, Any],
    *,
    schema_profile: dict[str, Any] | None = None,
    question_contract: dict[str, Any] | None = None,
    skill_plan: dict[str, Any] | None = None,
    include_schema: bool = True,
    include_contract: bool = True,
    include_skill_plan: bool = True,
    require_final_block: bool = True,
) -> str:
    parts = [
        "Question:",
        str(question_record.get("question", "")),
        "",
        "Constraints:",
        str(question_record.get("constraints", "")),
        "",
        "Required output format:",
        str(question_record.get("format", "")),
        "",
        f"Uploaded file: {question_record.get('file_name', '')}",
    ]

    if include_schema and schema_profile is not None:
        parts.extend(
            [
                "",
                "SCHEMA_PROFILE_JSON:",
                compact_json(schema_profile, max_chars=12000),
                "Use this schema profile instead of repeatedly exploring head/info/describe unless a contract check requires a small targeted inspection.",
            ]
        )

    if include_contract and question_contract is not None:
        parts.extend(
            [
                "",
                "QUESTION_CONTRACT_JSON:",
                compact_json(question_contract, max_chars=12000),
                "Follow the contract exactly. If the contract is ambiguous, use the original question and do not add unrequested cleaning, tuning, or features.",
            ]
        )

    if include_skill_plan and skill_plan is not None:
        parts.extend(
            [
                "",
                "SKILL_PLAN_JSON:",
                compact_json(skill_plan, max_chars=16000),
                "Use every enabled checklist and record verification evidence for the rules.",
            ]
        )

    if require_final_block:
        parts.extend(["", FINAL_BLOCK_INSTRUCTION.strip()])

    return "\n".join(parts)

