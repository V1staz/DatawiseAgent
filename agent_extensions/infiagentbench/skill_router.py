from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .skills import SKILL_REGISTRY, TASK_TYPE_TO_SKILL


def route_skills(question_contract: dict[str, Any]) -> dict[str, Any]:
    task_types = list(question_contract.get("task_types", []))
    if "final_formatting" not in task_types:
        task_types.append("final_formatting")

    enabled_skills: list[dict[str, Any]] = []
    merged_rules: list[str] = []
    for task_type in task_types:
        skill_name = TASK_TYPE_TO_SKILL.get(task_type)
        if not skill_name:
            continue
        skill = SKILL_REGISTRY[skill_name]
        enabled_skills.append(
            {
                "name": skill_name,
                "task_type": task_type,
                "checklist": skill["checklist"],
                "verification_rules": skill["verification_rules"],
            }
        )
        for rule in skill["verification_rules"]:
            if rule not in merged_rules:
                merged_rules.append(rule)

    return {
        "task_id": question_contract.get("task_id"),
        "task_types": task_types,
        "enabled_skills": enabled_skills,
        "verification_rules": merged_rules,
        "routing_reason": "Rule-based routing from question_contract.task_types.",
    }


def write_skill_plan(question_contract: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    skill_plan = route_skills(question_contract)
    write_json(skill_plan, output_path)
    return skill_plan


def _main() -> None:
    parser = argparse.ArgumentParser(description="Route an InfiAgentBench question to skill checklists.")
    parser.add_argument("--contract", required=True, help="question_contract.json path.")
    parser.add_argument("--output", default="skill_plan.json")
    args = parser.parse_args()
    contract = read_json(args.contract)
    write_skill_plan(contract, args.output)


if __name__ == "__main__":
    _main()

