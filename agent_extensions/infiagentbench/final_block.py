from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


PLACEHOLDER_RE = re.compile(
    r"^\s*(?:todo|tbd|none|null|nan|n/a|value_as_string|answer|r_value|p_value|significance|placeholder|\.\.\.)\s*$",
    re.I,
)


def _raw_decode_at(text: str, start: int) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "answers" in obj:
        return obj
    return None


def extract_final_block(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None

    marker_matches = list(re.finditer(r"FINAL_BLOCK|final block|final_block", text, re.I))
    candidate_starts: list[int] = []
    for marker in marker_matches:
        brace = text.find("{", marker.end())
        if brace >= 0:
            candidate_starts.append(brace)
    candidate_starts.extend(match.start() for match in re.finditer(r"\{", text))

    seen = set()
    for start in candidate_starts:
        if start in seen:
            continue
        seen.add(start)
        block = _raw_decode_at(text, start)
        if block is not None:
            return block
    return None


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_numeric_string(value: str) -> bool:
    value = _strip_outer_quotes(value.strip())
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value))


def _decimal_places(value: str) -> int | None:
    value = _strip_outer_quotes(value.strip())
    if not _is_numeric_string(value):
        return None
    if "e" in value.lower():
        return None
    if "." not in value:
        return 0
    return len(value.split(".", 1)[1])


def _expected_decimal_places(contract: dict[str, Any] | None) -> int | None:
    if not contract:
        return None
    fmt = contract.get("format_requirements") or contract.get("output_format") or {}
    return (
        fmt.get("decimal_places")
        if fmt.get("decimal_places") is not None
        else contract.get("decimal_places")
    )


def _expected_names(contract: dict[str, Any] | None) -> list[str]:
    if not contract:
        return []
    return list(contract.get("answer_names") or (contract.get("format_requirements") or {}).get("answer_names") or [])


def validate_final_block(
    final_block: dict[str, Any] | None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    answer_map: dict[str, str] = {}

    if not isinstance(final_block, dict):
        return {
            "passed": False,
            "errors": ["final_block_missing"],
            "warnings": [],
            "answer_map": {},
        }

    answers = final_block.get("answers")
    if not isinstance(answers, list) or not answers:
        errors.append("answers_missing_or_empty")
        answers = []

    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            errors.append(f"answers[{index}]_not_object")
            continue
        name = answer.get("name")
        value = answer.get("value")
        if not name:
            errors.append(f"answers[{index}]_missing_name")
            continue
        if value is None:
            errors.append(f"{name}:missing_value")
            continue
        value_text = str(value)
        answer_map[str(name)] = value_text
        if PLACEHOLDER_RE.fullmatch(value_text):
            errors.append(f"{name}:placeholder_value")
        if answer.get("format_checked") is False or answer.get("verified") is False:
            errors.append(f"{name}:format_not_checked")
        elif "format_checked" not in answer and "verified" not in answer:
            warnings.append(f"{name}:format_checked_flag_missing")

    expected_names = _expected_names(contract)
    for expected_name in expected_names:
        if expected_name not in answer_map:
            errors.append(f"{expected_name}:missing_expected_answer")
    extra_names = [name for name in answer_map if expected_names and name not in expected_names]
    if extra_names:
        warnings.append(f"extra_answers:{','.join(extra_names)}")

    expected_decimals = _expected_decimal_places(contract)
    if expected_decimals is not None:
        for name, value in answer_map.items():
            actual_decimals = _decimal_places(value)
            if actual_decimals is not None and actual_decimals != expected_decimals:
                errors.append(f"{name}:decimal_places_{actual_decimals}_expected_{expected_decimals}")

    format_requirements = (contract or {}).get("format_requirements") or (contract or {}).get("output_format") or {}
    quote_string = format_requirements.get("quote_string")
    container_type = format_requirements.get("container_type")
    for name, value in answer_map.items():
        stripped = value.strip()
        if quote_string is True and not (len(stripped) >= 2 and stripped[0] == stripped[-1] == '"'):
            errors.append(f"{name}:string_not_double_quoted")
        if container_type in {"list", "dict"}:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                errors.append(f"{name}:{container_type}_literal_invalid")
                continue
            if container_type == "list" and not isinstance(parsed, list):
                errors.append(f"{name}:list_literal_expected")
            if container_type == "dict" and not isinstance(parsed, dict):
                errors.append(f"{name}:dict_literal_expected")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "answer_map": answer_map,
    }


def final_block_to_response(
    final_block: dict[str, Any] | None,
    contract: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    validation = validate_final_block(final_block, contract)
    if not validation["passed"]:
        return "ERROR_NEED_RETRY", validation

    expected_names = _expected_names(contract)
    answer_map = validation["answer_map"]
    names = expected_names or list(answer_map.keys())
    response = "\n".join(f"@{name}[{answer_map[name]}]" for name in names if name in answer_map)
    if not response:
        validation["passed"] = False
        validation["errors"].append("empty_formatted_response")
        return "ERROR_NEED_RETRY", validation
    return response, validation


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract and validate an InfiAgentBench final block.")
    parser.add_argument("--response-text", help="Raw response text. If omitted, --response-file is required.")
    parser.add_argument("--response-file", help="Path to a text file containing the raw response.")
    parser.add_argument("--contract", help="Optional question_contract.json path.")
    parser.add_argument("--output", default="final_block.json")
    args = parser.parse_args()

    if args.response_text is not None:
        text = args.response_text
    elif args.response_file:
        text = Path(args.response_file).read_text(encoding="utf-8")
    else:
        raise SystemExit("Provide --response-text or --response-file.")

    contract = read_json(args.contract) if args.contract else None
    block = extract_final_block(text)
    response, validation = final_block_to_response(block, contract)
    write_json({"final_block": block, "formatted_response": response, "validation": validation}, args.output)


if __name__ == "__main__":
    _main()

