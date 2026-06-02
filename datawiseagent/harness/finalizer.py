"""FinalAnswerBlock extraction and deterministic reformat rendering."""

from __future__ import annotations

from typing import Any
from datetime import date
import json
import re

from .contract import extract_format_targets
from .schemas import FinalAnswerBlock, QuestionContract

_FINAL_BLOCK_MARKER_RE = re.compile(
    r"(?:FINAL_ANSWER_BLOCK|FinalAnswerBlock|final_answer_block)\s*[:=]?\s*(\{.*?\})",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_FORMAT_RE = re.compile(r"@(\w+)\[([^\]]*)\]")


def _json_loads_maybe(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    if isinstance(value, dict):
        return value
    return None


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in starts:
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    # Prefer larger objects because the final block contains nested answers/evidence.
    return sorted(set(candidates), key=len, reverse=True)[:20]


def _coerce_final_block(payload: dict[str, Any]) -> FinalAnswerBlock | None:
    if "final_answer_block" in payload and isinstance(payload["final_answer_block"], dict):
        payload = payload["final_answer_block"]
    if "FinalAnswerBlock" in payload and isinstance(payload["FinalAnswerBlock"], dict):
        payload = payload["FinalAnswerBlock"]
    if "answers" not in payload or not isinstance(payload.get("answers"), dict):
        return None
    return FinalAnswerBlock(
        question_id=payload.get("question_id", payload.get("id", "")),
        answers=dict(payload.get("answers", {})),
        format_targets=dict(payload.get("format_targets", {})),
        evidence=dict(payload.get("evidence", {})) if isinstance(payload.get("evidence"), dict) else {},
        verified=bool(payload.get("verified", False)),
        validator_report_id=payload.get("validator_report_id"),
    )


def extract_final_answer_block(response: str | dict[str, Any] | None) -> FinalAnswerBlock | None:
    """Extract a machine-readable final block from a response or result record."""
    if response is None:
        return None
    if isinstance(response, dict):
        block = response.get("final_block") or response.get("final_answer_block")
        if isinstance(block, dict):
            return _coerce_final_block(block)
        response = str(response.get("response", ""))

    text = str(response)
    for regex in (_FENCED_JSON_RE, _FINAL_BLOCK_MARKER_RE):
        for match in regex.finditer(text):
            payload = _json_loads_maybe(match.group(1))
            if payload:
                block = _coerce_final_block(payload)
                if block is not None:
                    return block

    for candidate in _balanced_json_candidates(text):
        payload = _json_loads_maybe(candidate)
        if payload:
            block = _coerce_final_block(payload)
            if block is not None:
                return block
    return None


def final_block_from_legacy_targets(
    response_text: str | None,
    contract: QuestionContract | None,
) -> FinalAnswerBlock | None:
    """Recover a FinalAnswerBlock from legacy @answer[value] output.

    This is a compatibility bridge for agents that correctly compute and print
    benchmark-format answers but ignore the machine-readable JSON instruction.
    The recovered block remains auditable through `verified=False` plus
    controller-added validator warnings.
    """

    if not response_text or contract is None or not contract.target_answers:
        return None
    selected_targets = {
        target.name: _select_legacy_target_value(response_text, target)
        for target in contract.target_answers
    }
    required_names = [target.name for target in contract.target_answers]
    if any(selected_targets.get(name) is None for name in required_names):
        return None

    answers = {
        target.name: _coerce_answer_value(str(selected_targets[target.name]), target.type)
        for target in contract.target_answers
    }
    return FinalAnswerBlock(
        question_id=contract.question_id,
        answers=answers,
        format_targets={
            name: f"@{name}[{selected_targets[name]}]" for name in required_names
        },
        evidence={
            "method": "Recovered from legacy @answer[value] lines in agent response.",
            "recovered_from": "legacy_at_answer",
        },
        verified=False,
    )


def _select_legacy_target_value(response_text: str, target: Any) -> str | None:
    matches = [
        value
        for name, value in _FORMAT_RE.findall(response_text)
        if name == target.name
    ]
    for value in reversed(matches):
        if not _looks_like_placeholder(value, target):
            return value
    return matches[-1] if matches else None


def _looks_like_placeholder(value: str, target: Any) -> bool:
    text = str(value).strip().strip("\"'")
    placeholders = {
        parsed
        for parsed in parse_reformat_targets(getattr(target, "format", "")).values()
    }
    placeholders.add(getattr(target, "name", ""))
    if text in placeholders:
        return True
    if getattr(target, "type", "") in {"integer", "float"}:
        return not bool(re.search(r"\d", text))
    return False


def _coerce_answer_value(value: str, expected_type: str) -> Any:
    text = str(value).strip()
    if expected_type == "boolean":
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return text
    if expected_type == "integer":
        try:
            return int(float(text.replace(",", "")))
        except ValueError:
            return text
    if expected_type == "float":
        try:
            return float(text.replace(",", "").rstrip("%"))
        except ValueError:
            return text
    return text


def build_format_targets(answers: dict[str, Any], format_text: str | None) -> dict[str, str]:
    targets = extract_format_targets(format_text)
    if not targets:
        return {
            name: f"@{name}[{_stringify_answer(value)}]"
            for name, value in answers.items()
        }
    rendered: dict[str, str] = {}
    for target in targets:
        value = answers.get(target.name)
        if value is None:
            continue
        rendered[target.name] = (
            f"@{target.name}[{_stringify_for_target(value, target, format_text)}]"
        )
    return rendered


def _stringify_answer(value: Any, rounding: int | None = None) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if rounding is not None:
            return f"{float(value):.{rounding}f}"
        return f"{value:g}"
    if isinstance(value, str) and rounding is not None:
        try:
            return f"{float(value.replace(',', '').strip().rstrip('%')):.{rounding}f}"
        except ValueError:
            return value
    if isinstance(value, list):
        if not value:
            # The benchmark's regex extractor cannot parse nested brackets; an
            # empty list is nevertheless represented as `[]`, which extracts as
            # `[` and matches the existing empty-list labels.
            return "[]"
        return ", ".join(_stringify_answer(item, rounding) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _stringify_for_target(value: Any, target: Any, format_text: str | None) -> str:
    if getattr(target, "type", "") == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
    if isinstance(value, bool):
        return _stringify_boolean_for_target(value, target, format_text)

    component_rendered = _stringify_components_if_applicable(value, target, format_text)
    if component_rendered is not None:
        return component_rendered

    rendered = _stringify_answer(value, getattr(target, "rounding", None))
    if _target_wants_quoted_string(target) and _is_unquoted_scalar_string(value, rendered):
        return json.dumps(str(value).strip().strip("\"'"), ensure_ascii=False)
    return rendered


def _stringify_boolean_for_target(value: bool, target: Any, format_text: str | None) -> str:
    text = " ".join(
        str(part or "")
        for part in [
            getattr(target, "description", ""),
            getattr(target, "format", ""),
            format_text or "",
        ]
    )
    has_title = bool(re.search(r"\bTrue\b|\bFalse\b", text))
    has_lower = bool(re.search(r"\btrue\b|\bfalse\b", text))
    if has_lower and not has_title:
        return str(value).lower()
    return str(value)


def _is_unquoted_scalar_string(value: Any, rendered: str) -> bool:
    if not isinstance(value, str):
        return False
    text = rendered.strip()
    if not text or "," in text:
        return False
    return not (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    )


def _target_wants_quoted_string(target: Any) -> bool:
    target_format = str(getattr(target, "format", "") or "")
    match = re.search(r"@\w+\[([^\]]*)\]", target_format)
    if not match:
        return False
    placeholder = match.group(1).strip()
    return bool(
        (placeholder.startswith('"') and '"' in placeholder[1:])
        or (placeholder.startswith("'") and "'" in placeholder[1:])
    )


def _stringify_components_if_applicable(
    value: Any,
    target: Any,
    format_text: str | None,
) -> str | None:
    component_names = _target_component_names(target)
    if len(component_names) < 2:
        return None

    values = _target_component_values(value)
    if len(values) != len(component_names):
        return None

    roundings = [
        _rounding_for_identifier(format_text or "", component)
        for component in component_names
    ]
    if not any(rounding is not None for rounding in roundings) and not _has_outer_sequence_delimiters(value):
        return None

    rendered: list[str] = []
    for item, rounding in zip(values, roundings):
        if rounding is not None and _is_numeric_text(item):
            rendered.append(f"{float(str(item).replace(',', '').strip()):.{rounding}f}")
        else:
            rendered.append(str(item).strip())
    return ", ".join(rendered)


def _target_component_names(target: Any) -> list[str]:
    target_format = str(getattr(target, "format", "") or "")
    match = re.search(r"@\w+\[([^\]]*)\]", target_format)
    if not match:
        return []
    return [part.strip().strip("\"'") for part in match.group(1).split(",") if part.strip()]


def _target_component_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return []
    text = value.strip()
    if _has_outer_sequence_delimiters(text):
        text = text[1:-1].strip()
    if "," not in text:
        return []
    return [part.strip().strip("\"'") for part in text.split(",")]


def _has_outer_sequence_delimiters(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return len(text) >= 2 and (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
    )


def _is_numeric_text(value: Any) -> bool:
    try:
        float(str(value).replace(",", "").strip().rstrip("%"))
        return True
    except Exception:
        return False


def _rounding_for_identifier(text: str, identifier: str) -> int | None:
    identifier = str(identifier or "").strip().strip("\"'")
    if not text or not identifier:
        return None
    snippets = _identifier_description_snippets(text, identifier)
    for snippet in snippets:
        value = _extract_rounding(snippet)
        if value is not None:
            return value
    return None


def _identifier_description_snippets(text: str, identifier: str) -> list[str]:
    wanted = identifier.lower()
    snippets: list[str] = []
    seen: set[str] = set()
    clauses = [part.strip() for part in re.split(r"[\n;]+", text) if part.strip()]
    sentence_boundary = re.compile(
        r"(?<=[.!?])\s+(?=(?:where\s+)?['\"]|(?:where\s+)?\w+\s+(?:is|are)\b|@)",
        flags=re.IGNORECASE,
    )
    split_clauses: list[str] = []
    for clause in clauses:
        split_clauses.extend(piece.strip() for piece in sentence_boundary.split(clause) if piece.strip())

    for clause in split_clauses:
        for match in re.finditer(
            r"(?P<ids>(?:['\"][^'\"]+['\"]\s*(?:,\s*and\s*|,\s*|\s+and\s+)?\s*){1,8})\s+"
            r"(?P<verb>is|are)\s+(?:an?|the)?\s*(?P<desc>[^;\n]{0,260})",
            clause,
            flags=re.IGNORECASE,
        ):
            ids = {item.strip().strip("\"'").lower() for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group("ids"))}
            if wanted in ids:
                desc = _trim_adjacent_target_phrase(match.group("desc"))
                snippet = f"{match.group('ids').strip()} {match.group('verb')} {desc.strip()}"
                if snippet not in seen:
                    seen.add(snippet)
                    snippets.append(snippet)

        escaped = re.escape(identifier)
        for pattern in (
            rf"['\"]{escaped}['\"]\s+(?:is|are)\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
            rf"\b{escaped}\b\s+(?:is|are)\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
            rf"['\"]{escaped}['\"]\s+should\s+be\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
            rf"\b{escaped}\b\s+should\s+be\s+(?:an?|the)?\s*([^;\n]{{0,260}})",
        ):
            match = re.search(pattern, clause, flags=re.IGNORECASE)
            if match:
                full = match.group(0)
                desc = _trim_adjacent_target_phrase(match.group(1))
                snippet = (full[: full.find(match.group(1))] + desc).strip()
                if snippet not in seen:
                    seen.add(snippet)
                    snippets.append(snippet)
    return snippets


def _trim_adjacent_target_phrase(text: str) -> str:
    return re.split(
        r"(?:\s+(?:and|while)\s+|\s*,\s*)(?:['\"][^'\"]+['\"]|\w+)\s+(?:is|are)\s+",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]


def _extract_rounding(text: str) -> int | None:
    lowered = text.lower()
    patterns = (
        r"round(?:ed|ing)?(?:\s+off)?(?:\s+the\s+answer)?\s+to\s+(\d+)\s+decimal",
        r"rounded\s+to\s+(\d+)\s+decimal",
        r"to\s+(\d+)\s+decimal\s+places",
        r"with\s+(\d+)\s+decimals?",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    word_to_int = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"(?:to|with)\s+(one|two|three|four|five|six)\s+decimals?", lowered)
    if match:
        return word_to_int[match.group(1)]
    return None


def render_reformat_response(
    final_block: FinalAnswerBlock | dict[str, Any],
    format_text: str | None = None,
    separator: str = "\n",
) -> str:
    """Render @answer[value] lines from FinalAnswerBlock answers.

    The rendered evaluation text is intentionally derived from `answers`, not
    from model-supplied `format_targets`, whenever the benchmark format is
    available. This prevents a contradictory final block from validating one
    value in `answers` while evaluating another value in `format_targets`.
    """
    if isinstance(final_block, dict):
        block = _coerce_final_block(final_block)
        if block is None:
            return ""
    else:
        block = final_block

    targets = build_format_targets(block.answers, format_text)
    if not format_text and not targets:
        targets = dict(block.format_targets or {})

    required_order = [target.name for target in extract_format_targets(format_text)]
    if required_order:
        ordered = [targets[name] for name in required_order if name in targets]
        ordered.extend(
            value for name, value in targets.items() if name not in set(required_order)
        )
    else:
        ordered = list(targets.values())
    return separator.join(ordered)


def normalize_final_block_to_contract(
    final_block: FinalAnswerBlock | None,
    contract: QuestionContract | None,
) -> FinalAnswerBlock | None:
    """Apply deterministic contract-level value normalizations before validation."""

    if final_block is None or contract is None:
        return final_block
    changed: dict[str, Any] = {}
    for target in contract.target_answers:
        if target.name not in final_block.answers:
            continue
        normalized = _normalize_boolean(final_block.answers[target.name], target)
        if normalized is None:
            normalized = _normalize_model_year(final_block.answers[target.name], target)
        if normalized is not None and normalized != final_block.answers[target.name]:
            final_block.answers[target.name] = normalized
            final_block.format_targets[target.name] = f"@{target.name}[{_stringify_for_target(normalized, target, contract.final_format)}]"
            changed[target.name] = normalized
    if changed:
        final_block.evidence = dict(final_block.evidence or {})
        final_block.evidence.setdefault("normalizations", {}).update(
            {
                name: {
                    "rule": _normalization_rule_name(value),
                    "normalized_value": value,
                }
                for name, value in changed.items()
            }
        )
    return final_block


def _normalization_rule_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean_string_to_lowercase_bool"
    return "two_digit_model_year_to_full_year"


def _normalize_boolean(value: Any, target: Any) -> bool | None:
    if getattr(target, "type", "") != "boolean" or not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _normalize_model_year(value: Any, target: Any) -> int | None:
    description = " ".join(
        str(part or "")
        for part in [
            getattr(target, "description", ""),
            getattr(target, "format", ""),
        ]
    ).lower()
    identity = " ".join(
        str(part or "")
        for part in [getattr(target, "name", ""), getattr(target, "format", "")]
    ).lower()
    if "from 1900 to the current year" not in description or not re.search(
        r"(?:vehicle|model)[_-]?year|modelyear|year_model",
        identity,
    ):
        return None
    try:
        numeric = int(float(str(value).replace(",", "").strip()))
    except Exception:
        return None
    if 0 <= numeric < 100:
        full_year = 1900 + numeric if numeric >= 30 else 2000 + numeric
        if 1900 <= full_year <= date.today().year:
            return full_year
    return None


def parse_reformat_targets(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return {name: value for name, value in _FORMAT_RE.findall(text)}
