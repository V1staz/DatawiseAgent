"""Contract, step, and final-block validation helpers."""

from __future__ import annotations

from hashlib import sha1
from typing import Any
from datetime import date
import re

from .finalizer import build_format_targets, parse_reformat_targets
from .schemas import DataCard, FinalAnswerBlock, QuestionContract, ValidationResult, to_plain


class Verifier:
    """Lightweight validator that blocks obvious finalization/format failures."""

    def validate_contract(
        self, contract: QuestionContract | None, data_card: DataCard | None = None
    ) -> ValidationResult:
        checks: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        if contract is None:
            return ValidationResult(False, errors=["contract_missing"], score=0.0)

        checks.append("contract_present")
        if contract.target_answers:
            checks.append("target_answers_present")
        else:
            errors.append("target_answers_missing")

        if data_card is not None and contract.required_columns:
            available = {col.name for col in data_card.columns}
            missing = [col for col in contract.required_columns if col not in available]
            if missing:
                errors.append("required_columns_missing:" + ",".join(missing))
            else:
                checks.append("required_columns_available")
        elif data_card is not None:
            warnings.append("required_columns_not_inferred")

        if contract.final_format:
            checks.append("final_format_present")
        else:
            errors.append("final_format_missing")

        passed = not errors
        return ValidationResult(
            passed=passed,
            checks=checks,
            errors=errors,
            warnings=warnings,
            score=self._score(checks, errors, warnings),
        )

    def validate_final_block(
        self,
        final_block: FinalAnswerBlock | None,
        contract: QuestionContract | None,
        response_text: str | None = None,
    ) -> ValidationResult:
        checks: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        if final_block is None:
            errors.append("final_block_missing")
            if response_text and parse_reformat_targets(response_text):
                warnings.append("legacy_at_answer_found_but_final_block_missing")
            return ValidationResult(False, checks, errors, warnings, score=0.0)

        checks.append("final_block_present")
        if final_block.answers:
            checks.append("answers_present")
        else:
            errors.append("answers_missing")

        if contract is not None:
            required = [target.name for target in contract.target_answers]
            missing_answers = [name for name in required if name not in final_block.answers]
            if missing_answers:
                errors.append("answer_names_missing:" + ",".join(missing_answers))
            else:
                checks.append("answer_names_covered")

            missing_formats = [name for name in required if name not in final_block.format_targets]
            if missing_formats:
                errors.append("format_targets_missing:" + ",".join(missing_formats))
            else:
                checks.append("format_targets_covered")

            for target in contract.target_answers:
                value = final_block.answers.get(target.name)
                if value is not None and target.type != "unknown":
                    if not self._type_matches(value, target.type):
                        warnings.append(f"answer_type_mismatch:{target.name}:{target.type}")
                    else:
                        checks.append(f"answer_type_ok:{target.name}")
                range_error = self._range_error(value, target)
                if range_error:
                    errors.append(range_error)

            for name, target_text in final_block.format_targets.items():
                if not re.fullmatch(rf"@{re.escape(name)}\[[\s\S]*\]", str(target_text)):
                    errors.append(f"format_target_invalid:{name}")
                else:
                    checks.append(f"format_target_syntax_ok:{name}")

            expected_formats = build_format_targets(final_block.answers, contract.final_format)
            target_by_name = {target.name: target for target in contract.target_answers}
            for name in required:
                expected = expected_formats.get(name)
                actual = final_block.format_targets.get(name)
                if expected is None or actual is None:
                    continue
                target = target_by_name.get(name)
                if not self._format_target_matches_expected(
                    name=name,
                    actual=str(actual),
                    expected=expected,
                    rounding=target.rounding if target else None,
                ):
                    errors.append(f"format_target_answer_mismatch:{name}")
                else:
                    checks.append(f"format_target_matches_answer:{name}")

        if final_block.verified:
            checks.append("verified_true")
        else:
            warnings.append("verified_false_or_missing")

        report_hash = sha1(repr(to_plain(final_block)).encode("utf-8")).hexdigest()[:12]
        passed = not errors
        return ValidationResult(
            passed=passed,
            checks=checks,
            errors=errors,
            warnings=warnings,
            score=self._score(checks, errors, warnings),
            report_id=f"validator-{report_hash}",
        )

    def branch_score(
        self,
        validation: ValidationResult,
        runtime_success: bool = True,
        branch_index: int = 0,
        semantic_validation: ValidationResult | None = None,
    ) -> float:
        # Format/schema validation remains the main gate, but semantic oracle
        # evidence now contributes to branch selection so hard tasks are not
        # selected solely because their JSON shape is valid.
        score = validation.score * 0.72
        if semantic_validation is not None:
            score += semantic_validation.score * 0.20
            if semantic_validation.errors:
                score -= 0.12
        else:
            score += 0.08
        if runtime_success:
            score += 0.12
        score -= 0.02 * max(branch_index, 0)
        return round(min(max(score, 0.0), 1.0), 4)

    def _score(self, checks: list[str], errors: list[str], warnings: list[str]) -> float:
        base = len(checks) / max(len(checks) + len(errors) + len(warnings), 1)
        if errors:
            base *= 0.45
        return round(base, 4)

    def _format_target_matches_expected(
        self,
        name: str,
        actual: str,
        expected: str,
        rounding: int | None = None,
    ) -> bool:
        if actual.strip() == expected:
            return True
        actual_value = parse_reformat_targets(actual).get(name)
        expected_value = parse_reformat_targets(expected).get(name)
        if actual_value is None or expected_value is None:
            return False
        try:
            actual_float = float(str(actual_value).replace(",", "").strip().rstrip("%"))
            expected_float = float(str(expected_value).replace(",", "").strip().rstrip("%"))
        except ValueError:
            return str(actual_value).strip() == str(expected_value).strip()
        if rounding is not None:
            return round(actual_float, rounding) == round(expected_float, rounding)
        return abs(actual_float - expected_float) < 1e-12

    def _type_matches(self, value: Any, expected: str) -> bool:
        if expected == "integer":
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                return True
            try:
                return float(str(value)).is_integer()
            except Exception:
                return False
        if expected == "float":
            if isinstance(value, bool):
                return False
            try:
                float(str(value))
                return True
            except Exception:
                return False
        if expected == "list":
            return isinstance(value, list)
        if expected == "dict":
            return isinstance(value, dict)
        if expected == "string":
            return isinstance(value, str)
        if expected == "boolean":
            if isinstance(value, bool):
                return True
            if isinstance(value, str):
                return value.strip().lower() in {"true", "false"}
            return False
        return True

    def _range_error(self, value: Any, target: Any) -> str | None:
        description = " ".join(
            str(part or "")
            for part in [
                getattr(target, "description", ""),
                getattr(target, "format", ""),
            ]
        ).lower()
        if "from 1900 to the current year" not in description:
            return None
        identity = " ".join(
            str(part or "")
            for part in [getattr(target, "name", ""), getattr(target, "format", "")]
        ).lower()
        if not re.search(r"(?:vehicle|model)[_-]?year|modelyear|year_model", identity):
            return None
        try:
            numeric_value = float(str(value).replace(",", "").strip().rstrip("%"))
        except Exception:
            return None
        current_year = date.today().year
        if numeric_value < 1900 or numeric_value > current_year:
            return f"answer_range_violation:{target.name}:1900-{current_year}"
        return None

    def report_dict(self, result: ValidationResult) -> dict[str, Any]:
        return to_plain(result)
