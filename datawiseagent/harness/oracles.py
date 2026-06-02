"""Semantic oracle planning for harness branch verification.

These oracles do not read benchmark labels.  They derive method-level checks from
question contracts and data cards, then ask the agent to expose enough evidence
for deterministic branch scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from .schemas import DataCard, FinalAnswerBlock, QuestionContract, ValidationResult, to_plain


@dataclass(slots=True)
class SemanticOracleCheck:
    id: str
    title: str
    severity: str = "medium"
    expected: str = ""
    rationale: str = ""
    evidence_fields: list[str] = field(default_factory=list)
    prompt: str = ""


def _combined_text(contract: QuestionContract | None) -> str:
    if contract is None:
        return ""
    parts = [contract.question or "", contract.raw_constraints or "", contract.final_format or ""]
    parts.extend(contract.filters or [])
    for target in contract.target_answers:
        parts.extend([target.name, target.format or "", target.description or ""])
    return "\n".join(parts).lower()


class SemanticOracleRegistry:
    """Plan and score method-level semantic checks."""

    def plan(self, contract: QuestionContract | None, data_card: DataCard | None = None) -> list[SemanticOracleCheck]:
        if contract is None:
            return []
        text = _combined_text(contract)
        concepts = set(contract.concepts or [])
        methods = set(contract.methods or [])
        checks: list[SemanticOracleCheck] = []

        self._add_std_checks(checks, text, methods)
        self._add_year_filter_checks(checks, text)
        self._add_ml_checks(checks, text, concepts)
        self._add_data_checks(checks, contract, data_card)
        return _dedupe(checks)

    def _add_std_checks(self, checks: list[SemanticOracleCheck], text: str, methods: set[str]) -> None:
        if "std" in methods or "standard deviation" in text or "z-score" in text or "z score" in text:
            if "population standard deviation" in text or "pstdev" in text or "ddof=0" in text:
                checks.append(
                    SemanticOracleCheck(
                        id="oracle.stats.std_population_ddof0",
                        title="Population standard deviation policy",
                        severity="hard",
                        expected="Use population standard deviation / ddof=0 because the question explicitly requested population std.",
                        evidence_fields=["method", "ddof", "statistical_definition"],
                        prompt="If standard deviation is used, write evidence.ddof=0 or mention population standard deviation explicitly.",
                    )
                )
            else:
                checks.append(
                    SemanticOracleCheck(
                        id="oracle.stats.std_sample_ddof1",
                        title="Sample standard deviation default",
                        severity="hard",
                        expected="Use sample standard deviation / ddof=1 unless population is explicit.",
                        rationale="InfiAgentBench tasks generally label unspecified std as sample std; this is a method convention, not a label lookup.",
                        evidence_fields=["method", "ddof", "statistical_definition"],
                        prompt="If standard deviation is used, write evidence.ddof=1 or mention sample standard deviation explicitly.",
                    )
                )

    def _add_year_filter_checks(self, checks: list[SemanticOracleCheck], text: str) -> None:
        if "2000 and beyond" in text or "years 2000 and beyond" in text:
            checks.append(
                SemanticOracleCheck(
                    id="oracle.filter.year_2000_inclusive",
                    title="Inclusive year boundary",
                    severity="hard",
                    expected="The filter for '2000 and beyond' must include year 2000, i.e. Year >= 2000.",
                    evidence_fields=["filters", "boundary", "sample_size"],
                    prompt="For '2000 and beyond', record the exact filter as >= 2000 and the resulting sample size.",
                )
            )
        elif re.search(r"\b(?:after|beyond)\s+2000\b", text):
            checks.append(
                SemanticOracleCheck(
                    id="oracle.filter.year_2000_exclusive_unless_constraint",
                    title="Exclusive year boundary when no inclusive wording exists",
                    severity="medium",
                    expected="Use > 2000 only when no explicit inclusive constraint such as '2000 and beyond' is present.",
                    evidence_fields=["filters", "boundary", "sample_size"],
                    prompt="Record whether the year boundary is > 2000 or >= 2000 and cite the exact wording that justifies it.",
                )
            )

    def _add_ml_checks(self, checks: list[SemanticOracleCheck], text: str, concepts: set[str]) -> None:
        ml_like = "Machine Learning" in concepts or any(word in text for word in ["train", "model", "predict", "classif", "regression"])
        feature_like = "Feature Engineering" in concepts or any(word in text for word in ["new feature", "engineered feature", "create a new column"])
        if ml_like:
            checks.append(
                SemanticOracleCheck(
                    id="oracle.ml.split_feature_metric_trace",
                    title="ML split/features/metric trace",
                    severity="hard",
                    expected="Expose train/test split, random_state, target, feature columns, and metric direction.",
                    evidence_fields=["feature_columns", "target", "split", "random_state", "metric"],
                    prompt="In FinalAnswerBlock.evidence include feature_columns, target, split/random_state, and metric used.",
                )
            )
        if ml_like and feature_like and any(phrase in text for phrase in ["improve prediction", "improves prediction", "new feature", "engineered feature"]):
            checks.append(
                SemanticOracleCheck(
                    id="oracle.ml.engineered_feature_baseline_all_numeric",
                    title="Engineered feature comparison baseline",
                    severity="hard",
                    expected="Compare all existing non-target numeric predictors versus the same baseline plus the engineered feature, unless a restricted feature set is explicit.",
                    evidence_fields=["baseline_features", "augmented_features", "engineered_feature", "same_split"],
                    prompt="For engineered-feature comparison, include evidence.baseline_features and evidence.augmented_features; the baseline should be all non-target numeric predictors unless restricted.",
                )
            )
        if ml_like and any(phrase in text for phrase in ["target label", "derived from", "classify", "predict based on"]):
            checks.append(
                SemanticOracleCheck(
                    id="oracle.ml.target_derived_leakage_policy",
                    title="Target-derived source feature leakage policy",
                    severity="hard",
                    expected="Include a target-derived source feature only when explicitly asked to predict based on that feature or no alternative predictors exist; otherwise exclude it as leakage.",
                    evidence_fields=["feature_columns", "excluded_features", "leakage_check"],
                    prompt="If the target is derived from a source column, explain whether that source column is included or excluded and why.",
                )
            )

    def _add_data_checks(
        self,
        checks: list[SemanticOracleCheck],
        contract: QuestionContract,
        data_card: DataCard | None,
    ) -> None:
        if data_card is not None and contract.required_columns:
            checks.append(
                SemanticOracleCheck(
                    id="oracle.data.required_columns_available",
                    title="Required column availability",
                    severity="medium",
                    expected="All required columns named by the contract should be checked against the Data Card before computation.",
                    evidence_fields=["source_columns"],
                    prompt="Before finalization, verify required source columns against the Data Card and include them in evidence.source_columns.",
                )
            )

    def to_prompt(self, checks: list[SemanticOracleCheck]) -> str:
        if not checks:
            return ""
        lines = ["# Executable Semantic Oracle Plan", "These checks derive from the contract and Data Card, not from benchmark labels."]
        for check in checks:
            fields = ", ".join(check.evidence_fields)
            lines.append(f"- [{check.severity}] {check.id}: {check.expected}")
            if fields:
                lines.append(f"  evidence_fields: {fields}")
            if check.prompt:
                lines.append(f"  instruction: {check.prompt}")
        return "\n".join(lines)

    def validate(
        self,
        final_block: FinalAnswerBlock | None,
        checks: list[SemanticOracleCheck],
        response_text: str | None = None,
    ) -> ValidationResult:
        if not checks:
            return ValidationResult(True, checks=["semantic_oracles_not_required"], score=1.0)
        passed_checks: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        evidence = final_block.evidence if final_block else {}
        evidence_text = _flatten_text(evidence) + "\n" + (response_text or "")[-5000:].lower()

        if final_block is None:
            return ValidationResult(False, errors=["semantic_oracle_final_block_missing"], score=0.0)

        for check in checks:
            self._validate_check(check, evidence, evidence_text, passed_checks, errors, warnings)

        return ValidationResult(
            passed=not errors,
            checks=passed_checks,
            errors=errors,
            warnings=warnings,
            score=_semantic_score(passed_checks, errors, warnings),
        )

    def manifest(self, checks: list[SemanticOracleCheck]) -> list[dict[str, Any]]:
        return [to_plain(check) for check in checks]

    def _validate_check(
        self,
        check: SemanticOracleCheck,
        evidence: dict[str, Any],
        evidence_text: str,
        passed_checks: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        missing = [field for field in check.evidence_fields if not _evidence_has_field(evidence, field)]
        if missing:
            warnings.append(f"semantic_evidence_missing:{check.id}:{','.join(missing)}")
        else:
            passed_checks.append(f"semantic_evidence_present:{check.id}")

        validator = _ORACLE_VALIDATORS.get(check.id)
        if validator is None:
            passed_checks.append(f"semantic_oracle_recorded:{check.id}")
            return
        validator(evidence, evidence_text, passed_checks, errors, warnings)


def _validate_std_sample(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if _mentions_population_std(evidence_text) or "ddof=0" in evidence_text or "ddof 0" in evidence_text:
        errors.append("semantic_oracle_violation:std_expected_ddof1")
    elif _mentions_sample_std(evidence_text) or "ddof=1" in evidence_text or "ddof 1" in evidence_text:
        passed_checks.append("semantic_oracle_pass:std_ddof1")
    else:
        warnings.append("semantic_oracle_unverified:std_ddof1")


def _validate_std_population(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if _mentions_sample_std(evidence_text) or "ddof=1" in evidence_text or "ddof 1" in evidence_text:
        errors.append("semantic_oracle_violation:std_expected_ddof0")
    elif _mentions_population_std(evidence_text) or "ddof=0" in evidence_text or "ddof 0" in evidence_text:
        passed_checks.append("semantic_oracle_pass:std_ddof0")
    else:
        warnings.append("semantic_oracle_unverified:std_ddof0")


def _validate_year_2000_inclusive(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    inclusive_pattern = r">=\s*2000|include\w*\s+2000|including\s+2000"
    if re.search(r">\s*2000", evidence_text) and not re.search(inclusive_pattern, evidence_text):
        errors.append("semantic_oracle_violation:year_2000_should_be_inclusive")
    elif re.search(inclusive_pattern, evidence_text):
        passed_checks.append("semantic_oracle_pass:year_2000_inclusive")
    else:
        warnings.append("semantic_oracle_unverified:year_2000_inclusive")


def _validate_engineered_feature_baseline(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if re.search(r"baseline[^\n]{0,120}(?:only|just)\s+(?:source|length|diameter|height)", evidence_text):
        errors.append("semantic_oracle_violation:engineered_feature_baseline_too_narrow")
    elif "all non-target numeric" in evidence_text or "all existing non-target numeric" in evidence_text:
        passed_checks.append("semantic_oracle_pass:all_numeric_baseline")
    else:
        warnings.append("semantic_oracle_unverified:all_numeric_baseline")


def _validate_ml_trace(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    _errors: list[str],
    warnings: list[str],
) -> None:
    has_feature_target = all(token in evidence_text for token in ["feature", "target"])
    has_split = any(token in evidence_text for token in ["random_state", "random state", "split"])
    if has_feature_target and has_split:
        passed_checks.append("semantic_oracle_pass:ml_trace")
    else:
        warnings.append("semantic_oracle_unverified:ml_trace")


def _validate_leakage_policy(
    _evidence: dict[str, Any],
    evidence_text: str,
    passed_checks: list[str],
    _errors: list[str],
    warnings: list[str],
) -> None:
    if "leakage" in evidence_text or "based on" in evidence_text or "excluded" in evidence_text:
        passed_checks.append("semantic_oracle_pass:leakage_policy_stated")
    else:
        warnings.append("semantic_oracle_unverified:leakage_policy")


def _validate_source_columns(
    evidence: dict[str, Any],
    _evidence_text: str,
    passed_checks: list[str],
    _errors: list[str],
    warnings: list[str],
) -> None:
    if evidence.get("source_columns"):
        passed_checks.append("semantic_oracle_pass:source_columns")
    else:
        warnings.append("semantic_oracle_unverified:source_columns")


_ORACLE_VALIDATORS = {
    "oracle.stats.std_sample_ddof1": _validate_std_sample,
    "oracle.stats.std_population_ddof0": _validate_std_population,
    "oracle.filter.year_2000_inclusive": _validate_year_2000_inclusive,
    "oracle.ml.engineered_feature_baseline_all_numeric": _validate_engineered_feature_baseline,
    "oracle.ml.split_feature_metric_trace": _validate_ml_trace,
    "oracle.ml.target_derived_leakage_policy": _validate_leakage_policy,
    "oracle.data.required_columns_available": _validate_source_columns,
}


def _dedupe(checks: list[SemanticOracleCheck]) -> list[SemanticOracleCheck]:
    seen: set[str] = set()
    output: list[SemanticOracleCheck] = []
    for check in checks:
        if check.id in seen:
            continue
        seen.add(check.id)
        output.append(check)
    return output


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_flatten_text(item)}" for key, item in value.items()).lower()
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_flatten_text(item) for item in value).lower()
    return str(value).lower()


def _evidence_has_field(evidence: dict[str, Any], field: str) -> bool:
    if field in evidence and evidence[field] not in (None, "", [], {}):
        return True
    # Allow nested oracle_evidence or method_notes maps.
    for nested_key in ("oracle_evidence", "method_notes", "verification"):
        nested = evidence.get(nested_key)
        if isinstance(nested, dict) and nested.get(field) not in (None, "", [], {}):
            return True
    return False


def _mentions_sample_std(text: str) -> bool:
    return "sample standard" in text or "statistics.stdev" in text or "pandas std" in text or "ddof=1" in text


def _mentions_population_std(text: str) -> bool:
    return "population standard" in text or "statistics.pstdev" in text or "pstdev" in text or "ddof=0" in text


def _semantic_score(checks: list[str], errors: list[str], warnings: list[str]) -> float:
    base = len(checks) / max(len(checks) + len(errors) + len(warnings), 1)
    if errors:
        base *= 0.35
    return round(max(0.0, min(base, 1.0)), 4)
