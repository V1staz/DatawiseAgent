"""Skill routing and fixed protocol snippets for the harness prompt."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .rules import HarnessRule, RuleBook


@dataclass(frozen=True, slots=True)
class SkillProtocol:
    name: str
    description: str
    required_checks: tuple[str, ...]
    prompt_directives: tuple[str, ...]


_SKILLS: dict[str, SkillProtocol] = {
    "summary_statistics": SkillProtocol(
        name="summary_statistics",
        description="Means, medians, counts, sums, standard deviations, grouped summaries.",
        required_checks=("column_exists", "sample_size", "missing_policy", "rounding", "ddof_policy"),
        prompt_directives=(
            "State the exact column(s), missing-value policy, grouping grain, and sample size before reporting a statistic.",
            "For standard deviation, default to sample standard deviation (ddof=1, statistics.stdev / pandas std) unless the question explicitly says population, pstdev, or ddof=0.",
        ),
    ),
    "distribution_analysis": SkillProtocol(
        name="distribution_analysis",
        description="Distribution tests, histograms, normality checks, quantiles.",
        required_checks=("column_exists", "sample_size", "test_statistic", "p_value_threshold"),
        prompt_directives=(
            "Record the statistical test and threshold before interpreting significance.",
            "Preserve both statistic and p-value if the format asks for both.",
        ),
    ),
    "correlation_analysis": SkillProtocol(
        name="correlation_analysis",
        description="Pearson/Spearman correlation and significance decisions.",
        required_checks=("method", "column_exists", "pairwise_sample_size", "sign_preserved", "p_value_if_required"),
        prompt_directives=(
            "Record Pearson vs Spearman explicitly and preserve the sign of r.",
            "If comparing correlations, state whether absolute value or signed value is used.",
        ),
    ),
    "outlier_detection": SkillProtocol(
        name="outlier_detection",
        description="IQR, z-score, threshold based outlier detection.",
        required_checks=("column_exists", "threshold_formula", "outlier_count", "index_base", "sample_size"),
        prompt_directives=(
            "For IQR, record Q1, Q3, IQR, lower, upper, and outlier count.",
            "For z-score, record mean, std, threshold, and whether std is population or sample; after removing outliers, default follow-up standard deviation to ddof=1 unless population is explicit.",
        ),
    ),
    "comprehensive_data_preprocessing": SkillProtocol(
        name="comprehensive_data_preprocessing",
        description="Deletion, imputation, conversion, filtering, normalization sequencing.",
        required_checks=("operation_order", "shape_delta", "missing_delta", "dtype_delta", "no_unrequested_imputation"),
        prompt_directives=(
            "Apply preprocessing in the order stated by the question; log shape and missing values after each step.",
            "If the question forbids extra cleaning or says no missing values, do not invent imputation.",
        ),
    ),
    "feature_engineering": SkillProtocol(
        name="feature_engineering",
        description="Create derived columns, buckets, aggregations, transformations.",
        required_checks=("formula", "source_columns", "preview_values", "group_count"),
        prompt_directives=(
            "State the feature formula before creating the column.",
            "Verify the first few generated values against the formula before aggregation.",
            "Respect boundary wording exactly: phrases like '2000 and beyond' include 2000, while 'after/beyond 2000' only excludes 2000 when no inclusive wording is present.",
            "If a broad question phrase conflicts with explicit constraints, prefer the explicit constraints; for example, 'years 2000 and beyond' means Year >= 2000 even if the question text also says 'beyond 2000'.",
        ),
    ),
    "machine_learning": SkillProtocol(
        name="machine_learning",
        description="Train/test split, model fitting, metrics, leakage checks.",
        required_checks=("feature_columns", "target_column", "split", "random_state", "metric_direction", "leakage_check"),
        prompt_directives=(
            "Record train/test split, random_state, feature columns, target, and metric exactly.",
            "Do not tune hyperparameters unless the question explicitly asks for it.",
            "If the target label is derived from a feature, include that source feature only when the question explicitly says to classify/predict based on that feature or gives no alternative predictors; otherwise treat it as leakage and use the remaining relevant features.",
            "For regression/classification comparisons, unless a restricted feature set is specified, use all relevant available predictors and compare base predictors versus base predictors plus the engineered feature.",
            "When a question asks whether a new engineered feature improves prediction, do not limit the baseline model to only the source columns used to build that feature; use all existing non-target numeric predictors as the baseline and add the engineered feature for the comparison, keeping the same split.",
        ),
    ),
    "final_format": SkillProtocol(
        name="final_format",
        description="Machine-readable FinalAnswerBlock and @answer[value] rendering.",
        required_checks=("answer_names", "value_types", "rounding", "format_targets", "verified_true"),
        prompt_directives=(
            "End with exactly one fenced JSON FinalAnswerBlock after all computation is complete.",
            "Every required @answer_name[value] must appear in format_targets and must be derived from answers.",
        ),
    ),
}

_CONCEPT_TO_SKILL = {
    "Summary Statistics": "summary_statistics",
    "Distribution Analysis": "distribution_analysis",
    "Correlation Analysis": "correlation_analysis",
    "Outlier Detection": "outlier_detection",
    "Comprehensive Data Preprocessing": "comprehensive_data_preprocessing",
    "Feature Engineering": "feature_engineering",
    "Machine Learning": "machine_learning",
    "Final Format": "final_format",
}

_KEYWORD_TO_SKILL = {
    "mean": "summary_statistics",
    "median": "summary_statistics",
    "standard deviation": "summary_statistics",
    "average": "summary_statistics",
    "correlation": "correlation_analysis",
    "pearson": "correlation_analysis",
    "spearman": "correlation_analysis",
    "outlier": "outlier_detection",
    "iqr": "outlier_detection",
    "z-score": "outlier_detection",
    "shapiro": "distribution_analysis",
    "distribution": "distribution_analysis",
    "preprocess": "comprehensive_data_preprocessing",
    "clean": "comprehensive_data_preprocessing",
    "missing": "comprehensive_data_preprocessing",
    "new feature": "feature_engineering",
    "create a new column": "feature_engineering",
    "train": "machine_learning",
    "model": "machine_learning",
    "accuracy": "machine_learning",
}


class SkillRegistry:
    def route(self, question: dict[str, Any]) -> list[SkillProtocol]:
        names: list[str] = []
        seen: set[str] = set()
        for concept in question.get("concepts", []) or []:
            name = _CONCEPT_TO_SKILL.get(concept)
            if name and name not in seen:
                seen.add(name)
                names.append(name)

        text = "\n".join(
            str(question.get(key, "")) for key in ("question", "constraints", "format")
        ).lower()
        for keyword, name in _KEYWORD_TO_SKILL.items():
            if keyword in text and name not in seen:
                seen.add(name)
                names.append(name)
        if "final_format" not in seen:
            names.append("final_format")
        return [_SKILLS[name] for name in names]

    def get(self, name: str) -> SkillProtocol:
        return _SKILLS[name]

    def to_prompt(self, skills: list[SkillProtocol], rules: list[HarnessRule] | None = None) -> str:
        """Render routed skills plus persistent rule directives.

        `prompt_directives` remain on SkillProtocol for backward compatibility,
        but the runtime path now prefers JSONL-backed RuleBook cards so prompt
        engineering can be audited and persisted outside one-off prompts.
        """
        if rules is None:
            rules = RuleBook().select_for_skills([skill.name for skill in skills])
        rules_by_skill: dict[str, list[HarnessRule]] = {}
        for rule in rules:
            rules_by_skill.setdefault(rule.skill, []).append(rule)

        lines = ["# Routed Skill Protocols"]
        for skill in skills:
            payload = asdict(skill)
            lines.append(f"## {skill.name}")
            lines.append(f"description: {payload['description']}")
            lines.append("required_checks: " + ", ".join(skill.required_checks))
            selected = rules_by_skill.get(skill.name, [])
            if selected:
                lines.append("persistent_rules:")
                for rule in selected:
                    lines.append(f"- [{rule.id}] {rule.directive}")
            else:
                lines.append("directives:")
                for directive in skill.prompt_directives:
                    lines.append(f"- {directive}")
        return "\n".join(lines)
