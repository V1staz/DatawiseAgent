"""Machine-readable DataModeling skill cards.

The markdown guidance files remain the human-readable surface.  Skill cards are
the compact, testable bridge from a task route to concrete modeling actions,
quality evidence, and targeted repair hints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class SkillCard:
    id: str
    title: str
    families: tuple[str, ...]
    metric_families: tuple[str, ...]
    when_to_use: str
    required_actions: tuple[str, ...]
    validation_evidence: tuple[str, ...]
    repair_hints: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...] = ()
    prompt_budget: int = 360

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_summary(self) -> str:
        actions = "; ".join(self.required_actions[:3])
        evidence = "; ".join(self.validation_evidence[:2])
        return f"- `{self.id}`: {self.when_to_use} Actions: {actions}. Evidence: {evidence}."


DEFAULT_SKILL_CARDS: tuple[SkillCard, ...] = (
    SkillCard(
        id="schema_first_submission",
        title="Schema-first submission discipline",
        families=("all",),
        metric_families=("all",),
        when_to_use="Always enforce exact row count, ID order, target columns, dtypes, and final path before optimizing.",
        required_actions=(
            "read train.csv, test.csv, and sample_submission.csv before modeling",
            "derive final_submission.csv from the sample schema, preserving ID/order",
            "read back final_submission.csv before finishing",
        ),
        validation_evidence=(
            "final row count and columns match the local contract",
            "per-target missing count is zero",
        ),
        repair_hints=(
            "fix schema/ID/order before changing model logic",
            "copy a modeled candidate to final_submission.csv only after strict validation",
        ),
    ),
    SkillCard(
        id="fallback_separation_and_manifest",
        title="Fallback separation and manifest evidence",
        families=("all",),
        metric_families=("all",),
        when_to_use="Always separate fallback recovery from improved modeled submissions.",
        required_actions=(
            "write fallback_submission.csv only as recovery output",
            "write final_submission.csv only for selected modeled/improved candidate",
            "write submission_manifest.json with stage, validation metric, baseline metric, trained rows, and prediction diversity",
        ),
        validation_evidence=(
            "manifest stage is improved only when validation beats fallback",
            "prediction_summary reports per-target diversity",
        ),
        repair_hints=(
            "if manifest is missing, add it next to final_submission.csv",
            "if final equals fallback, rerun modeling or mark fallback_recovery instead of improved",
        ),
        forbidden_shortcuts=("final_submission_without_manifest", "fallback_as_improved_final"),
    ),
    SkillCard(
        id="public_data_availability_guard",
        title="Public data availability guard",
        families=("data_unavailable_or_pointer",),
        metric_families=("all", "unknown", "probability", "regression", "classification", "text"),
        when_to_use="Use when train/test/sample public files are missing or Git LFS pointer stubs rather than real CSV data.",
        required_actions=(
            "verify train.csv, test.csv, and sample_submission.csv are real public CSVs before modeling",
            "do not treat Git LFS pointer text as a text feature or tabular schema",
            "surface the data availability blocker instead of running meaningless modeling",
        ),
        validation_evidence=(
            "data_availability.json lists blocked public files",
            "protocol_plan target_type is data_unavailable_or_pointer",
        ),
        repair_hints=(
            "restore/download the missing public CSV data before attempting a modeled final",
            "if only sample_submission is usable, do not claim improved modeling",
        ),
        forbidden_shortcuts=("modeling_on_lfs_pointer_bytes", "text_misroute_from_pointer_file"),
    ),
    SkillCard(
        id="classification_probability_modeling",
        title="Probability modeling for ranking/probability metrics",
        families=("tabular_binary_classification", "text_modeling"),
        metric_families=("probability",),
        when_to_use="Use for ROC AUC, log loss, average precision, normalized gini, or probability-output tasks.",
        required_actions=(
            "train a classifier that can produce predict_proba or calibrated scores",
            "output continuous probabilities in the metric range, not hard labels",
            "use stratified validation when labels are imbalanced and enough rows exist",
        ),
        validation_evidence=(
            "probability target has values beyond just {0, 1}",
            "validation metric is compared against a constant/majority fallback",
        ),
        repair_hints=(
            "replace hard class predictions with predict_proba[:, 1] or calibrated decision scores",
            "if all probabilities are equal, add usable features/preprocessing and rerun validation",
        ),
        forbidden_shortcuts=("hard_labels_for_probability_metric", "constant_probability_final"),
    ),
    SkillCard(
        id="multiclass_class_coverage",
        title="Multiclass coverage and label preservation",
        families=("tabular_multiclass_classification",),
        metric_families=("classification",),
        when_to_use="Use for class/top-k/classification metrics with more than two classes.",
        required_actions=(
            "preserve original class labels and sample submission target semantics",
            "validate class coverage instead of predicting one majority class everywhere",
            "for top-k metrics, output ranked classes in the required format",
        ),
        validation_evidence=(
            "predicted class diversity is reported",
            "validation compares against majority-class fallback",
        ),
        repair_hints=(
            "if all rows have one class, train a multiclass model or use per-class probabilities/rankings",
            "if labels are remapped, convert predictions back to original labels",
        ),
        forbidden_shortcuts=("single_class_final", "unmapped_encoded_labels"),
    ),
    SkillCard(
        id="regression_cv_baseline",
        title="Regression validation against a baseline",
        families=("tabular_regression", "time_series_or_grouped"),
        metric_families=("regression",),
        when_to_use="Use for numeric regression/correlation metrics where low-information constants often validate structurally but score poorly.",
        required_actions=(
            "build a baseline using public train targets only",
            "train at least one numeric model or defensible feature baseline",
            "compare validation/proxy performance against the baseline with the metric direction",
        ),
        validation_evidence=(
            "validation_metric_value and baseline_metric_value are both present",
            "prediction diversity is non-constant unless fallback_recovery is declared",
        ),
        repair_hints=(
            "if final is constant, add feature preprocessing/modeling or mark fallback_recovery",
            "if validation is not better than baseline, do not claim stage=improved",
        ),
        forbidden_shortcuts=("constant_final_without_validation_evidence",),
    ),
    SkillCard(
        id="multi_output_nonconstant_modeling",
        title="Multi-output non-constant modeling",
        families=("multi_output",),
        metric_families=("regression", "classification", "probability", "unknown"),
        when_to_use="Use when the submission has multiple target columns or multi-target task hints.",
        required_actions=(
            "model each target or use a multi-output estimator",
            "avoid copying one constant or identical vector across all targets",
            "report per-target prediction diversity in the manifest",
        ),
        validation_evidence=(
            "most target columns have non-zero diversity",
            "manifest explains per-target or multi-output strategy",
        ),
        repair_hints=(
            "if many targets are constant, train per-target baselines/models and rewrite all target columns",
            "if a target truly has no signal, document it separately instead of making all columns constant",
        ),
        forbidden_shortcuts=("all_targets_constant", "identical_prediction_vector_for_all_targets"),
    ),
    SkillCard(
        id="text_feature_baseline",
        title="Text feature baseline",
        families=("text_modeling",),
        metric_families=("probability", "regression", "classification", "text"),
        when_to_use="Use when contract/profile detects text columns or text metrics.",
        required_actions=(
            "use text columns through vectorization, hashing, length/count features, or task-specific parsing",
            "combine text features with tabular features when useful",
            "validate against a non-text fallback when possible",
        ),
        validation_evidence=(
            "manifest lists text features or vectorizer family used",
            "text-task predictions are not one constant numeric value",
        ),
        repair_hints=(
            "if text output is constant, add a simple text vectorizer/length feature baseline",
            "if output type is wrong, align with sample submission and metric prediction kind",
        ),
        forbidden_shortcuts=("ignored_text_columns", "constant_text_metric_output"),
    ),
    SkillCard(
        id="time_group_split_awareness",
        title="Time/group-aware validation",
        families=("time_series_or_grouped",),
        metric_families=("regression", "probability", "classification", "unknown"),
        when_to_use="Use when dates, forecast wording, grouped entities, or time-series hints appear.",
        required_actions=(
            "identify datetime/group columns and avoid random leakage-prone validation when inappropriate",
            "create lag/date/group aggregate features from public train data only when safe",
            "compare against a seasonal/group/global fallback",
        ),
        validation_evidence=(
            "manifest names validation_strategy and time/group columns considered",
            "final is not merely a global mean unless declared fallback_recovery",
        ),
        repair_hints=(
            "if final is a global mean, add time/group features or a group-aware fallback",
            "if validation leaks future rows, switch to ordered/group split",
        ),
        forbidden_shortcuts=("global_mean_as_improved_forecast", "time_leakage_validation"),
    ),
)


def all_skill_cards() -> list[SkillCard]:
    return list(DEFAULT_SKILL_CARDS)


def get_skill_cards(ids: Iterable[str]) -> list[SkillCard]:
    by_id = {card.id: card for card in DEFAULT_SKILL_CARDS}
    return [by_id[card_id] for card_id in ids if card_id in by_id]


def skill_card_ids_for_route(family: str, metric_family: str, facets: Iterable[str] | None = None) -> list[str]:
    route_families = {family, *(facets or [])}
    selected: list[str] = []
    for card in DEFAULT_SKILL_CARDS:
        family_match = "all" in card.families or bool(route_families.intersection(card.families))
        metric_match = "all" in card.metric_families or metric_family in card.metric_families
        if family_match and metric_match:
            selected.append(card.id)
    return selected
