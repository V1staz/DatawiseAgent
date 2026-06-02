from __future__ import annotations

import importlib.util
import asyncio
import contextlib
import io
import logging
import sys
import types
import tempfile
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "evaluation" / "InfiAgentBench" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from datawiseagent.harness.contract import ContractParser, extract_format_targets
from datawiseagent.harness.finalizer import (
    build_format_targets,
    final_block_from_legacy_targets,
    normalize_final_block_to_contract,
    render_reformat_response,
)
from datawiseagent.harness.memory import FailureMemoryStore
from datawiseagent.harness.oracles import SemanticOracleRegistry
from datawiseagent.harness.rules import RuleBook
from datawiseagent.harness.self_memory_schema import SELF_MEMORY_COPY_POLICY, SELF_UPDATED_MEMORY_KIND
from datawiseagent.harness.schemas import FinalAnswerBlock, HarnessContext, QuestionContract, TargetAnswer
from datawiseagent.harness.controller import HarnessConfig, HarnessController
from datawiseagent.harness.search import SearchController
from datawiseagent.harness.skills import SkillRegistry
from datawiseagent.harness.verifier import Verifier
from datawiseagent.management.manager import DatawiseAgentManager


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class HarnessRegressionTests(unittest.TestCase):
    def test_per_answer_rounding_is_target_specific(self) -> None:
        fmt = """@shapiro_wilk_statistic[test_statistic]
@shapiro_wilk_p_value[p_value]
where "test_statistic" is a number. Rounding off the answer to two decimal places.
where "p_value" is a number. Rounding off the answer to four decimal places."""
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.rounding) for target in targets],
            [("shapiro_wilk_statistic", 2), ("shapiro_wilk_p_value", 4)],
        )
        rendered = build_format_targets(
            {
                "shapiro_wilk_statistic": 0.5555,
                "shapiro_wilk_p_value": "0.00023456",
            },
            fmt,
        )
        self.assertEqual(rendered["shapiro_wilk_statistic"], "@shapiro_wilk_statistic[0.56]")
        self.assertEqual(rendered["shapiro_wilk_p_value"], "@shapiro_wilk_p_value[0.0002]")

    def test_number_targets_are_not_forced_to_integer(self) -> None:
        fmt = (
            "@mean_fare_child[mean_fare], where \"mean_fare\" is a "
            "float number rounded to 2 decimal places."
        )
        [target] = extract_format_targets(fmt)
        self.assertEqual(target.type, "float")

    def test_prediction_name_does_not_trigger_dict_type(self) -> None:
        fmt = (
            '@prediction_accuracy[accuracy], where "accuracy" is a '
            "float number rounded to 2 decimal places."
        )
        [target] = extract_format_targets(fmt)
        self.assertEqual(target.type, "float")

    def test_input_count_phrase_does_not_override_correlation_target(self) -> None:
        fmt = (
            '@correlation_coefficient[r_value] where "r_value" is a number '
            "between -1 and 1, rounded to three decimal places, representing "
            "the correlation between the review count and the bubble score."
        )
        [target] = extract_format_targets(fmt)
        self.assertEqual(target.type, "float")

    def test_target_specific_type_phrase_prevents_adjacent_leakage(self) -> None:
        fmt = (
            '@p_value[p_value] @normality_test[normality_test] where "p_value" '
            'is a number between 0 and 1, rounded to four decimal places. '
            '"normality_test" is a string which can either be "normal" or '
            '"not_normal".'
        )
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.type) for target in targets],
            [("p_value", "float"), ("normality_test", "string")],
        )

    def test_target_specific_rounding_ignores_leading_target_list(self) -> None:
        fmt = (
            '@correlation_coefficient[r_value] @p_value[p_value] '
            '@relationship_type[relationship_type] where r_value is a number '
            'between -1 and 1, rounded to two decimal places. Where p_value '
            'is a number between 0 and 1, rounded to four decimal places. '
            'Where relationship_type is a string that can either be "linear", '
            '"nonlinear", or "none".'
        )
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.type, target.rounding) for target in targets],
            [
                ("correlation_coefficient", "float", 2),
                ("p_value", "float", 4),
                ("relationship_type", "string", None),
            ],
        )
        rendered = build_format_targets(
            {
                "correlation_coefficient": 0.02,
                "p_value": 0.1704,
                "relationship_type": "none",
            },
            fmt,
        )
        self.assertEqual(rendered["p_value"], "@p_value[0.1704]")

    def test_boolean_target_type_and_legacy_coercion(self) -> None:
        fmt = (
            '@norm_test_pvalue[pvalue] @is_normal[isNormal] where "pvalue" '
            'is a decimal number rounded to four decimal places and "isNormal" '
            "is a boolean value, 'True' if normal and 'False' otherwise."
        )
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.type) for target in targets],
            [("norm_test_pvalue", "float"), ("is_normal", "boolean")],
        )
        contract = QuestionContract(
            question_id=130,
            question="q",
            target_answers=targets,
            final_format=fmt,
        )
        block = final_block_from_legacy_targets(
            "@norm_test_pvalue[0.0000] @is_normal[False]",
            contract,
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertIs(block.answers["is_normal"], False)
        validation = Verifier().validate_final_block(block, contract)
        self.assertNotIn("answer_type_mismatch:is_normal:boolean", validation.warnings)
        self.assertIn("@is_normal[False]", render_reformat_response(block, fmt))

    def test_boolean_target_can_render_lowercase_when_format_uses_lowercase(self) -> None:
        fmt = (
            "@significant_correlation[significant_correlation] where "
            '"significant_correlation" is a boolean value indicating whether '
            "there is a significant correlation (true) or not (false)."
        )
        block = FinalAnswerBlock(
            question_id="q",
            answers={"significant_correlation": True},
            format_targets={"significant_correlation": "@significant_correlation[True]"},
            verified=True,
        )
        self.assertEqual(
            render_reformat_response(block, fmt),
            "@significant_correlation[true]",
        )

    def test_grouped_float_descriptions_do_not_mark_skewness_as_integer(self) -> None:
        fmt = (
            "@age_skewness[skewness_value]\n"
            "@age_kurtosis[kurtosis_value]\n"
            "@age_values_within_one_stdev[number]\n"
            "@fare_skewness[skewness_value]\n"
            "@fare_kurtosis[kurtosis_value]\n"
            "@fare_values_within_one_stdev[number]\n"
            'where "skewness_value", "kurtosis_value" are floats with two '
            'decimals, "number" is a positive integer.'
        )
        targets = extract_format_targets(fmt)
        by_name = {target.name: target for target in targets}
        self.assertEqual(by_name["age_skewness"].type, "float")
        self.assertEqual(by_name["age_skewness"].rounding, 2)
        self.assertEqual(by_name["fare_kurtosis"].type, "float")
        self.assertEqual(by_name["fare_values_within_one_stdev"].type, "integer")

    def test_mixed_component_rounding_in_single_answer_target(self) -> None:
        fmt = (
            "@fare_after_scaling[min_fare, max_fare, mean_fare]\n"
            'where "min_fare" and "max_fare" are rounded to two decimal places, '
            'while "mean_fare" is rounded to four decimal places.'
        )
        rendered = build_format_targets(
            {"fare_after_scaling": "0.0, 1.0, 0.0629"},
            fmt,
        )
        self.assertEqual(
            rendered["fare_after_scaling"],
            "@fare_after_scaling[0.00, 1.00, 0.0629]",
        )

    def test_bracketed_component_string_is_unwrapped_for_multi_value_target(self) -> None:
        fmt = (
            "@sex_encoded_count[label_0_count, label_1_count]\n"
            "@fare_after_scaling[min_fare, max_fare, mean_fare]\n"
            'where "min_fare" and "max_fare" are rounded to two decimal places, '
            'while "mean_fare" is rounded to four decimal places.'
        )
        rendered = build_format_targets(
            {
                "sex_encoded_count": "[314, 577]",
                "fare_after_scaling": "[0.0, 1.0, 0.0629]",
            },
            fmt,
        )
        self.assertEqual(rendered["sex_encoded_count"], "@sex_encoded_count[314, 577]")
        self.assertEqual(
            rendered["fare_after_scaling"],
            "@fare_after_scaling[0.00, 1.00, 0.0629]",
        )

    def test_quoted_string_placeholders_are_preserved(self) -> None:
        fmt = '@missing_values_handled["Yes"/"No"], @distribution_type[distribution type]'
        rendered = build_format_targets(
            {"missing_values_handled": "Yes", "distribution_type": "Non-Normal"},
            fmt,
        )
        self.assertEqual(rendered["missing_values_handled"], '@missing_values_handled["Yes"]')
        self.assertEqual(rendered["distribution_type"], "@distribution_type[Non-Normal]")

    def test_nonempty_list_targets_render_without_nested_brackets(self) -> None:
        fmt = "@outlier_countries[list_of_strings]"
        rendered = build_format_targets(
            {"outlier_countries": ["Kuwait", "Saudi Arabia"]},
            fmt,
        )
        self.assertEqual(rendered["outlier_countries"], "@outlier_countries[Kuwait, Saudi Arabia]")

    def test_mixed_integer_and_floating_point_targets(self) -> None:
        fmt = (
            "@total_outliers[total_outliers]\n"
            "@mean_charges_outliers[mean_charges_outliers]\n"
            "@median_charges_outliers[median_charges_outliers]\n"
            'where "total_outliers" is an integer, "mean_charges_outliers" '
            'and "median_charges_outliers" are floating-point numbers rounded '
            "to two decimal places."
        )
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.type) for target in targets],
            [
                ("total_outliers", "integer"),
                ("mean_charges_outliers", "float"),
                ("median_charges_outliers", "float"),
            ],
        )

    def test_average_std_targets_float_and_year_range_checked(self) -> None:
        fmt = (
            "@highest_horsepower_vehicle[vehicle_model_year]\n"
            "@average_horsepower[same_year_avg_horsepower]\n"
            "@standard_deviation[same_year_horsepower_std]\n"
            'where "vehicle_model_year" is an integer from 1900 to the current year. '
            '"same_year_avg_horsepower" and "same_year_horsepower_std" are numbers '
            "rounded to two decimal places."
        )
        targets = extract_format_targets(fmt)
        self.assertEqual(
            [(target.name, target.type) for target in targets],
            [
                ("highest_horsepower_vehicle", "integer"),
                ("average_horsepower", "float"),
                ("standard_deviation", "float"),
            ],
        )
        contract = QuestionContract(
            question_id=722,
            question="q",
            target_answers=targets,
            final_format=fmt,
        )
        bad_block = FinalAnswerBlock(
            question_id=722,
            answers={
                "highest_horsepower_vehicle": 73,
                "average_horsepower": 130.48,
                "standard_deviation": 45.83,
            },
            format_targets={
                "highest_horsepower_vehicle": "@highest_horsepower_vehicle[73]",
                "average_horsepower": "@average_horsepower[130.48]",
                "standard_deviation": "@standard_deviation[45.83]",
            },
            verified=True,
        )
        validation = Verifier().validate_final_block(bad_block, contract)
        self.assertFalse(validation.passed)
        self.assertTrue(
            any(error.startswith("answer_range_violation:highest_horsepower_vehicle") for error in validation.errors)
        )
        normalized = normalize_final_block_to_contract(bad_block, contract)
        assert normalized is not None
        self.assertEqual(normalized.answers["highest_horsepower_vehicle"], 1973)
        self.assertEqual(
            normalized.format_targets["highest_horsepower_vehicle"],
            "@highest_horsepower_vehicle[1973]",
        )
        self.assertTrue(Verifier().validate_final_block(normalized, contract).passed)

    def test_conditional_alternative_targets_are_not_required_by_contract(self) -> None:
        fmt = (
            '@correlation_coefficient[r_value] @p_value[p_value] where "r_value" '
            'is a number between -1 and 1; "p_value" is a number between 0 and 1. '
            'If there is no significant correlation, please simply output '
            '@correlation_status["No significant correlation"]'
        )
        self.assertEqual(
            [target.name for target in extract_format_targets(fmt)],
            ["correlation_coefficient", "p_value", "correlation_status"],
        )
        contract = ContractParser().parse(
            {
                "id": 756,
                "question": "q",
                "constraints": "",
                "format": fmt,
                "concepts": ["Correlation Analysis"],
                "level": "medium",
            }
        )
        self.assertEqual(
            [target.name for target in contract.target_answers],
            ["correlation_coefficient", "p_value"],
        )

    def test_final_block_format_targets_must_match_answers(self) -> None:
        block = FinalAnswerBlock(
            question_id="q",
            answers={"a": 1},
            format_targets={"a": "@a[2]"},
            verified=True,
        )
        contract = QuestionContract(
            question_id="q",
            question="q",
            target_answers=[TargetAnswer(name="a", type="integer", format="@a[a]")],
            final_format="@a[a]",
        )
        validation = Verifier().validate_final_block(block, contract)
        self.assertFalse(validation.passed)
        self.assertIn("format_target_answer_mismatch:a", validation.errors)
        self.assertEqual(render_reformat_response(block, "@a[a]"), "@a[1]")

    def test_format_target_trailing_zero_is_numeric_match(self) -> None:
        block = FinalAnswerBlock(
            question_id="q",
            answers={"mse": 11439.6},
            format_targets={"mse": "@mse[11439.6]"},
            verified=True,
        )
        contract = QuestionContract(
            question_id="q",
            question="q",
            target_answers=[
                TargetAnswer(name="mse", type="float", rounding=2, format="@mse[x]")
            ],
            final_format="@mse[x] where x is a number rounded to two decimal places.",
        )
        validation = Verifier().validate_final_block(block, contract)
        self.assertTrue(validation.passed)

    def test_legacy_answer_targets_can_recover_final_block(self) -> None:
        contract = QuestionContract(
            question_id="q",
            question="q",
            target_answers=[
                TargetAnswer(name="correlation_coefficient", type="float", rounding=2)
            ],
            final_format="@correlation_coefficient[r_value]",
        )
        block = final_block_from_legacy_targets(
            "Final Answer: @correlation_coefficient[0.21]",
            contract,
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block.answers["correlation_coefficient"], 0.21)
        self.assertFalse(block.verified)

    def test_percent_float_legacy_answer_renders_numeric_value(self) -> None:
        contract = QuestionContract(
            question_id=589,
            question="q",
            target_answers=[
                TargetAnswer(
                    name="abandonment_rate",
                    type="float",
                    rounding=2,
                    format="@abandonment_rate[x]",
                )
            ],
            final_format="@abandonment_rate[x] where x is rounded to two decimal places.",
        )
        block = final_block_from_legacy_targets("@abandonment_rate[6.25%]", contract)
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block.answers["abandonment_rate"], 6.25)
        self.assertEqual(render_reformat_response(block, contract.final_format), "@abandonment_rate[6.25]")
        validation = Verifier().validate_final_block(block, contract)
        self.assertTrue(validation.passed)

    def test_legacy_recovery_ignores_late_template_placeholders(self) -> None:
        contract = QuestionContract(
            question_id=589,
            question="q",
            target_answers=[
                TargetAnswer(
                    name="abandonment_rate",
                    type="float",
                    rounding=2,
                    format="@abandonment_rate[abandonment_rate_%]",
                )
            ],
            final_format="@abandonment_rate[abandonment_rate_%]",
        )
        block = final_block_from_legacy_targets(
            "Final answer: @abandonment_rate[6.25%]\n"
            "Format matches @abandonment_rate[abandonment_rate_%].",
            contract,
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block.answers["abandonment_rate"], 6.25)
        self.assertEqual(render_reformat_response(block, contract.final_format), "@abandonment_rate[6.25]")

    def test_hard_only_policy_is_strictly_hard_level(self) -> None:
        controller = SearchController("hard-only")
        for level, expected in [("easy", False), ("medium", False), ("hard", True)]:
            context = HarnessContext(
                question={"id": "x", "level": level, "concepts": ["a", "b"]},
                table_path="x.csv",
                mode="full",
                prompt="",
            )
            self.assertIs(controller.pretrigger(context), expected)

    def test_eval_uses_reformat_response_even_when_raw_response_empty(self) -> None:
        eval_closed_form = load_module(
            "eval_closed_form_for_test",
            SCRIPTS / "eval_closed_form.py",
        )
        labels = [{"id": "q1", "common_answers": [["answer", "42"]]}]
        responses = [{"id": "q1", "response": "", "reformat_response": "@answer[42]"}]
        results = eval_closed_form.evaluate_responses(labels, responses)
        self.assertEqual(len(results), 1)
        self.assertTrue(all(results[0]["correctness"].values()))

    def test_eval_does_not_fallback_to_raw_when_reformat_is_empty(self) -> None:
        eval_closed_form = load_module(
            "eval_closed_form_empty_reformat_for_test",
            SCRIPTS / "eval_closed_form.py",
        )
        labels = [{"id": "q1", "common_answers": [["answer", "42"]]}]
        responses = [
            {
                "id": "q1",
                "response": "@answer[42]",
                "reformat_response": "",
                "final_block_missing": True,
            }
        ]
        results = eval_closed_form.evaluate_responses(labels, responses)
        self.assertEqual(len(results), 1)
        self.assertFalse(all(results[0]["correctness"].values()))

    def test_reformat_call_has_bounded_retries(self) -> None:
        reformat = load_module("reformat_for_test", SCRIPTS / "reformat.py")
        attempts: list[int] = []

        class FakeCompletions:
            def create(self, **_data):
                attempts.append(1)
                raise ValueError("boom")

        class FakeOpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        old_openai = sys.modules.get("openai")
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=FakeOpenAI)
        old_logging_disable = logging.root.manager.disable
        args = types.SimpleNamespace(
            max_resp=16,
            model="fake-model",
            api_key="test-key",
            url="http://localhost/v1",
            request_timeout=0.01,
            max_retries=2,
            retry_sleep=0,
        )
        try:
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    reformat.call([{"role": "user", "content": "x"}], args)
            self.assertEqual(len(attempts), 2)
        finally:
            logging.disable(old_logging_disable)
            if old_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = old_openai

    def test_skill_prompt_contains_statistical_convention_repairs(self) -> None:
        registry = SkillRegistry()
        skills = registry.route(
            {
                "concepts": ["Summary Statistics", "Outlier Detection", "Feature Engineering", "Machine Learning"],
                "question": "standard deviation z-score new feature train model",
                "constraints": "",
                "format": "",
            }
        )
        prompt = registry.to_prompt(skills)
        self.assertIn("ddof=1", prompt)
        self.assertIn("2000 and beyond", prompt)
        self.assertIn("prefer the explicit constraints", prompt)
        self.assertIn("treat it as leakage", prompt)
        self.assertIn("classify/predict based on that feature", prompt)
        self.assertIn("all existing non-target numeric predictors", prompt)

    def test_session_broadcast_does_not_serialize_without_websockets(self) -> None:
        class ExplodingSessionContent:
            def model_dump_json(self) -> str:
                raise RuntimeError("should not serialize when no websocket is attached")

        manager = DatawiseAgentManager()
        asyncio.run(
            manager.broadcast_session_update(
                user_id="00000000-0000-0000-0000-000000000001",
                session_id="00000000-0000-0000-0000-000000000002",
                session_content=ExplodingSessionContent(),
            )
        )

    def test_rulebook_selects_persistent_rules_without_report_memory(self) -> None:
        contract = QuestionContract(
            question_id=549,
            question="Does a new engineered feature improve prediction?",
            level="hard",
            concepts=["Machine Learning", "Feature Engineering"],
            methods=["train_test_split"],
            target_answers=[TargetAnswer(name="rmse", type="float")],
            final_format="@rmse[x]",
        )
        rules = RuleBook().select(contract, question={"question": contract.question, "constraints": "", "format": contract.final_format})
        ids = [rule.id for rule in rules]
        self.assertIn("ml.engineered_feature_all_numeric_baseline", ids)
        self.assertIn("final.block_schema", ids)
        self.assertTrue(all("report" not in rule.source.lower() for rule in rules))

    def test_memory_retrieval_is_hierarchical_and_ignores_question_scope_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FailureMemoryStore(memory_path)
            contract = QuestionContract(
                question_id=1,
                question="standard deviation by year",
                level="hard",
                concepts=["Summary Statistics"],
                methods=["std"],
                required_columns=["Year", "Damage"],
            )
            store.add({
                "signature": "domain-std",
                "scope_level": "domain",
                "source": "semantic_oracle_report",
                "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                "failure_type": "semantic_oracle_gap",
                "repair": "Use sample standard deviation / ddof=1 when population is not explicit.",
                "severity": "hard",
                "contains_benchmark_label": False,
            })
            store.add({
                "signature": "question-only",
                "scope_level": "question",
                "source": "agent_trace",
                "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                "failure_type": "question_local",
                "repair": "Do not inject by default.",
                "contains_benchmark_label": False,
            })
            hits = store.retrieve(contract, columns=["Year", "Damage"], limit=5)
            self.assertEqual([hit["signature"] for hit in hits], ["domain-std"])
            self.assertEqual(hits[0]["memory_kind"], SELF_UPDATED_MEMORY_KIND)
            self.assertFalse(hits[0]["transferable"])

    def test_memory_retrieval_only_uses_self_updated_nontransfer_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "mixed_memory.jsonl"
            records = [
                {
                    "signature": "self-memory",
                    "memory_kind": SELF_UPDATED_MEMORY_KIND,
                    "scope_level": "task",
                    "source": "validator_report",
                    "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                    "failure_type": "validator_gap",
                    "repair": "Use this local self-updated repair.",
                    "severity": "hard",
                    "contains_benchmark_label": False,
                    "transferable": False,
                    "copy_policy": SELF_MEMORY_COPY_POLICY,
                },
                {
                    "signature": "curated-rule-shaped",
                    "memory_kind": "curated_rule",
                    "scope_level": "task",
                    "source": "curated_harness_rulebook",
                    "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                    "repair": "Curated rule cards are selected by RuleBook, not memory retrieval.",
                    "contains_benchmark_label": False,
                },
                {
                    "signature": "human-report",
                    "memory_kind": "human_report",
                    "scope_level": "task",
                    "source": "analysis_report",
                    "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                    "repair": "Human analysis reports must not be injected as self memory.",
                    "human_report_derived": True,
                    "contains_benchmark_label": False,
                },
                {
                    "signature": "copied-memory",
                    "memory_kind": "self_updated",
                    "scope_level": "task",
                    "source": "semantic_oracle_report",
                    "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                    "repair": "Transferred memories are out of self-update scope.",
                    "transferable": True,
                    "contains_benchmark_label": False,
                },
                {
                    "signature": "label-bearing",
                    "memory_kind": "self_updated",
                    "scope_level": "task",
                    "source": "semantic_oracle_report",
                    "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                    "repair": "Labels must never enter retrieval.",
                    "contains_benchmark_label": True,
                },
            ]
            memory_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )
            store = FailureMemoryStore(memory_path)
            contract = QuestionContract(
                question_id=1,
                question="standard deviation by year",
                concepts=["Summary Statistics"],
                methods=["std"],
                required_columns=["Damage"],
            )
            hits = store.retrieve(contract, columns=["Damage"], limit=10)
            self.assertEqual([hit["signature"] for hit in hits], ["self-memory"])

    def test_self_memory_recording_marks_records_local_and_nontransferable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FailureMemoryStore(memory_path)
            contract = QuestionContract(
                question_id=7,
                question="Calculate standard deviation of Damage.",
                level="medium",
                concepts=["Summary Statistics"],
                methods=["std"],
                required_columns=["Damage"],
            )
            block = FinalAnswerBlock(
                question_id=7,
                answers={"std": 1.0},
                format_targets={"std": "@std[1.0]"},
                evidence={"method": "sample std with ddof=1", "source_columns": ["Damage"], "sample_size": 3},
                verified=True,
            )
            store.record_branch_outcome(
                contract,
                branch_name="deterministic_skill_branch",
                validation={"passed": True, "errors": [], "warnings": []},
                semantic_validation={"passed": True, "errors": [], "warnings": []},
                runtime={},
                final_block=block,
            )
            [record] = [json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(record["memory_kind"], SELF_UPDATED_MEMORY_KIND)
            self.assertEqual(record["provenance"], "agent_harness_artifact")
            self.assertEqual(record["copy_policy"], SELF_MEMORY_COPY_POLICY)
            self.assertFalse(record["transferable"])
            self.assertFalse(record["curated_rule"])
            self.assertFalse(record["human_report_derived"])

    def test_self_memory_recording_rejects_rule_report_transfer_and_label_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FailureMemoryStore(memory_path)
            blocked_records = [
                {"signature": "label", "contains_benchmark_label": True},
                {"signature": "report", "human_report_derived": True},
                {"signature": "rule", "memory_kind": "curated_rule"},
                {"signature": "transfer", "transferable": True},
            ]
            for record in blocked_records:
                with self.assertRaises(ValueError):
                    store.add(record)
            self.assertFalse(memory_path.exists())

    def test_semantic_oracle_flags_wrong_std_policy_and_scores_branch(self) -> None:
        registry = SemanticOracleRegistry()
        contract = QuestionContract(
            question_id=2,
            question="Calculate the standard deviation of Damage.",
            concepts=["Summary Statistics"],
            methods=["std"],
            target_answers=[TargetAnswer(name="std", type="float")],
            final_format="@std[x]",
        )
        checks = registry.plan(contract)
        self.assertTrue(any(check.id == "oracle.stats.std_sample_ddof1" for check in checks))
        block = FinalAnswerBlock(
            question_id=2,
            answers={"std": 1.0},
            format_targets={"std": "@std[1.0]"},
            evidence={"method": "population standard deviation", "source_columns": ["Damage"], "sample_size": 3},
            verified=True,
        )
        semantic = registry.validate(block, checks)
        self.assertFalse(semantic.passed)
        self.assertIn("semantic_oracle_violation:std_expected_ddof1", semantic.errors)
        schema = Verifier().validate_final_block(block, contract)
        score = Verifier().branch_score(schema, runtime_success=True, semantic_validation=semantic)
        self.assertLess(score, Verifier().branch_score(schema, runtime_success=True))

    def test_controller_persists_rule_and_oracle_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "toy.csv"
            csv.write_text("Year,Damage\n2000,1\n2001,2\n", encoding="utf-8")
            controller = HarnessController(
                HarnessConfig(
                    mode="full",
                    search_policy="hard-only",
                    artifact_root=root / "artifacts",
                    memory_path=root / "memory.jsonl",
                )
            )
            question = {
                "id": "toy",
                "level": "hard",
                "concepts": ["Summary Statistics", "Feature Engineering"],
                "question": "For years 2000 and beyond, calculate the standard deviation of Damage.",
                "constraints": "Only consider years 2000 and beyond.",
                "format": "@std[x] where x is a number rounded to two decimal places.",
                "file_name": "toy.csv",
            }
            context = controller.prepare(question, root)
            assert context.artifacts is not None
            self.assertTrue(Path(context.artifacts.rule_manifest_path).exists())
            self.assertTrue(Path(context.artifacts.oracle_plan_path).exists())
            self.assertIn("# Executable Semantic Oracle Plan", context.prompt)
            branches = controller.branch_prompts(context)
            self.assertEqual(branches[1]["parent_index"], 0)
            self.assertIn("oracle_focus", branches[2])


    def test_memory_retrieval_can_be_disabled_while_recording_path_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "toy.csv").write_text("Damage\n1\n2\n", encoding="utf-8")
            memory_path = root / "memory.jsonl"
            memory_path.write_text(json.dumps({
                "signature": "preexisting",
                "scope_level": "domain",
                "source": "semantic_oracle_report",
                "question_features": {"concepts": ["Summary Statistics"], "methods": ["std"], "columns": ["Damage"]},
                "failure_type": "semantic_oracle_gap",
                "repair": "This should not be injected when retrieval is disabled.",
                "contains_benchmark_label": False,
            }) + "\n", encoding="utf-8")
            controller = HarnessController(HarnessConfig(
                mode="full",
                artifact_root=root / "artifacts",
                memory_path=memory_path,
                memory_retrieval=False,
                memory_recording=True,
            ))
            context = controller.prepare({
                "id": "m",
                "level": "hard",
                "concepts": ["Summary Statistics"],
                "question": "Calculate standard deviation of Damage.",
                "constraints": "",
                "format": "@std[x]",
                "file_name": "toy.csv",
            }, root)
            self.assertEqual(context.memory_hits, [])
            self.assertNotIn("Retrieved Persistent Memories", context.prompt)



if __name__ == "__main__":
    unittest.main()
