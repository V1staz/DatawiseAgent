from __future__ import annotations

from agent_extensions.infiagentbench.question_contract import parse_question_contract
from agent_extensions.infiagentbench.skill_router import route_skills
from agent_extensions.infiagentbench.verifier import verify_execution


def _final_block(name: str = "correlation_coefficient", value: str = "0.90"):
    return {"answers": [{"name": name, "value": value, "format_checked": True}]}


def test_correlation_strongest_requires_abs_r():
    question = {
        "id": 100,
        "question": "Find the strongest Pearson correlation with the target column and return the signed correlation coefficient.",
        "concepts": ["Correlation Analysis"],
        "constraints": "Use Pearson correlation. Round to two decimal places.",
        "format": "@correlation_coefficient[r_value] where r_value is rounded to two decimal places.",
    }
    schema = {"columns": ["target", "a", "b"]}
    contract = parse_question_contract(question, schema)
    skill_plan = route_skills(contract)

    bad_report = verify_execution(
        schema_profile=schema,
        question_contract=contract,
        skill_plan=skill_plan,
        execution_log="corr = df.corr(); best = corr['target'].max()",
        final_block=_final_block(),
    )
    assert bad_report["status"] == "fail"
    assert any(check["name"] == "correlation_abs_r_if_strongest" and check["status"] == "fail" for check in bad_report["checks"])

    good_report = verify_execution(
        schema_profile=schema,
        question_contract=contract,
        skill_plan=skill_plan,
        execution_log="corr = df.corr(); best = corr['target'].drop('target').abs().idxmax(); signed_r = corr.loc[best, 'target']",
        final_block=_final_block(),
    )
    assert not any(check["name"] == "correlation_abs_r_if_strongest" and check["status"] == "fail" for check in good_report["checks"])


def test_feature_engineering_requires_formula_echo_and_head_check():
    question = {
        "id": 5,
        "question": 'Generate a new feature called "FamilySize" by summing the "SibSp" and "Parch" columns. Then calculate the Pearson correlation coefficient between "FamilySize" and "Fare".',
        "concepts": ["Feature Engineering", "Correlation Analysis"],
        "constraints": "Round to two decimal places.",
        "format": "@correlation_coefficient[r_value] where r_value is rounded to two decimal places.",
    }
    schema = {"columns": ["SibSp", "Parch", "Fare"]}
    contract = parse_question_contract(question, schema)

    bad_report = verify_execution(
        schema_profile=schema,
        question_contract=contract,
        skill_plan=route_skills(contract),
        execution_log="df['FamilySize'] = df['SibSp'] + df['Parch']",
        final_block=_final_block(),
    )
    assert any(check["name"] == "feature_first_five_rows_checked" and check["status"] == "warn" for check in bad_report["checks"])

    good_report = verify_execution(
        schema_profile=schema,
        question_contract=contract,
        skill_plan=route_skills(contract),
        execution_log="# FamilySize = SibSp + Parch\ndf['FamilySize'] = df['SibSp'] + df['Parch']\ndf[['SibSp','Parch','FamilySize']].head()",
        final_block=_final_block(),
    )
    assert not any(check["name"].startswith("feature_") and check["status"] == "fail" for check in good_report["checks"])
    assert any(check["name"] == "feature_first_five_rows_checked" and check["status"] == "pass" for check in good_report["checks"])

