from __future__ import annotations

from agent_extensions.infiagentbench.question_contract import parse_question_contract


def test_contract_parser_extracts_feature_correlation_constraints():
    question = {
        "id": 5,
        "question": 'Generate a new feature called "FamilySize" by summing the "SibSp" and "Parch" columns. Then, calculate the Pearson correlation coefficient (r) between the "FamilySize" and "Fare" columns.',
        "concepts": ["Feature Engineering", "Correlation Analysis"],
        "constraints": "Create a new column 'FamilySize' that is the sum of 'SibSp' and 'Parch' for each row.\nCalculate the Pearson correlation coefficient between 'FamilySize' and 'Fare'\nDo not perform any further data cleaning or preprocessing steps before calculating the correlation.",
        "format": '@correlation_coefficient[r_value]\nwhere "r_value" is the Pearson correlation coefficient between \'FamilySize\' and \'Fare\', a number between -1 and 1, rounded to two decimal places.',
    }
    schema = {"columns": ["SibSp", "Parch", "Fare"]}

    contract = parse_question_contract(question, schema)

    assert contract["answer_names"] == ["correlation_coefficient"]
    assert set(contract["required_columns"]) >= {"SibSp", "Parch", "Fare"}
    assert "feature_engineering" in contract["task_types"]
    assert "correlation_analysis" in contract["task_types"]
    assert "final_formatting" in contract["task_types"]
    assert contract["decimal_places"] == 2
    assert "pearson" in contract["required_methods"]
    assert any("do not perform" in item.lower() for item in contract["forbidden_actions"])
    assert contract["feature_definitions"][0]["formula"] == "FamilySize = SibSp + Parch"

