from __future__ import annotations

from agent_extensions.infiagentbench.final_block import (
    extract_final_block,
    final_block_to_response,
    validate_final_block,
)


def test_final_block_extraction_and_reformatting():
    text = """
Some notebook summary.
FINAL_BLOCK
{"answers":[{"name":"mean_fare","value":"34.65","raw_value":34.645,"format_checked":true}],"verification":{"passed":true,"warnings":[]}}
"""
    contract = {
        "answer_names": ["mean_fare"],
        "format_requirements": {"decimal_places": 2, "quote_string": None, "container_type": "numeric"},
    }

    block = extract_final_block(text)
    response, validation = final_block_to_response(block, contract)

    assert validation["passed"]
    assert response == "@mean_fare[34.65]"


def test_formatting_verifier_rejects_bad_decimals_and_missing_quotes():
    numeric_contract = {
        "answer_names": ["mean_fare"],
        "format_requirements": {"decimal_places": 2, "quote_string": None, "container_type": "numeric"},
    }
    bad_numeric = {"answers": [{"name": "mean_fare", "value": "34.6", "format_checked": True}]}

    assert not validate_final_block(bad_numeric, numeric_contract)["passed"]

    string_contract = {
        "answer_names": ["station"],
        "format_requirements": {"decimal_places": None, "quote_string": True, "container_type": "string"},
    }
    bad_string = {"answers": [{"name": "station", "value": "ABC", "format_checked": True}]}

    assert not validate_final_block(bad_string, string_contract)["passed"]

