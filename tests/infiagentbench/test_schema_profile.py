from __future__ import annotations

from agent_extensions.infiagentbench.schema_profile import profile_csv


def test_schema_profile_detects_core_fields(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "id,event_date,value,category\n"
        "1,2024-01-01,10,A\n"
        "2,2024-01-02,20,B\n"
        "3,2024-01-03,,A\n"
        "4,2024-01-04,40,C\n",
        encoding="utf-8",
    )

    profile = profile_csv(csv_path)

    assert profile["shape"] == [4, 4]
    assert profile["columns"] == ["id", "event_date", "value", "category"]
    assert profile["missing_count"]["value"] == 1
    assert profile["missing_ratio"]["value"] == 0.25
    assert profile["numeric_summary"]["value"]["min"] == 10
    assert "category" in profile["possible_categorical_columns"]
    assert any(item["column"] == "event_date" for item in profile["possible_datetime_columns"])
    assert any(item["column"] == "id" for item in profile["possible_id_columns"])
    assert profile["normalized_column_name_mapping"]["event_date"] == "event_date"

