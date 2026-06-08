from datawiseagent.common.types import ConvertType, LLMResult
from datawiseagent.common.types.cell import CodeCell, MarkdownCell, NotebookCell


def test_final_answer_block_python_fence_is_not_executable_code():
    response = LLMResult(
        role="assistant",
        content="""```python
final_answer_block = {
  "answers": {"answer": 1},
  "format_targets": {"answer": "@answer[1]"},
  "verified": false,
  "validator_report_id": null
}
```""",
    )

    cells = NotebookCell.llm_result_convert(response, parse_mode=ConvertType.CONVERT_CELLS)

    assert len(cells) == 1
    assert isinstance(cells[0], MarkdownCell)
    assert not any(isinstance(cell, CodeCell) for cell in cells)
