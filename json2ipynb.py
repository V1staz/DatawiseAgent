import argparse
import json
from pathlib import Path


def _as_lines(text: str):
    if text is None:
        return []
    if isinstance(text, list):
        return text
    # Preserve newlines for ipynb stream output.
    return text.splitlines(keepends=True)


def _new_markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _as_lines(source),
    }


def _new_code_cell(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "source": _as_lines(source),
        "outputs": [],
    }


def _append_stream_output(code_cell: dict, text: str, name: str = "stdout"):
    if text is None:
        return
    code_cell["outputs"].append(
        {
            "output_type": "stream",
            "name": name,
            "text": _as_lines(text),
        }
    )


def convert_session_json_to_ipynb(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chat_history = data.get("chat_history", {})
    init_cells = chat_history.get("init_cells", [])
    cells = chat_history.get("cells", [])

    nb_cells = []
    last_code_cell = None

    def consume_cell(cell: dict):
        nonlocal last_code_cell
        cell_type = cell.get("cell_type")
        content = cell.get("content", "")

        if cell_type in ("markdown", "user_markdown", "step_markdown"):
            nb_cells.append(_new_markdown_cell(content))
            last_code_cell = None
            return

        if cell_type == "code":
            code_cell = _new_code_cell(content)
            nb_cells.append(code_cell)
            last_code_cell = code_cell
            return

        if cell_type == "codeoutput":
            if last_code_cell is None:
                # No code cell to attach to; create a standalone markdown cell.
                nb_cells.append(_new_markdown_cell(content))
                return
            code_result = cell.get("code_result", {})
            output_text = code_result.get("output", "") or content
            _append_stream_output(last_code_cell, output_text)
            return

    for cell in init_cells:
        consume_cell(cell)

    for cell in cells:
        consume_cell(cell)

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert DatawiseAgent session JSON to ipynb.")
    parser.add_argument("input_json", type=Path, help="Path to session JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output .ipynb file (default: same name as input).",
    )
    args = parser.parse_args()

    input_path: Path = args.input_json
    if args.output is None:
        output_path = input_path.with_suffix(".ipynb")
    else:
        output_path = args.output

    convert_session_json_to_ipynb(input_path, output_path)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
