#!/usr/bin/env python3
"""Build Colab notebooks from Markdown source files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "source" / "SLM_Workshop_Live_Finetune.md"
OUTPUT = ROOT / "notebooks" / "SLM_Workshop_Live_Finetune.ipynb"


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def add_cell(cells: list[dict], cell_type: str, content: list[str]) -> None:
    while content and content[0].strip() == "":
        content.pop(0)
    while content and content[-1].strip() == "":
        content.pop()
    if not content:
        return

    cell: dict = {
        "cell_type": cell_type,
        "metadata": {},
        "source": content,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)


def parse_markdown_notebook(markdown: str) -> list[dict]:
    cells: list[dict] = []
    markdown_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False
    code_is_python = False

    for raw_line in source_lines(markdown):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            fence_info = stripped[3:].strip().lower()
            if not in_code:
                in_code = True
                code_is_python = fence_info in {"python", "py"}
                if code_is_python:
                    add_cell(cells, "markdown", markdown_buffer)
                    markdown_buffer = []
                    code_buffer = []
                else:
                    markdown_buffer.append(raw_line)
                continue

            if code_is_python:
                add_cell(cells, "code", code_buffer)
                code_buffer = []
            else:
                markdown_buffer.append(raw_line)
            in_code = False
            code_is_python = False
            continue

        if in_code and code_is_python:
            code_buffer.append(raw_line)
        else:
            markdown_buffer.append(raw_line)

    if in_code and code_is_python:
        markdown_buffer.extend(["```python\n", *code_buffer])

    add_cell(cells, "markdown", markdown_buffer)
    return cells


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    notebook = build_notebook(parse_markdown_notebook(markdown))
    OUTPUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
