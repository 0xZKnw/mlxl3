"""Run after cargo build: Python/Rust reasoning fragments must match exactly."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Execute the actual pure parser source without importing the Metal runtime.
source = ROOT / "src/mlxl3/cli.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
parser_nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"ThinkingSplitter", "_prompt_prefills_thinking"}]
assert len(parser_nodes) == 2
namespace = {}
exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(source), "exec"), namespace)
ThinkingSplitter = namespace["ThinkingSplitter"]
BINARY = Path(os.environ.get(
    "MLXL3_NATIVE_BINARY", ROOT / "target/debug/mlxl3-rs"
))


def fragments(chunks: list[str], prompt: str = "") -> list[tuple[str, str]]:
    splitter = ThinkingSplitter()
    splitter.configure_for_prompt(prompt)
    expected = []
    for chunk in chunks:
        expected.extend(splitter.feed(chunk))
    expected.extend(splitter.finish())
    assert BINARY.is_file(), "Build the native binary with cargo build first"
    process = subprocess.run(
        [str(BINARY), "split", "--prompt", prompt], input=json.dumps(chunks),
        capture_output=True, text=True, check=True, timeout=5,
    )
    actual = [(f["channel"], f["text"]) for f in json.loads(process.stdout)]
    assert actual == expected
    return actual


@pytest.mark.parametrize("open_marker,close_marker", [
    ("<think>", "</think>"), ("<|channel>thought", "<channel|>"),
])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_streaming_boundaries(open_marker: str, close_marker: str, newline: str) -> None:
    text = f"{open_marker}{newline}réfléchis 🦀{close_marker}{newline}bonjour"
    chunkings = [[text], list(text)] + [[text[:i], text[i:]] for i in range(len(text) + 1)]
    for chunks in chunkings:
        result = fragments(chunks)
        assert "".join(t for c, t in result if c == "thinking") == "réfléchis 🦀"
        assert "".join(t for c, t in result if c == "answer") == "bonjour"


@pytest.mark.parametrize("chunks,prompt,expected", [
    (["<think>\r", ""], "", [("thinking", "\r")]),
    (["<think>\r", "x"], "", [("thinking", "\rx")]),
    (["<think>\r", "<thi"], "", [("thinking", "\r<thi")]),
    (["<think>\r", "\n"], "", []),
    (["\r"], "assistant\n<think>", [("thinking", "\r")]),
    (["raison</think>\r", "\nanswer<thi"], "assistant\n<think>\n", [
        ("thinking", "raison"), ("answer", "answer"), ("answer", "<thi"),
    ]),
])
def test_incomplete_and_prefilled_streams(chunks, prompt, expected) -> None:
    result = fragments(chunks, prompt)
    assert result == expected
