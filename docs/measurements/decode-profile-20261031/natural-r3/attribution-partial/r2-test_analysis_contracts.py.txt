"""Finite checks of actual reader/selection syntax; no profiling or GPU."""

import ast
import xml.etree.ElementTree as E
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("analyze-partial.py")


def execute_actual_node(node, namespace):
    code = compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec")
    exec(code, namespace)  # noqa: S102 -- trusted repository AST, disposable fixture namespace.


def reader(source):
    tree = ast.parse(source)
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "table"
    )
    namespace = {"E": E}
    execute_actual_node(node, namespace)
    return namespace["table"]


GOOD = '<trace-query-result><node><schema name="test"><col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col></schema><row><start-time id="1">2</start-time><duration id="2">3</duration></row><row><start-time ref="1"/><duration ref="2"/></row></node></trace-query-result>'


def test_reader_literal_alias_oracle(tmp_path):
    path = tmp_path / "table.xml"
    path.write_text(GOOD)
    rows, _ = reader(SOURCE.read_text())(path)
    assert len(rows) == 2
    assert [[int(row[key].text) for key in ("start", "duration")] for row in rows] == [
        [2, 3],
        [2, 3],
    ]


@pytest.mark.parametrize(
    "bad",
    [
        GOOD.replace('duration id="2"', 'duration id="1"'),
        GOOD.replace('duration ref="2"', 'duration ref="missing"'),
        GOOD.replace('duration ref="2"', 'duration ref="1"'),
        GOOD.replace('<duration ref="2"/>', ""),
        GOOD.replace("<row>", "<notrow>").replace("</row>", "</notrow>"),
        GOOD.replace('<schema name="test">', '<schema name="extra"/><schema name="test">'),
    ],
)
def test_reader_refuses_corruption_and_multiple_schemas(tmp_path, bad):
    path = tmp_path / "bad.xml"
    path.write_text(bad)
    with pytest.raises((AssertionError, E.ParseError)):
        reader(SOURCE.read_text())(path)


def select_from_actual_syntax(source, events, first, last):
    tree = ast.parse(source)
    namespace = {
        "ev": events,
        "first": first,
        "last": last,
        "span": lambda event: (event["a"], event["b"]),
    }
    for variable in ("covered_steps", "steps"):
        statement = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == variable
        )
        execute_actual_node(statement, namespace)
    return [event["data"]["step"] for event in namespace["steps"]]


def test_selection_and_threshold_mutation_are_nontrivial():
    events = [
        {"name": "output_step", "data": {"step": index}, "a": index, "b": index + 1}
        for index in range(1, 90)
    ]
    source = SOURCE.read_text()
    assert select_from_actual_syntax(source, events, 10, 70) == list(range(16, 56))
    assert select_from_actual_syntax(source, events, 69, 90) == list(range(69, 90))
    assert select_from_actual_syntax(source, events, 89, 89) == []
    mutant = source.replace('e["data"]["step"] >= 16', 'e["data"]["step"] >= 1')
    assert mutant != source
    assert select_from_actual_syntax(mutant, events, 10, 70) != list(range(16, 56))


def test_minimum_16_assertion_acceptance_boundary():
    tree = ast.parse(SOURCE.read_text())
    statement = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assert)
        and isinstance(node.msg, ast.Constant)
        and node.msg.value == "fewer than 16 complete transitions"
    )
    for count in (0, 1, 5, 15):
        with pytest.raises(AssertionError):
            execute_actual_node(statement, {"n": count})
    for count in (16, 40):
        execute_actual_node(statement, {"n": count})


def test_missing_schema_guard_mutation_is_detected(tmp_path):
    path = tmp_path / "double.xml"
    extra = '<schema name="extra"><col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col></schema>'
    path.write_text(GOOD.replace('<schema name="test">', extra + '<schema name="test">'))
    source = SOURCE.read_text()
    with pytest.raises(AssertionError):
        reader(source)(path)
    mutant = source.replace(
        'assert len(schemas) == 1, "export must contain exactly one schema"',
        "assert len(schemas) >= 1",
    )
    assert mutant != source
    # Removing only the guard silently accepts an ambiguous two-schema export.
    rows, _ = reader(mutant)(path)
    assert len(rows) == 2 and list(rows[0]) == ["start", "duration"]
