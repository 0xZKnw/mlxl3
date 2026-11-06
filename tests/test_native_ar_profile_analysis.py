"""Disposable xctrace fixtures exercise the real streaming analysis entry point."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "ar_profile_analysis", SCRIPTS / "analyze_native_ar_profile.py"
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
sys.path.pop(0)


def table(path, name, columns, rows):
    schema = "".join(
        "<col><mnemonic>"
        + c
        + "</mnemonic><engineering-type>"
        + A.TYPES[c]
        + "</engineering-type></col>"
        for c in columns
    )
    path.write_text(
        '<trace-query-result><node><schema name="'
        + name
        + '">'
        + schema
        + "</schema>"
        + "".join("<row>" + r + "</row>" for r in rows)
        + "</node></trace-query-result>"
    )


def scalar(name, value):
    return "<" + A.TYPES[name] + ">" + escape(str(value)) + "</" + A.TYPES[name] + ">"


def process():
    return "<process><pid>7</pid></process>"


def thread():
    return "<thread><tid>11</tid>" + process() + "</thread>"


def fixtures(root):
    cols = ["time", "thread", "process", "thread-state", "weight", "stack"]
    table(
        root / "time-profile.xml",
        "time-profile",
        cols,
        [
            scalar("time", 100)
            + thread()
            + process()
            + scalar("thread-state", "Running")
            + scalar("weight", 1000000)
            + '<tagged-backtrace><frame name="pread"/><frame name="caller"/><frame name="caller"/></tagged-backtrace>'
        ],
    )
    cols = [
        "start",
        "duration",
        "process",
        "thread",
        "call",
        "cputime",
        "waittime",
        "arg1",
        "arg4",
        "return",
        "errno",
    ]
    table(
        root / "syscall.xml",
        "syscall",
        cols,
        [
            scalar("start", 200)
            + scalar("duration", 1000)
            + process()
            + thread()
            + scalar("call", "BSC_preadv")
            + scalar("cputime", 400)
            + scalar("waittime", 600)
            + scalar("arg1", 4)
            + scalar("arg4", 128)
            + scalar("return", 1248012)
            + scalar("errno", 0)
        ],
    )
    cols = ["start", "duration", "process", "syscall", "type", "return", "bytes", "errno"]
    table(
        root / "FsSyscall.xml",
        "FsSyscall",
        cols,
        [
            scalar("start", 200)
            + scalar("duration", 1000)
            + process()
            + scalar("syscall", "BSC_preadv")
            + scalar("type", "Read")
            + scalar("return", 1248012)
            + scalar("bytes", 1248012)
            + scalar("errno", 0)
        ],
    )
    cols = [
        "start",
        "duration",
        "process",
        "thread",
        "state",
        "cputime",
        "waittime",
        "thermal-throttled",
    ]
    rows = [
        scalar("start", 0)
        + scalar("duration", 200)
        + process()
        + thread()
        + scalar("state", "Unknown")
        + "<sentinel/><sentinel/><sentinel/>",
        scalar("start", 200)
        + scalar("duration", 1000)
        + process()
        + thread()
        + scalar("state", "Waiting")
        + scalar("cputime", 400)
        + scalar("waittime", 600)
        + scalar("thermal-throttled", 0),
    ]
    table(root / "thread-state.xml", "thread-state", cols, rows)


def test_real_summary_keeps_cpu_samples_waits_and_logical_bytes_distinct(tmp_path):
    fixtures(tmp_path)
    report = A.summarize_cpu(tmp_path, 7, tmp_path / "out")
    assert report["samples"]["rows"] == 1
    assert report["samples"]["leaf_running_weight_ns"] == [("pread", 1000000)]
    assert dict(report["samples"]["inclusive_running_weight_ns"])["caller"] == 1000000
    assert report["syscalls"][0]["read_return_bytes"] == 1248012
    assert report["syscalls"][0]["cpu_ns_known"] == 400
    assert report["syscalls"][0]["wait_ns_known"] == 600
    assert report["filesystem"]["interval_union_ns"] == 1000
    assert report["scheduler"]["known_state_thread_ns"] == {"Waiting": 1000}
    assert "Unknown" not in report["scheduler"]["known_state_thread_ns"]


def test_backward_refs_and_negative_trace_relative_clock_are_preserved(tmp_path):
    p = tmp_path / "x.xml"
    table(
        p,
        "toy",
        ["start", "process"],
        [
            '<start-time>-1</start-time><process id="1"><pid>7</pid></process>',
            '<start-time>2</start-time><process ref="1"/>',
        ],
    )
    rows = list(A.table_rows(p, "toy", {"start", "process"}))
    assert A.number(rows[0]["start"], signed=True) == -1
    assert rows[1]["process"]["pid"] == 7


def test_same_row_earlier_sibling_definitions_resolve_in_later_threads(tmp_path):
    p = tmp_path / "x.xml"
    table(
        p,
        "toy",
        ["process", "thread"],
        [
            '<process id="1"><pid>7</pid></process><thread id="2"><tid>11</tid><process ref="1"/></thread>',
            '<process ref="1"/><thread ref="2"/>',
        ],
    )
    row = next(A.table_rows(p, "toy", {"process", "thread"}))
    assert row["process"]["pid"] == row["thread"]["pid"] == 7


def test_some_unavailable_real_backtraces_are_counted_without_discarding_their_weight(tmp_path):
    fixtures(tmp_path)
    p = tmp_path / "time-profile.xml"
    s = p.read_text()
    extra = "<row>" + scalar("time", 200) + thread() + process()
    extra += scalar("thread-state", "Running") + scalar("weight", 2000000) + "<sentinel/></row>"
    p.write_text(s.replace("</node>", extra + "</node>"))
    report = A.summarize_cpu(tmp_path, 7, tmp_path / "out")
    assert report["samples"]["unattributed_running_weight_ns"] == 2000000
    assert report["samples"]["weight_ns_by_state"]["Running"] == 3000000
    assert dict(report["samples"]["leaf_running_weight_ns"])["<unavailable backtrace>"] == 2000000


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('ref="1"', 'ref="999"'),
        lambda s: s.replace("<start-time>2</start-time>", "<string>2</string>"),
        lambda s: s.replace("<engineering-type>start-time", "<engineering-type>duration"),
        lambda s: s.replace('<schema name="toy"', '<schema name="wrong"'),
        lambda s: s.replace("<start-time>2</start-time>", ""),
        lambda s: s[:-10],
        lambda s: s.replace("<trace-query-result>", "<trace-query-result><!DOCTYPE x>"),
        lambda s: s.replace(
            '<process ref="1"/>', '<process id="1"><pid>7</pid></process><process ref="1"/>'
        ),
    ],
)
def test_invalid_incomplete_or_ambiguous_inputs_never_pass(tmp_path, change):
    p = tmp_path / "x.xml"
    table(
        p,
        "toy",
        ["start", "process"],
        [
            '<start-time>1</start-time><process id="1"><pid>7</pid></process>',
            '<start-time>2</start-time><process ref="1"/>',
        ],
    )
    p.write_text(change(p.read_text()))
    with pytest.raises((ValueError, A.ET.ParseError)):
        list(A.table_rows(p, "toy", {"start", "process"}))


def test_wrong_pid_and_empty_stacks_are_rejected(tmp_path):
    fixtures(tmp_path)
    with pytest.raises(ValueError, match="empty target"):
        A.summarize_cpu(tmp_path, 8, tmp_path / "wrong")
    p = tmp_path / "time-profile.xml"
    p.write_text(
        p.read_text().replace(
            '<frame name="pread"/><frame name="caller"/><frame name="caller"/>', ""
        )
    )
    with pytest.raises(ValueError, match="empty running"):
        A.summarize_cpu(tmp_path, 7, tmp_path / "empty")


def test_unavailable_duration_cells_remain_unknown_instead_of_zero(tmp_path):
    fixtures(tmp_path)
    for name in ("syscall.xml", "thread-state.xml"):
        p = tmp_path / name
        p.write_text(
            p.read_text()
            .replace("<duration-on-core>400</duration-on-core>", "<sentinel/>")
            .replace("<duration-waiting>600</duration-waiting>", "<sentinel/>")
        )
    report = A.summarize_cpu(tmp_path, 7, tmp_path / "out")
    assert report["syscalls"][0]["cpu_ns_known"] is None
    assert report["syscalls"][0]["cpu_duration_cells"] == 0
    assert report["scheduler"]["oncore_ns_known_by_thread"][11] is None
    assert report["scheduler"]["state_duration_ns_by_thread"][11] == {"Waiting": 1000}
    assert report["scheduler"]["thread_interval_overlap_ns"][11] == 0


def test_cli_failed_parse_writes_explicit_failure_without_success_report(tmp_path):
    fixtures(tmp_path)
    p = tmp_path / "syscall.xml"
    p.write_text(p.read_text()[:-20])
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "analyze_native_ar_profile.py"),
            "--directory",
            str(tmp_path),
            "--pid",
            "7",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert json.loads((out / "analysis-status.json").read_text())["status"] == "failed"
    assert not (out / "cpu-summary.json").exists()
