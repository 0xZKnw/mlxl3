"""Real pipe/process boundaries for the benchmark transport; no GPU needed."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack

import pytest

from scripts import native_json_process as protocol


def spawn(source):
    return protocol.JsonProcess([sys.executable, "-u", "-c", source], stderr=subprocess.DEVNULL)


def assert_reaped(child):
    assert child.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child.process.pid, 0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "NaN", "inf", "-inf", "1e400", "", "word"])
def test_invalid_durations_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive and finite"):
        protocol.positive_seconds(value)


@pytest.mark.parametrize("value,expected", [("0.001", 0.001), ("600", 600), ("1e3", 1000)])
def test_finite_positive_durations(value, expected):
    assert protocol.positive_seconds(value) == expected


def test_silence_has_a_deadline_and_child_is_reaped():
    with spawn("import time; print('{}', flush=True); time.sleep(60)") as child:
        assert child.receive(time.monotonic() + 3) == {}
        with pytest.raises(TimeoutError, match="waiting for JSON"):
            child.receive(time.monotonic() + 0.15)
    assert_reaped(child)


def test_partial_response_has_a_deadline():
    with (
        spawn("import os, time; os.write(1, b'{\"data\":'); time.sleep(60)") as child,
        pytest.raises(TimeoutError, match="waiting for JSON"),
    ):
        child.receive(time.monotonic() + 0.3)
    assert_reaped(child)


def test_event_stream_cannot_extend_the_request_deadline():
    source = "import time\nwhile True:\n print('{}', flush=True)\n time.sleep(0.005)"
    with spawn(source) as child:
        assert child.receive(time.monotonic() + 3) == {}
        deadline = time.monotonic() + 0.15
        count = 0
        with pytest.raises(TimeoutError, match="waiting for JSON"):
            while True:
                child.receive(deadline)
                count += 1
        assert count > 0
    assert_reaped(child)


def test_blocked_stdin_write_is_bounded():
    with spawn("import time; print('{}', flush=True); time.sleep(60)") as child:
        assert child.receive(time.monotonic() + 3) == {}
        with pytest.raises(TimeoutError, match="writing request"):
            child.send({"payload": "x" * (4 * 1024 * 1024)}, time.monotonic() + 0.15)
    assert_reaped(child)


def test_closed_stdin_does_not_mask_the_error_or_leak_child():
    with spawn("import os, time; os.close(0); print('{}', flush=True); time.sleep(60)") as child:
        assert child.receive(time.monotonic() + 3) == {}
        with pytest.raises(RuntimeError, match="closed stdin"):
            child.send({"request": 1}, time.monotonic() + 1)
    assert_reaped(child)


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"invalid\n", "invalid JSON"),
        (b"[]\n", "must be an object"),
        (b'{"data":', "complete JSON"),
        (b'{"data":"\xff"}\n', "invalid JSON"),
    ],
)
def test_invalid_and_truncated_responses_fail(payload, message):
    with (
        spawn(f"import os; os.write(1, {payload!r})") as child,
        pytest.raises(RuntimeError, match=message),
    ):
        child.receive(time.monotonic() + 3)
    assert_reaped(child)


def test_chunked_utf8_and_multiple_buffered_responses():
    payload = (json.dumps({"text": "é"}, ensure_ascii=False) + '\n{"second":2}\n').encode()
    split = payload.index(b"\xc3") + 1
    with spawn(
        f"import os; os.write(1, {payload[:split]!r}); os.write(1, {payload[split:]!r})"
    ) as child:
        assert child.receive(time.monotonic() + 3) == {"text": "é"}
        assert child.receive(time.monotonic() + 3) == {"second": 2}
        with pytest.raises(RuntimeError, match="complete JSON"):
            child.receive(time.monotonic() + 3)
    assert_reaped(child)


def test_unterminated_oversized_response_is_rejected(monkeypatch):
    monkeypatch.setattr(protocol, "MAX_JSON_LINE", 64)
    with (
        spawn("import os, time; os.write(1, b'x' * 65); time.sleep(60)") as child,
        pytest.raises(RuntimeError, match="size limit"),
    ):
        child.receive(time.monotonic() + 3)
    assert_reaped(child)


def test_child_ignoring_sigterm_is_killed_and_reaped():
    source = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('{}', flush=True); time.sleep(60)"
    with spawn(source) as child:
        assert child.receive(time.monotonic() + 3) == {}
    assert child.process.returncode == -signal.SIGKILL
    assert_reaped(child)


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_exception_and_interrupt_cleanup(exception):
    with (
        pytest.raises(exception),
        spawn("import time; print('{}', flush=True); time.sleep(60)") as child,
    ):
        child.receive(time.monotonic() + 3)
        raise exception("fixture interruption")
    assert_reaped(child)


def test_second_startup_failure_closes_the_first_child(tmp_path):
    with pytest.raises(FileNotFoundError), ExitStack() as stack:
        child = stack.enter_context(spawn("import time; print('{}', flush=True); time.sleep(60)"))
        child.receive(time.monotonic() + 3)
        stack.enter_context(protocol.JsonProcess([str(tmp_path / "absent-codec")]))
    assert_reaped(child)
