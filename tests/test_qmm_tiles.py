"""CPU-only regressions for the physical QMM comparison protocol."""

import importlib.util
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qmm_tiles", ROOT / "native/check_qmm_tiles.py")
qmm_tiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qmm_tiles)


def test_cli_rejects_empty_codec_outputs(tmp_path):
    codec = tmp_path / "empty-codec"
    codec.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    data = [] if request['op'] == 'mlx-linear' else [[], []]\n"
        "    print(json.dumps({'data': data}), flush=True)\n"
    )
    codec.chmod(0o755)
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native/check_qmm_tiles.py"),
            str(codec),
            str(codec),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode != 0, "empty codec outputs must not certify QMM parity"
    assert "output" in run.stderr.lower()
    assert "all FP16 outputs finite and bit-identical" not in run.stdout
    if output.exists():
        assert not json.loads(output.read_text()).get("cases")


def write_synthetic_codec(path, *, nonfinite=False):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    widths = request['widths']\n"
        "    rows = len(request['x']) // (len(request['suh']) // len(widths))\n"
        f"    groups = [[{0x7C00 if nonfinite else 0}] * (rows * w) for w in widths]\n"
        "    data = groups[0] if request['op'] == 'mlx-linear' else groups\n"
        "    print(json.dumps({'data': data}), flush=True)\n"
    )
    path.chmod(0o755)


def test_cli_checks_all_linear_grouped_and_fallback_values(tmp_path):
    codec = tmp_path / "valid-codec"
    write_synthetic_codec(codec)
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native/check_qmm_tiles.py"),
            str(codec),
            str(codec),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    cases = json.loads(output.read_text())["cases"]
    assert len(cases) == 64
    assert sum(case["finite_fp16_outputs"] for case in cases) == 1_887_232
    assert all(
        case["finite_fp16_outputs"] == case["rows"] * sum(case["widths"]) > 0 and case["bit_exact"]
        for case in cases
    )


def test_cli_rejects_a_nonfinite_candidate(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_synthetic_codec(baseline)
    write_synthetic_codec(candidate, nonfinite=True)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native/check_qmm_tiles.py"),
            str(baseline),
            str(candidate),
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode != 0
    assert "output FP16 words must be finite" in run.stderr


def test_valid_single_and_ragged_group_outputs():
    assert qmm_tiles.validate_output([0, 0x8000, 0x3C00, 0xBC00], 2, [2]) == [
        0,
        0x8000,
        0x3C00,
        0xBC00,
    ]
    assert qmm_tiles.validate_output([[0, 1], [2, 3, 4, 5]], 2, [1, 2]) == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize(
    "data,rows,widths",
    [
        ([], 1, [1]),
        ([0], 2, [1]),
        ([0, 0], 1, [1]),
        ([[0]], 1, [1]),
        (None, 1, [1]),
        ({"0": 0}, 1, [1]),
        ([[0]], 1, [1, 2]),
        ([[0], [0, 0], [0]], 1, [1, 2]),
        ([[], [0, 0]], 1, [1, 2]),
        ([[0], [0]], 1, [1, 2]),
        ([[0], [0, 0, 0]], 1, [1, 2]),
        ([0, [0, 0]], 1, [1, 2]),
        ([0], 0, [1]),
        ([0], -1, [1]),
        ([], 1, []),
        ([], 1, [0]),
        ([], 1, [-1]),
    ],
)
def test_rejects_invalid_output_shapes(data, rows, widths):
    with pytest.raises(ValueError, match="output"):
        qmm_tiles.validate_output(data, rows, widths)


@pytest.mark.parametrize(
    "word", [True, False, 0.0, "0", None, -1, 65536, 0x7C00, 0xFC00, 0x7C01, 0x7FFF, 0xFC01, 0xFFFF]
)
def test_rejects_invalid_or_nonfinite_words(word):
    with pytest.raises(ValueError, match="output"):
        qmm_tiles.validate_output([word], 1, [1])


def test_all_fp16_bit_patterns_match_an_independent_decoder():
    finite = []
    for word in range(65536):
        expected = math.isfinite(struct.unpack("<e", struct.pack("<H", word))[0])
        if expected:
            finite.append(word)
        else:
            with pytest.raises(ValueError, match="finite"):
                qmm_tiles.validate_output([word], 1, [1])
    assert qmm_tiles.validate_output(finite, 1, [len(finite)]) == finite


@pytest.mark.parametrize(
    "behavior,message",
    [
        ("silent", "timed out"),
        ("partial", "timed out"),
        ("closed_stdin", "closed stdin"),
        ("eof", "complete JSON"),
        ("invalid", "invalid JSON"),
        ("error", "fixture codec failed"),
        ("missing", "no data"),
    ],
)
def test_cli_codec_protocol_failure_is_recorded_and_reaped(tmp_path, behavior, message):
    codec = tmp_path / "failing-codec"
    codec.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "with Path(__file__).with_suffix('.pids').open('a') as f: f.write(str(os.getpid()) + '\\n')\n"
        f"behavior = {behavior!r}\n"
        "if behavior == 'closed_stdin': os.close(0); time.sleep(60)\n"
        "for line in sys.stdin:\n"
        "    if behavior == 'silent': time.sleep(60)\n"
        "    if behavior == 'partial': os.write(1, b'{\"data\":'); time.sleep(60)\n"
        "    if behavior == 'eof': sys.exit(0)\n"
        "    if behavior == 'invalid': print('invalid JSON', flush=True)\n"
        "    if behavior == 'error': print(json.dumps({'error': 'fixture codec failed'}), flush=True)\n"
        "    if behavior == 'missing': print('{}', flush=True)\n"
    )
    codec.chmod(0o755)
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native/check_qmm_tiles.py"),
            str(codec),
            str(codec),
            "--output",
            str(output),
            "--request-timeout",
            "0.3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert run.returncode != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is None
    assert report["cases"] == [] and message in report["error"]["message"]
    for word in codec.with_suffix(".pids").read_text().splitlines():
        pid = int(word)
        deadline = time.monotonic() + 2
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                pytest.fail("QMM comparison left a codec alive")
            time.sleep(0.01)


def test_cli_second_codec_startup_failure_is_saved(tmp_path):
    codec = tmp_path / "first-codec"
    codec.write_text(f"#!{sys.executable}\nimport sys\nfor line in sys.stdin: pass\n")
    codec.chmod(0o755)
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "native/check_qmm_tiles.py"),
            str(codec),
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert run.returncode != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is None
    assert report["error"]["type"] == "FileNotFoundError" and report["cases"] == []
