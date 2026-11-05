"""Independent loss oracles and real subprocess failure checks, without a GPU."""

import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import eval_lfm25_quality as quality


@pytest.mark.parametrize("tokens,window", [(0, 1), (1, 0), (-1, 2), (3, 2), (2, -1)])
def test_empty_or_partial_evaluations_rejected(tokens, window):
    with pytest.raises(ValueError):
        quality.window_count(tokens, window)


def test_losses_match_independent_binary_distribution_and_offset_invariance():
    nll, teacher_nll, kl = quality.losses([math.log(3), 0], [0, 0], 0)
    assert nll == pytest.approx(math.log(4 / 3))
    assert teacher_nll == pytest.approx(math.log(2))
    assert kl == pytest.approx(0.5 * math.log(2 / 3) + 0.5 * math.log(2))
    rng = np.random.default_rng(184)
    for size in range(1, 17):
        actual, teacher = rng.normal(size=(2, size))
        for target in range(size):
            result = quality.losses(actual, teacher, target)
            oracle = np.logaddexp.reduce(actual) - actual[target]
            assert result[0] == pytest.approx(oracle)
            assert result == pytest.approx(quality.losses(actual + 2048, teacher - 1024, target))
            assert quality.losses(actual, actual, target)[2] == 0


@pytest.mark.parametrize(
    "actual,teacher,target",
    [
        ([], [], 0),
        ([float("nan")], [0], 0),
        ([float("inf")], [0], 0),
        ([[0]], [0], 0),
        ([0], [0, 0], 0),
        ([0], [0], -1),
        ([0], [0], 1),
        ([0], [0], True),
    ],
)
def test_invalid_loss_inputs_rejected(actual, teacher, target):
    with pytest.raises(ValueError):
        quality.losses(actual, teacher, target)


def fake_native(tmp_path, mode):
    executable = tmp_path / "native"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "print(os.getpid(), file=sys.stderr, flush=True)\n"
        f"mode = {mode!r}\n"
        "tokens = list(map(int,sys.argv[sys.argv.index('--tokens')+1].split(',')))\n"
        "if mode == 'silent': time.sleep(60)\n"
        "if mode == 'short': tokens = tokens[:-1]\n"
        "if mode == 'extra': tokens.append(tokens[-1])\n"
        "for token in tokens:\n"
        "    if mode == 'malformed': print('bad JSON', flush=True); break\n"
        "    bits = [0x7e00,0] if mode == 'nonfinite' or (mode == 'first-nonfinite' and token == 0) else [True,0] if mode == 'boolean' else [0,0]\n"
        "    print(json.dumps({'token':999 if mode == 'wrong-token' else token, 'logits':bits}), flush=True)\n"
        "if mode == 'exit-error': sys.exit(2)\n"
    )
    executable.chmod(0o700)
    return executable


def test_native_window_checks_real_transport_and_loss(tmp_path):
    binary = fake_native(tmp_path, "ok")
    result = quality.native_window(
        binary, tmp_path, [0, 1], [1, 0], np.zeros((2, 2)), tmp_path / "stderr", 2
    )
    assert result["tokens"] == result["greedy_matches"] == 2
    assert result["nll"] == result["bf16_nll"] == pytest.approx(math.log(2))
    assert result["kl"] == 0


def test_subset_scores_only_requested_positions_but_reads_the_entire_window(tmp_path):
    binary = fake_native(tmp_path, "ok")
    teacher = np.array([[math.log(3), 0], [0, 0]])
    result = quality.native_window(
        binary,
        tmp_path,
        [0, 1],
        [1, 0],
        teacher,
        tmp_path / "stderr",
        2,
        score_first=1,
        score_last=1,
    )
    assert result["tokens"] == result["greedy_matches"] == 1
    assert result["bf16_nll"] == result["nll"] == pytest.approx(math.log(2))
    assert result["kl"] == 0


@pytest.mark.parametrize("mode", ["first-nonfinite", "extra", "short"])
def test_unscored_native_rows_still_require_valid_complete_output(tmp_path, mode):
    binary = fake_native(tmp_path, mode)
    with pytest.raises((ValueError, RuntimeError)):
        quality.native_window(
            binary,
            tmp_path,
            [0, 1],
            [1, 0],
            np.zeros((2, 2)),
            tmp_path / "stderr",
            2,
            score_first=1,
            score_last=1,
        )


@pytest.mark.parametrize("first,last", [(0, -1), (2, 2), (-1, 1), (1, 0), (True, 1)])
def test_invalid_score_ranges_fail_before_starting_a_process(tmp_path, first, last):
    with pytest.raises(ValueError):
        quality.native_window(
            tmp_path / "absent",
            tmp_path,
            [0, 1],
            [1, 0],
            np.zeros((2, 2)),
            tmp_path / "stderr",
            2,
            score_first=first,
            score_last=last,
        )


def test_unscored_teacher_or_label_corruption_is_rejected(tmp_path):
    teacher = np.array([[float("nan"), 0], [0, 0]])
    with pytest.raises(ValueError):
        quality.native_window(
            tmp_path / "absent",
            tmp_path,
            [0, 1],
            [1, 0],
            teacher,
            tmp_path / "stderr",
            2,
            score_first=1,
            score_last=1,
        )
    with pytest.raises(ValueError):
        quality.native_window(
            tmp_path / "absent",
            tmp_path,
            [0, 1],
            [True, 0],
            np.zeros((2, 2)),
            tmp_path / "stderr",
            2,
            score_first=1,
            score_last=1,
        )


def test_score_range_against_independent_finite_index_enumeration():
    for length in range(1, 17):
        for first in range(-1, length + 1):
            for last in range(-1, length + 1):
                selected = [i for i in range(length) if first <= i <= last]
                if first < 0 or last >= length or not selected:
                    with pytest.raises(ValueError):
                        quality.scored_count(length, first, last)
                else:
                    assert quality.scored_count(length, first, last) == len(selected)


@pytest.mark.parametrize(
    "mode",
    ["short", "extra", "nonfinite", "boolean", "wrong-token", "malformed", "silent", "exit-error"],
)
def test_native_failures_rejected_and_child_joined(tmp_path, mode):
    binary = fake_native(tmp_path, mode)
    stderr = tmp_path / "stderr"
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        quality.native_window(binary, tmp_path, [0, 1], [1, 0], np.zeros((2, 2)), stderr, 0.5)
    child = int(stderr.read_text().splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_main_records_failure_and_preserves_previous_attempt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"vocab_size": 2}))
    data = tmp_path / "data"
    data.write_text("corpus")
    output = tmp_path / "quality.json"
    module = types.ModuleType("transformers")
    module.PreTrainedTokenizerFast = types.SimpleNamespace(
        from_pretrained=lambda _: types.SimpleNamespace(encode=lambda _: [0, 1, 0])
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

    def interrupted(*args):
        raise ValueError("teacher interrupted")

    monkeypatch.setattr(quality, "teacher_windows", interrupted)
    argv = [
        str(source),
        str(tmp_path),
        "--native-binary",
        str(fake_native(tmp_path, "ok")),
        "--data",
        str(data),
        "--work-dir",
        str(tmp_path / "work"),
        "--output",
        str(output),
        "--tokens",
        "2",
        "--window",
        "2",
    ]
    with pytest.raises(ValueError, match="teacher interrupted"):
        quality.main(argv)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["windows"] == []
    assert "native" not in report and report["elapsed_seconds"] > 0
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        quality.main(argv)
    assert output.read_bytes() == original


def test_main_completes_only_with_all_native_rows(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"vocab_size": 2}))
    data = tmp_path / "data"
    data.write_text("corpus")
    output = tmp_path / "quality.json"
    tokenizer = types.ModuleType("transformers")
    tokenizer.PreTrainedTokenizerFast = types.SimpleNamespace(
        from_pretrained=lambda _: types.SimpleNamespace(encode=lambda _: [0, 1, 0])
    )
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: None
    core.get_active_memory = lambda: 0
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "transformers", tokenizer)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    def teacher(source, data, tokens, window, destination):
        np.save(destination / "teacher-00.npy", np.zeros((window, 2)))

    monkeypatch.setattr(quality, "teacher_windows", teacher)
    quality.main(
        [
            str(source),
            str(tmp_path),
            "--native-binary",
            str(fake_native(tmp_path, "ok")),
            "--data",
            str(data),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--tokens",
            "2",
            "--window",
            "2",
        ]
    )
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and len(report["windows"]) == 1
    assert report["windows"][0]["tokens"] == 2
    assert report["native"]["perplexity"] == report["bf16"]["perplexity"] == 2
    assert report["mean_kl_bf16_to_native"] == report["ppl_change_percent"] == 0
