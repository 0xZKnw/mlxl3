import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from eval_lfm25_candidate import file_sha, ppl_gate, run
from test_lfm25_quality import fake_native


def frozen_fixture(tmp_path):
    binary = fake_native(tmp_path, "ok")
    directory = tmp_path / "ids"
    directory.mkdir()
    teacher = tmp_path / "teacher-00.npy"
    np.save(teacher, np.zeros((2, 2)))
    import hashlib

    ids = [0, 1, 0]
    case = {
        "input_tokens": 2,
        "window": 2,
        "scored_tokens": 2,
        "score_first": 0,
        "score_last": 1,
        "token_sha256": hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest(),
        "teacher_dir": str(tmp_path),
        "teacher_files": {teacher.name: file_sha(teacher)},
        "windows": [{"tokens": 2, "bf16_nll": math.log(2), "kl": 0.1}],
        "native": {"perplexity": 1.95},
        "bf16": {"perplexity": 2},
        "mean_kl_bf16_to_native": 0.1,
    }
    baseline = {
        "status": "complete",
        "native_sha256": file_sha(binary),
        "cases": {"main-8192": case},
    }
    bos = copy.deepcopy(case)
    bos.update(scored_tokens=1, score_first=1)
    bos["windows"][0]["tokens"] = 1
    baseline["cases"]["bos-paired"] = bos
    for name in baseline["cases"]:
        (directory / f"{name}.json").write_text(json.dumps(ids))
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    return binary, path, directory, baseline


def test_real_candidate_flow_accepts_a_perfect_teacher_even_above_v2_ppl(tmp_path):
    binary, baseline, ids, _ = frozen_fixture(tmp_path)
    result = run(baseline, ids, tmp_path, binary, tmp_path / "result.json", 2)
    assert result["status"] == "complete" and result["quality_passed"] is True
    assert result["publication"] is False
    assert sum(c["scored_tokens"] for c in result["cases"].values()) == 3
    assert all(c["native"]["perplexity"] == pytest.approx(2) for c in result["cases"].values())
    assert all(c["mean_kl_bf16_to_native"] == 0 for c in result["cases"].values())
    with pytest.raises(FileExistsError):
        run(baseline, ids, tmp_path, binary, tmp_path / "result.json", 2)


@pytest.mark.parametrize(
    "corruption", ["ids", "teacher", "label_protocol", "binary", "incomplete", "empty", "native"]
)
def test_invalid_candidate_evidence_cannot_report_passed(tmp_path, corruption):
    binary, baseline_path, ids, baseline = frozen_fixture(tmp_path)
    if corruption == "ids":
        (ids / "bos-paired.json").write_text("[0,0,0]")
    elif corruption == "teacher":
        np.save(tmp_path / "teacher-00.npy", np.ones((2, 2)))
    elif corruption == "label_protocol":
        baseline["cases"]["bos-paired"]["windows"][0]["bf16_nll"] = 5
    elif corruption == "binary":
        baseline["native_sha256"] = "wrong"
    elif corruption == "incomplete":
        baseline["status"] = "failed"
    elif corruption == "empty":
        baseline["cases"].clear()
    else:
        binary = fake_native(tmp_path, "first-nonfinite")
        baseline["native_sha256"] = file_sha(binary)
    baseline_path.write_text(json.dumps(baseline))
    output = tmp_path / "failed.json"
    with pytest.raises((ValueError, RuntimeError)):
        run(baseline_path, ids, tmp_path, binary, output, 2)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["quality_passed"] is False
    assert report["publication"] is False


def test_perplexity_guard_has_a_teacher_fixed_point_and_a_real_regression_boundary():
    assert ppl_gate(28.1536261114, 26.8616281999, 28.1536261114)
    assert ppl_gate(2, 1.95, 2)
    assert not ppl_gate(2.1, 1.95, 2)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ppl_gate(invalid, 1.95, 2)
