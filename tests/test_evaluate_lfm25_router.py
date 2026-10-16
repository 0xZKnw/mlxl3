from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_lfm25_router as score


def independent_metrics(actual, teacher, labels):
    totals = [[], [], []]
    matches = 0
    for row, reference, label in zip(actual, teacher, labels, strict=True):

        def distribution(values):
            peak = max(map(float, values))
            unnormalized = [math.exp(float(v) - peak) for v in values]
            denominator = math.fsum(unnormalized)
            return [v / denominator for v in unnormalized]

        p, q = distribution(row), distribution(reference)
        totals[0].append(-math.log(p[label]))
        totals[1].append(-math.log(q[label]))
        totals[2].append(math.fsum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True)))
        matches += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
    nll, teacher_nll, kl = [math.fsum(values) / len(labels) for values in totals]
    return {
        "nll": nll,
        "teacher_nll": teacher_nll,
        "kl": kl,
        "ppl": math.exp(nll),
        "teacher_ppl": math.exp(teacher_nll),
        "greedy_matches": matches,
    }


def vectors(tmp_path, rows=71):
    rng = np.random.default_rng(20261015)
    actual = rng.normal(size=(rows, 5)).astype(np.float16)
    teacher = rng.normal(size=(rows, 5)).astype(np.float32)
    labels = [i % 5 for i in range(rows)]
    a, t = tmp_path / "actual.npy", tmp_path / "teacher.npy"
    np.save(a, actual)
    np.save(t, teacher)
    return a, t, actual, teacher, labels


def test_score_matches_independent_full_softmax_and_file_provenance(tmp_path, monkeypatch):
    a, t, x, y, labels = vectors(tmp_path)
    expected = independent_metrics(x, y, labels)
    old_load, old_finite = np.load, np.isfinite
    chunks, loads = [], []

    def finite(values, *args, **kwargs):
        if values.ndim == 2:
            chunks.append(len(values))
            assert len(values) <= 32
        return old_finite(values, *args, **kwargs)

    def load(*args, **kwargs):
        loads.append(kwargs)
        return old_load(*args, **kwargs)

    monkeypatch.setattr(np, "isfinite", finite)
    monkeypatch.setattr(np, "load", load)
    result = score.score_case(a, t, labels, vocabulary=5)
    assert result["status"] == "complete" and result["tokens"] == 71
    assert chunks == [32, 32, 32, 32, 7, 7]
    assert all(k == {"mmap_mode": "r", "allow_pickle": False} for k in loads)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, rel=2e-13, abs=2e-13)
    for key, path in (("actual", a), ("teacher", t)):
        assert result[key]["bytes"] == path.stat().st_size
        assert result[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        result["labels_u32le_sha256"]
        == hashlib.sha256(np.asarray(labels, dtype="<u4").tobytes()).hexdigest()
    )
    assert score.score_case(t, t, labels, vocabulary=5)["kl"] == 0


@pytest.mark.parametrize(
    "change",
    ["short", "extra", "vocab", "rank", "f64", "integer", "nan", "inf", "truncated", "emptyfile"],
)
@pytest.mark.parametrize("side", ["actual", "teacher"])
def test_bad_or_partial_logits_cannot_be_scored(tmp_path, change, side):
    a, t, x, y, labels = vectors(tmp_path, 3)
    path, values = (a, x) if side == "actual" else (t, y)
    if change == "short":
        values = values[:-1]
    elif change == "extra":
        values = np.concatenate([values, values[:1]])
    elif change == "vocab":
        values = values[:, :4]
    elif change == "rank":
        values = values.reshape(-1)
    elif change == "f64":
        values = values.astype(np.float64)
    elif change == "integer":
        values = values.astype(np.int32)
    elif change in ("nan", "inf"):
        values[-1, -1] = float(change)
    if change == "emptyfile":
        path.write_bytes(b"")
    else:
        np.save(path, values)
        if change == "truncated":
            path.write_bytes(path.read_bytes()[:130])
    with pytest.raises((ValueError, OSError, EOFError)):
        score.score_case(a, t, labels, vocabulary=5)


@pytest.mark.parametrize("labels", [[], [True], [-1], [5], [1.0], (1,)])
def test_invalid_labels_are_rejected_before_loading(labels):
    with pytest.raises(ValueError):
        score.score_case("absent", "absent", labels, vocabulary=5)


@pytest.mark.parametrize("chunk", [0, 33, True, 1.5])
def test_chunk_bound_cannot_be_bypassed(chunk):
    with pytest.raises(ValueError):
        score.score_case("absent", "absent", [0], vocabulary=5, chunk_size=chunk)


def metric(tokens=2, kl=0.2, nll=1.0, teacher_nll=1.0):
    return {
        "status": "complete",
        "tokens": tokens,
        "kl": kl,
        "nll": nll,
        "ppl": math.exp(nll),
        "teacher_nll": teacher_nll,
        "teacher_ppl": math.exp(teacher_nll),
        "greedy_matches": tokens,
        "labels_u32le_sha256": "a" * 64,
        "teacher": {"sha256": "b" * 64, "bytes": 100},
        "actual": {"sha256": "c" * 64, "bytes": 100},
    }


def panel():
    baseline = {name: metric() for name in score.VALIDATION_NAMES}
    candidates = {alpha: copy.deepcopy(baseline) for alpha in score.ALPHAS}
    return baseline, candidates, [f"domain{i // 2}" for i in range(12)]


def set_metrics(row, **changes):
    row.update(changes)
    row["ppl"], row["teacher_ppl"] = math.exp(row["nll"]), math.exp(row["teacher_nll"])


def test_selection_uses_registered_alpha_order_for_ties_and_teacher_ppl_fixed_point():
    base, candidates, domains = panel()
    for row in base.values():
        set_metrics(row, nll=math.log(2), teacher_nll=math.log(4))
    candidates = {alpha: copy.deepcopy(base) for alpha in reversed(score.ALPHAS)}
    for rows in candidates.values():
        for row in rows.values():
            set_metrics(row, kl=0.198, nll=math.log(4))
    result = score.select_candidate({"status": "complete", "cases": base}, candidates, domains)
    assert result["selected"] == "0.25"
    assert list(result["candidates"]) == list(score.ALPHAS)
    assert all(row["eligible"] for row in result["candidates"].values())
    candidates["1"]["validation-00"]["kl"] = 0.19
    assert score.select_candidate(base, candidates, domains)["selected"] == "1"


def test_domain_kl_uses_token_weighting_and_rejects_a_hidden_domain_regression():
    base, candidates, domains = panel()
    for rows in (base, *candidates.values()):
        rows["validation-00"]["tokens"] = rows["validation-00"]["greedy_matches"] = 1
        rows["validation-01"]["tokens"] = rows["validation-01"]["greedy_matches"] = 9
    for row in candidates["0.25"].values():
        row["kl"] = 0.1
    candidates["0.25"]["validation-00"]["kl"] = 0
    candidates["0.25"]["validation-01"]["kl"] = 0.3
    result = score.select_candidate(base, candidates, domains)
    chosen = result["candidates"]["0.25"]
    assert chosen["overall"]["kl"] < 0.198
    assert chosen["domains"]["domain0"]["kl"] == pytest.approx(0.27)
    assert chosen["failed_domains"] == ["domain0"] and result["selected"] is None


def test_domain_ppl_is_exp_of_weighted_nll_not_average_perplexities():
    base, candidates, domains = panel()
    for rows in (base, *candidates.values()):
        set_metrics(rows["validation-00"], nll=0, teacher_nll=0)
        set_metrics(rows["validation-01"], nll=math.log(9), teacher_nll=0)
    for row in candidates["0.25"].values():
        row["kl"] = 0.1
    for name in ("validation-00", "validation-01"):
        set_metrics(candidates["0.25"][name], nll=math.log(4))
    result = score.select_candidate(base, candidates, domains)
    assert result["baseline_domains"]["domain0"]["ppl"] == pytest.approx(3)
    assert result["candidates"]["0.25"]["failed_domains"] == ["domain0"]
    assert result["selected"] is None


@pytest.mark.parametrize(
    "bad",
    [
        "report",
        "case",
        "missing",
        "extra",
        "duplicate",
        "missingalpha",
        "unknownalpha",
        "zero",
        "matches",
        "nan",
        "inf",
        "negative",
        "ppl",
        "labels",
        "teacher",
        "teacher_nll",
        "tokens",
        "artifact",
    ],
)
def test_incomplete_corrupt_or_misaligned_evidence_never_selects(bad):
    base, candidates, domains = panel()
    rows = candidates["0.25"]
    row = rows["validation-00"]
    if bad == "report":
        candidates["0.25"] = {"status": "running", "cases": rows}
    elif bad == "case":
        row["status"] = "failed"
    elif bad == "missing":
        rows.pop("validation-00")
    elif bad == "extra":
        rows["extra"] = metric()
    elif bad == "duplicate":
        candidates["0.25"] = [dict(value, name="validation-00") for value in rows.values()]
    elif bad == "missingalpha":
        candidates.pop("1")
    elif bad == "unknownalpha":
        candidates["0.75"] = candidates.pop("1")
    elif bad == "zero":
        row["tokens"] = 0
    elif bad == "matches":
        row["greedy_matches"] += 1
    elif bad in ("nan", "inf", "negative"):
        row["kl"] = {"nan": float("nan"), "inf": float("inf"), "negative": -0.1}[bad]
    elif bad == "ppl":
        row["ppl"] = 100
    elif bad == "labels":
        row["labels_u32le_sha256"] = "d" * 64
    elif bad == "teacher":
        row["teacher"]["sha256"] = "d" * 64
    elif bad == "teacher_nll":
        set_metrics(row, teacher_nll=2)
    elif bad == "tokens":
        row["tokens"] += 1
    else:
        row["teacher"] = []
    with pytest.raises(ValueError):
        score.select_candidate(base, candidates, domains)


def test_selection_list_api_zero_baseline_and_domain_boundaries():
    base, candidates, domains = panel()
    result = score.select_candidate(
        list(base.values()), {a: list(rows.values()) for a, rows in candidates.items()}, domains
    )
    assert result["selected"] is None
    for row in base.values():
        row["kl"] = 0
    for rows in candidates.values():
        for row in rows.values():
            row["kl"] = 0
    assert score.select_candidate(base, candidates, domains)["selected"] is None
    assert score.improves_kl(0.99, 1)
    assert not score.improves_kl(math.nextafter(0.99, math.inf), 1)
    assert score.domain_gate(1.02, 1, 4.08, 2, 4)
    assert not score.domain_gate(math.nextafter(1.02, math.inf), 1, 4, 2, 4)
    assert not score.domain_gate(1, 1, math.nextafter(4.08, math.inf), 2, 4)
    with pytest.raises(ValueError):
        score.select_candidate(base, candidates, ["one"] * 12)


@pytest.fixture
def synthetic_plan(monkeypatch):
    monkeypatch.setattr(score, "validate_plan", lambda plan: plan)


def capture_manifest(folder, plan):
    records = {}
    for path in folder.glob("*.npy"):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        records[path.name] = {
            **score.artifact(path),
            "shape": list(values.shape),
            "storage_dtype": str(values.dtype),
        }
    manifest = {
        "status": "complete",
        "kind": "teacher" if folder.name == "teacher" else "quantized",
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "files": records,
    }
    (folder / "report.json").write_text(json.dumps(manifest))
    return manifest


def final_plan():
    return {
        "status": "frozen",
        "final": {
            name: {"input_ids": [0, 1] * 512, "labels": [1, 0] * 512}
            for name in ("chat-holdout", "cjk-holdout")
        },
    }


def test_run_score_and_cli_write_atomic_complete_or_failed_reports(
    tmp_path, monkeypatch, capsys, synthetic_plan
):
    teacher, actual = tmp_path / "teacher", tmp_path / "actual"
    teacher.mkdir()
    actual.mkdir()
    plan = final_plan()
    for name in plan["final"]:
        np.save(
            teacher / f"{name}.npy", np.tile(np.array([[1, 2], [3, 4]], dtype=np.float32), (512, 1))
        )
        np.save(
            actual / f"{name}.npy", np.tile(np.array([[0, 1], [2, 3]], dtype=np.float16), (512, 1))
        )
    capture_manifest(teacher, plan)
    capture_manifest(actual, plan)
    result = score.run_score(plan, teacher, actual, "final", vocabulary=2)
    assert result["status"] == "complete" and result["overall"]["tokens"] == 2048
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "score.json"
    original = score.run_score
    monkeypatch.setattr(
        score, "run_score", lambda p, t, c, phase: original(p, t, c, phase, vocabulary=2)
    )
    args = [
        "score",
        "--plan",
        str(plan_path),
        "--teacher-dir",
        str(teacher),
        "--candidate-dir",
        str(actual),
        "--output",
        str(output),
        "--phase",
        "final",
    ]
    replace = Path.replace
    transitions = []

    def checked_replace(path, destination):
        transitions.append(json.loads(path.read_text())["status"])
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", checked_replace)
    score.main(args)
    assert transitions == ["running", "complete"]
    assert json.loads(output.read_text())["status"] == "complete"
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        score.main(args)
    assert output.read_bytes() == saved and not list(tmp_path.glob("*.tmp"))
    output.unlink()
    (actual / "cjk-holdout.npy").write_bytes(b"")
    with pytest.raises((ValueError, EOFError)):
        score.main(args)
    assert json.loads(output.read_text())["status"] == "failed"
    assert not capsys.readouterr().out


def test_cli_subprocess_missing_output_and_interrupt_never_claim_completion(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(final_plan()))
    args = [
        "score",
        "--plan",
        str(plan_path),
        "--teacher-dir",
        str(tmp_path),
        "--candidate-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "failed.json"),
        "--phase",
        "final",
    ]
    result = subprocess.run(
        [sys.executable, str(Path(score.__file__)), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0 and not result.stdout
    assert json.loads((tmp_path / "failed.json").read_text())["status"] == "failed"
    args[args.index("--output") + 1] = str(tmp_path / "interrupt.json")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("fixture")

    monkeypatch.setattr(score, "run_score", interrupted)
    with pytest.raises(KeyboardInterrupt):
        score.main(args)
    assert json.loads((tmp_path / "interrupt.json").read_text())["status"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))


def test_duplicate_plan_keys_and_unfrozen_plans_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"frozen","status":"running"}')
    with pytest.raises(ValueError):
        score._read_json(path)
    for plan in ({}, {"status": "running"}, []):
        with pytest.raises(ValueError):
            score.run_score(plan, tmp_path, tmp_path)


def test_final_holdouts_cannot_silently_drop_scored_positions(tmp_path, synthetic_plan):
    plan = final_plan()
    plan["final"]["chat-holdout"]["labels"].pop()
    with pytest.raises(ValueError, match="1024"):
        score.run_score(plan, tmp_path, tmp_path, "final", vocabulary=2)


@pytest.mark.parametrize(
    "change",
    ["running", "failed", "missing", "wrongplan", "shape", "sha", "bytes", "dtype", "kind"],
)
def test_preallocated_silent_or_changed_capture_never_becomes_complete(
    tmp_path, synthetic_plan, change
):
    teacher, actual = tmp_path / "teacher", tmp_path / "actual"
    teacher.mkdir()
    actual.mkdir()
    plan = final_plan()
    for folder in (teacher, actual):
        for name in plan["final"]:
            np.save(folder / f"{name}.npy", np.zeros((1024, 2), dtype=np.float32))
        capture_manifest(folder, plan)
    manifest = json.loads((actual / "report.json").read_text())
    row = manifest["files"]["chat-holdout.npy"]
    if change in ("running", "failed"):
        manifest["status"] = change
    elif change == "missing":
        del manifest["files"]["chat-holdout.npy"]
    elif change == "wrongplan":
        manifest["plan_sha256"] = "0" * 64
    elif change == "shape":
        row["shape"][0] -= 1
    elif change == "sha":
        row["sha256"] = "0" * 64
    elif change == "dtype":
        row["storage_dtype"] = "float16"
    elif change == "kind":
        manifest["kind"] = "teacher"
    else:
        row["bytes"] -= 1
    (actual / "report.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        score.run_score(plan, teacher, actual, "final", vocabulary=2)


def test_validation_scoring_runs_real_losses_for_all_twelve_cases(tmp_path, synthetic_plan):
    teacher, actual = tmp_path / "teacher", tmp_path / "actual"
    teacher.mkdir()
    actual.mkdir()
    ids, labels = ([0, 1] * 512)[:1023], ([1, 0] * 512)[:1023]
    domains = [f"domain{i // 2}" for i in range(12)]
    plan = {
        "status": "frozen",
        "validation": {
            "input_ids": [ids] * 12,
            "labels": [labels] * 12,
            "domains": domains,
            "shape": [12, 1023],
        },
    }
    for folder in (teacher, actual):
        for name in score.VALIDATION_NAMES:
            np.save(folder / f"{name}.npy", np.zeros((1023, 2), dtype=np.float32))
        capture_manifest(folder, plan)
    result = score.run_score(plan, teacher, actual, vocabulary=2)
    assert result["status"] == "complete" and result["overall"]["tokens"] == 12276
    assert set(result["domains"]) == set(domains)
    assert all(
        v["tokens"] == 2046 and v["ppl"] == pytest.approx(2) for v in result["domains"].values()
    )
    assert (
        score.select_candidate(result, {a: result for a in score.ALPHAS}, domains)["selected"]
        is None
    )
