from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_plan_is_the_reviewed_disjoint_dataset():
    from lfm25_router_plan import validate_plan

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/measurements/quant-lfm25-router38-20261015/input-plan.json"
    )
    plan = json.loads(path.read_text())
    assert validate_plan(plan) is plan
    assert plan["fit_allowed"] is True and plan["separation"]["all_passed"] is True
    assert plan["train"]["shape"] == [16, 2048]
    assert plan["validation"]["shape"] == [12, 1023]
    plan["train"]["input_ids"][0][0] += 1
    with pytest.raises(ValueError, match="frozen"):
        validate_plan(plan)


@pytest.mark.parametrize("remaining,limit,expected", [(31, 100, 1), (100, 20, 20), (100, 200, 70)])
def test_bounded_child_reserves_cleanup(remaining, limit, expected):
    from lfm25_router_plan import child_timeout

    assert child_timeout(remaining, limit) == expected


@pytest.mark.parametrize("remaining,limit", [(30, 10), (-1, 10), (100, 0), (100, True), (True, 2)])
def test_invalid_or_exhausted_budget_cannot_start_a_child(remaining, limit):
    from lfm25_router_plan import child_timeout

    with pytest.raises(ValueError):
        child_timeout(remaining, limit)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import recalibrate_lfm25_router_modal as target

    monkeypatch.setattr(target, "DATA", tmp_path / "router38")
    monkeypatch.setattr(target, "FIT", tmp_path / "fit")
    monkeypatch.setattr(target, "CACHE", tmp_path / "cache.bin")
    monkeypatch.setattr(target, "CACHE_SOURCE", tmp_path / "prior-cache.bin")
    target.CACHE_SOURCE.write_bytes(bytes(176))
    monkeypatch.setattr(target, "CACHE_SHA256", hashlib.sha256(bytes(176)).hexdigest())
    monkeypatch.setattr(target, "validate_plan", lambda p: p)
    monkeypatch.setattr(target, "V2_SHARDS", {"model.safetensors": "a" * 64})
    commits = []
    monkeypatch.setattr(target, "volume", SimpleNamespace(commit=lambda: commits.append(True)))
    return target, commits


def fake_children(target, monkeypatch, failure=None):
    commands = []

    def child(command, log, timeout):
        assert command[:2] == [target.sys.executable, "-u"]
        assert 0 < timeout <= 2300
        name = Path(log).stem
        commands.append(name)
        if name == "quant-train" and failure == "timeout":
            raise TimeoutError("silent child")
        if name == "fit":
            target.FIT.mkdir()
            candidates = {}
            for label in target.ALPHAS:
                folder = target.FIT / label
                folder.mkdir()
                candidates[label] = {
                    "checkpointdir": str(folder),
                    "shards": {"model.safetensors": {"sha256": "a" * 64}},
                }
                (target.FIT / (label + ".gates.safetensors")).write_bytes(b"gate artifact")
            result = {"status": "complete", "candidates": candidates}
            folder = target.FIT
        else:
            folder = target.DATA / name
            folder.mkdir()
            expected = (
                {f"layer-{i}.safetensors" for i in range(2, 40)}
                | {"routing-parameters.safetensors"}
                if name in ("teacher", "quant-train")
                else set()
            )
            if name == "teacher" or name.startswith("validation-"):
                expected |= {f"validation-{i:02d}.npy" for i in range(12)}
            if name == "teacher" or name.startswith("final-"):
                expected |= {"chat-holdout.npy", "cjk-holdout.npy"}
            result = {
                "status": "running" if failure == "partial" and name == "teacher" else "complete",
                "files": {n: {"bytes": 123, "sha256": "a" * 64} for n in expected},
                "passive_hook_control": {"bitexact": True},
                "model_files": {"model.safetensors": {"sha256": "a" * 64}},
            }
            if failure == "missing" and name == "teacher":
                result["files"].pop("layer-2.safetensors")
            if failure == "reload" and name == "validation-alpha-025":
                result["model_files"]["model.safetensors"]["sha256"] = "b" * 64
            if failure == "passive" and name == "quant-train":
                result["passive_hook_control"]["bitexact"] = False
        if failure == "cache" and name == "teacher":
            target.CACHE.write_bytes(b"X" * 176)
        if failure == "cache_append" and name == "teacher":
            target.CACHE.write_bytes(target.CACHE.read_bytes() + b"duplicate-key-record")
        (folder / "report.json").write_text(json.dumps(result))

    monkeypatch.setattr(target.base, "run_logged", child)
    return commands


def final_scores(kl=0.1):
    return {
        "status": "complete",
        "cases": {
            name: {
                "kl": kl,
                "ppl": 10.0,
                "teacher_ppl": 9.0,
                "tokens": 1024,
                "labels_u32le_sha256": "a" * 64,
                "teacher": {"sha256": "b" * 64},
            }
            for name in ("chat-holdout", "cjk-holdout")
        },
    }


def install_scores(monkeypatch, selected):
    import evaluate_lfm25_router as scorer

    def score(plan, teacher, candidate, phase):
        if phase == "validation":
            return {"status": "complete", "fixture": str(candidate)}
        return final_scores(0.1 if candidate.name == "final-baseline" else 0.08)

    monkeypatch.setattr(scorer, "run_score", score)
    monkeypatch.setattr(scorer, "select_candidate", lambda *args: {"selected": selected})


def test_pipeline_orders_real_reload_stages_and_persists_small_artifacts(runner, monkeypatch):
    target, commits = runner
    commands = fake_children(target, monkeypatch)
    install_scores(monkeypatch, "0.5")
    result = target.execute({"validation": {"domains": ["fixture"]}})
    assert commands == [
        "teacher",
        "quant-train",
        "fit",
        "validation-baseline",
        "validation-alpha-025",
        "validation-alpha-050",
        "validation-alpha-100",
        "final-baseline",
        "final-alpha-050",
    ]
    assert result["status"] == "complete" and result["selected"] == "alpha-050"
    assert result["final_quality_passed"] is True and result["publication"] is False
    assert result["cache_prefix_preserved"] is True and commits == [True]
    assert len(list((target.DATA / "fit-artifacts").glob("*.safetensors"))) == 3
    assert json.loads((target.DATA / "report.json").read_text()) == result


def test_validation_rejection_never_selects_using_final_data(runner, monkeypatch):
    target, commits = runner
    commands = fake_children(target, monkeypatch)
    install_scores(monkeypatch, None)
    result = target.execute({"validation": {"domains": ["fixture"]}})
    assert result["stage"] == "rejected_validation" and result["selected"] is None
    assert not any(name.startswith("final-") for name in commands)
    assert result["publication"] is False and commits == [True]


@pytest.mark.parametrize(
    "failure,exception",
    [
        ("timeout", TimeoutError),
        ("partial", ValueError),
        ("cache", ValueError),
        ("cache_append", ValueError),
        ("missing", ValueError),
        ("passive", ValueError),
        ("reload", ValueError),
    ],
)
def test_incomplete_interrupted_or_changed_cache_is_failed_and_committed(
    runner, monkeypatch, failure, exception
):
    target, commits = runner
    fake_children(target, monkeypatch, failure)
    install_scores(monkeypatch, None)
    with pytest.raises(exception):
        target.execute({"validation": {"domains": ["fixture"]}})
    report = json.loads((target.DATA / "report.json").read_text())
    assert report["status"] == "failed" and report["publication"] is False
    assert report["final_quality_passed"] is False and commits == [True]


def test_previous_artifacts_are_not_overwritten(runner):
    target, commits = runner
    target.DATA.mkdir()
    (target.DATA / "evidence").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        target.execute({})
    assert (target.DATA / "evidence").read_bytes() == b"preserve" and commits == []


def test_fixed_cache_is_checked_before_any_child_or_directory(runner):
    target, commits = runner
    target.CACHE_SOURCE.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="cache"):
        target.execute({})
    assert not target.DATA.exists() and commits == []


def test_final_guards_keep_ppl_and_reference_identity_separate():
    from recalibrate_lfm25_router_modal import final_guard

    baseline, candidate = final_scores(), final_scores(0.08)
    assert final_guard(baseline, candidate)
    candidate["cases"]["chat-holdout"]["ppl"] = 10.21
    assert not final_guard(baseline, candidate)
    candidate = copy.deepcopy(baseline)
    candidate["cases"]["chat-holdout"]["teacher"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        final_guard(baseline, candidate)
    with pytest.raises(ValueError):
        final_guard({"status": "running"}, candidate)


def test_artifact_persistence_failure_cannot_leave_a_complete_report(runner, monkeypatch):
    target, commits = runner
    fake_children(target, monkeypatch)
    install_scores(monkeypatch, "0.5")

    def fail_copy(*args):
        raise OSError("volume write failed")

    monkeypatch.setattr(target.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="volume write"):
        target.execute({"validation": {"domains": ["fixture"]}})
    report = json.loads((target.DATA / "report.json").read_text())
    assert report["status"] == "failed" and report["selected"] is None
    assert report["final_quality_passed"] is False and commits == [True]
