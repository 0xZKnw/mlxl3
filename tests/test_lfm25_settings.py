from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from pilot_lfm25_settings import (
    PROFILES,
    SEEDS,
    applied_weights,
    choose_settings,
    kl_rows,
    profile_options,
    settings_args,
)


@pytest.mark.parametrize("profile", PROFILES)
def test_real_converter_command_keeps_hq_calibration_and_changes_only_settings(profile):
    import quantize_lfm25_settings_modal as module

    cb, scaled = profile_options(profile)
    command = module.settings_command(
        Path("source"), Path("work"), Path("out"), False, profile, module.base.conversion_command
    )
    assert command[command.index("-cb") + 1] == cb
    assert ("--scaled_hessian" in command) == scaled
    assert "--hq" in command and "--recipe" not in command
    for flag, expected in (
        ("-b", "4"),
        ("-hb", "16"),
        ("-cr", "250"),
        ("-cc", "2048"),
        ("--out_scales", "auto"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert command.count("-cb") == 1


@pytest.mark.parametrize("cb,scaled", [("other", False), ("mcg", 1), ("mul1", None), ("", True)])
def test_unknown_quantizer_settings_fail(cb, scaled):
    with pytest.raises(ValueError):
        settings_args(cb, scaled)
    with pytest.raises(ValueError):
        profile_options("not-a-profile")


def matrix():
    return {
        "status": "complete",
        "restoration_verified": True,
        "validation": [
            {
                "profile": p,
                "seed": s,
                "window": w,
                "rows": 128,
                "vocabulary": 128000,
                "kl": (1.0, 0.8, 0.9, 0.7)[i],
            }
            for i, p in enumerate(PROFILES)
            for s in SEEDS
            for w in range(12)
        ],
        "projections": [
            {
                "profile": p,
                "seed": s,
                "key": f"projection-{k}",
                "rms": 0.1 if p == PROFILES[0] else 0.09,
            }
            for p in PROFILES
            for s in SEEDS
            for k in range(29)
        ],
    }


def test_selection_uses_both_seeds_and_rejects_domain_or_single_projection_regression():
    data = matrix()
    assert choose_settings(data)["selected"]["profile"] == "mul1-scaled"
    for r in data["validation"]:
        if r["profile"] == "mul1-scaled" and r["window"] < 2:
            r["kl"] = 1.2
    assert choose_settings(data)["selected"]["profile"] == "mcg-scaled"
    next(r for r in data["projections"] if r["profile"] == "mcg-scaled")["rms"] = 0.2
    assert choose_settings(data)["selected"]["profile"] == "mul1-standard"
    for r in data["projections"]:
        if r["profile"] != PROFILES[0]:
            r["rms"] = 0.2
    assert choose_settings(data) is None


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "unrestored",
        "missing_seed",
        "duplicate",
        "wrong_count",
        "vocab",
        "nan",
        "zero",
        "negative",
        "missing_projection",
        "duplicate_projection",
    ],
)
def test_incomplete_or_nonfinite_reports_never_select(failure):
    data = matrix()
    if failure == "partial":
        data["status"] = "running"
    elif failure == "unrestored":
        data["restoration_verified"] = False
    elif failure == "missing_seed":
        data["validation"] = [r for r in data["validation"] if r["seed"] != 26]
    elif failure == "duplicate":
        data["validation"].append(copy.deepcopy(data["validation"][0]))
    elif failure == "wrong_count":
        data["validation"][0]["rows"] = 0
    elif failure == "vocab":
        data["validation"][0]["vocabulary"] = 10
    elif failure == "nan":
        data["validation"][0]["kl"] = float("nan")
    elif failure == "zero":
        data["validation"][0]["kl"] = 0.0
    elif failure == "negative":
        data["validation"][0]["kl"] = -0.1
    elif failure == "missing_projection":
        data["projections"].pop()
    else:
        data["projections"].append(copy.deepcopy(data["projections"][0]))
    with pytest.raises(ValueError):
        choose_settings(data)


def test_full_vocab_kl_has_an_independent_closed_form_and_detects_mutation():
    teacher = torch.tensor([[math.log(3), 0.0]], dtype=torch.float64)
    candidate = torch.zeros_like(teacher)
    result = kl_rows(teacher, candidate)
    assert result["kl"] == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5), abs=1e-14)
    assert result["rows"] == 1 and result["vocabulary"] == 2
    assert kl_rows(teacher, teacher)["kl"] == 0
    assert kl_rows(teacher + 10000, candidate + 10000)["kl"] == pytest.approx(
        result["kl"], abs=1e-12
    )


@pytest.mark.parametrize("kind", ["empty", "mismatch", "nan", "inf"])
def test_bad_logits_refused(kind):
    a = torch.ones(2, 3)
    b = a.clone()
    if kind == "empty":
        a = b = torch.empty(0, 3)
    elif kind == "mismatch":
        b = torch.ones(2, 4)
    else:
        b[0, 0] = float(kind)
    with pytest.raises(ValueError):
        kl_rows(a, b)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("raises", [False, True])
def test_real_tensor_allocations_mutate_and_restore_on_success_or_error(seed, raises):
    generator = torch.Generator().manual_seed(seed)
    live = {
        "a": torch.randn(3, 4, generator=generator),
        "b": torch.randn(4, 2, generator=generator),
    }
    original = {k: t.clone() for k, t in live.items()}
    pointers = {k: t.data_ptr() for k, t in live.items()}
    replacements = {k: torch.full_like(t, 2) for k, t in live.items()}
    try:
        with applied_weights(live, replacements):
            assert all(torch.equal(t, torch.full_like(t, 2)) for t in live.values())
            assert pointers == {k: t.data_ptr() for k, t in live.items()}
            if raises:
                raise RuntimeError("interrupted panel")
    except RuntimeError:
        assert raises
    assert all(torch.equal(live[k].view(torch.uint8), original[k].view(torch.uint8)) for k in live)


@pytest.mark.parametrize("kind", ["empty", "missing", "shape", "dtype", "nan"])
def test_invalid_weight_panel_rejected_before_mutation(kind):
    live = {"a": torch.ones(2, 2), "b": torch.ones(2, 2)}
    values = {k: t.clone() * 2 for k, t in live.items()}
    if kind == "empty":
        live = values = {}
    elif kind == "missing":
        values.pop("b")
    elif kind == "shape":
        values["b"] = torch.ones(3, 2)
    elif kind == "dtype":
        values["b"] = values["b"].half()
    else:
        values["b"][0, 0] = float("nan")
    with pytest.raises(ValueError), applied_weights(live, values):
        pass
    assert all(torch.equal(t, torch.ones_like(t)) for t in live.values())


@pytest.mark.parametrize("fails", [False, True])
def test_probe_entry_uses_official_converter_and_restores_globals(tmp_path, monkeypatch, fails):
    import quantize_lfm25_settings_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path / "settings30")
    monkeypatch.setattr(module, "preparation", lambda: {"folder": str(tmp_path / "prepared")})
    before = (
        module.base.DATA,
        module.base.volume,
        module.base.conversion_command,
        module.base.run_logged,
    )

    def convert(files, candidate):
        assert files == {"fixture": b"test"} and candidate == "v3"
        assert module.base.DATA == module.DATA / "mul1-scaled"
        command = module.base.conversion_command(Path("src"), Path("work"), Path("out"), True)
        assert command[command.index("-cb") + 1] == "mul1" and "--scaled_hessian" in command
        assert "--cal_data" not in command
        if fails:
            raise RuntimeError("converter failed")
        output = module.base.DATA / "probe-1"
        output.mkdir()
        return json.dumps({"status": "complete", "folder": str(output)})

    monkeypatch.setattr(module.base, "convert_model", convert)
    monkeypatch.setattr(module.volume, "commit", lambda: None)
    if fails:
        with pytest.raises(RuntimeError):
            module.convert_profile("mul1-scaled", {"fixture": b"test"})
    else:
        assert (
            json.loads(module.convert_profile("mul1-scaled", {"fixture": b"test"}))[
                "settings_profile"
            ]
            == "mul1-scaled"
        )
    assert before == (
        module.base.DATA,
        module.base.volume,
        module.base.conversion_command,
        module.base.run_logged,
    )


@pytest.mark.parametrize(
    "failure",
    [None, "missing_native", "wrong_profile", "partial", "different_selection", "changed_helper"],
)
def test_full_conversion_requires_matching_pilot_and_native_approval(
    tmp_path, monkeypatch, failure
):
    import quantize_lfm25_settings_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "preparation", lambda: {"folder": str(tmp_path / "prepared")})
    root = tmp_path / "mul1-scaled"
    root.mkdir()
    approval = {"profile": "mul1-scaled", "native_sha256": module.NATIVE_SHA256}
    data = matrix()
    data.update(
        selection=choose_settings(data),
        script_sha256=hashlib.sha256(
            Path(module.__file__).with_name("pilot_lfm25_settings.py").read_bytes()
        ).hexdigest(),
    )
    if failure == "missing_native":
        approval.pop("native_sha256")
    elif failure == "wrong_profile":
        approval["profile"] = "mcg-standard"
    elif failure == "partial":
        data["status"] = "running"
    elif failure == "different_selection":
        data["selection"]["selected"]["profile"] = "mcg-scaled"
    elif failure == "changed_helper":
        data["script_sha256"] = "bad"
    (root / "probe-approved.json").write_text(json.dumps(approval))
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot/report.json").write_text(json.dumps(data))

    def convert(files, candidate):
        assert failure is None and files is None and candidate == "v3"
        command = module.base.conversion_command(Path("src"), Path("work"), Path("out"), False)
        assert command[command.index("--cal_data") + 1] == str(
            tmp_path / "prepared/original.safetensors"
        )
        assert command[command.index("-cb") + 1] == "mul1" and "--scaled_hessian" in command
        output = root / "v3"
        output.mkdir()
        return json.dumps({"status": "complete", "folder": str(output)})

    monkeypatch.setattr(module.base, "convert_model", convert)
    monkeypatch.setattr(module.volume, "commit", lambda: None)
    if failure:
        with pytest.raises(ValueError):
            module.convert_profile("mul1-scaled")
    else:
        result = json.loads(module.convert_profile("mul1-scaled"))
        assert result["status"] == "complete" and result["settings_profile"] == "mul1-scaled"


@pytest.mark.parametrize("partial", [False, True])
def test_pilot_launcher_failure_persists_failed_status(tmp_path, monkeypatch, partial):
    import quantize_lfm25_settings_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(
        module, "preparation", lambda: {"source": "source", "folder": str(tmp_path / "prepared")}
    )
    report = tmp_path / "pilot/report.json"

    def interrupted(*args):
        if partial:
            report.parent.mkdir()
            report.write_text(json.dumps({"status": "running", "projections": [{"done": 1}]}))
        raise TimeoutError("bounded child interrupted")

    monkeypatch.setattr(module.base, "run_logged", interrupted)
    monkeypatch.setattr(module.volume, "commit", lambda: None)
    with pytest.raises(TimeoutError):
        module.pilot.local()
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and result["selection"] is None
    assert "TimeoutError" in result["launcher_error"]
    assert bool(result.get("projections")) == partial
    original = report.read_bytes()
    with pytest.raises(FileExistsError):
        module.pilot.local()
    assert report.read_bytes() == original
