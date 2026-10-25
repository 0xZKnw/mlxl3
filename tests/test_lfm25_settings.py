from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from pilot_lfm25_settings import (
    EXPERTS,
    PROFILES,
    SEEDS,
    applied_weights,
    choose_settings,
    kl_rows,
    profile_options,
    projection_scaled,
    settings_args,
)

torch = pytest.importorskip("torch")


def test_modal_cli_keeps_authentication_and_candidate_profiles_distinct(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MODAL_CONFIG_PATH=str(tmp_path / "modal.toml"), NO_COLOR="1")
    env.pop("MODAL_PROFILE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "run",
            "scripts/quantize_lfm25_settings_modal.py",
            "--help",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--candidate-profile TEXT" in result.stdout
    assert result.stdout.count("--profile TEXT") == 1


def test_registered_expert_panel_keeps_the_successful_preselection():
    prior = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/measurements/quant-lfm25-settings30-20261015/pilot.json"
        ).read_text()
    )
    assert prior["status"] == "complete" and prior["restoration_verified"]
    assert EXPERTS == {int(k): v["chosen"] for k, v in prior["expert_selection"].items()}
    assert all(
        len(set(values)) == 3 and all(0 <= v < 64 for v in values) for values in EXPERTS.values()
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
    if profile in ("mcg-no-down", "mcg-trunk-only"):
        assert command[2:4] == ["/opt/mlxl3/convert_lfm25_settings.py", profile]
    else:
        assert command[2:4] == ["-m", "exllamav3.conversion.convert_model"]


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
    assert choose_settings(data)["selected"]["profile"] == "mcg-trunk-only"
    for r in data["validation"]:
        if r["profile"] == "mcg-trunk-only" and r["window"] < 2:
            r["kl"] = 1.2
    assert choose_settings(data)["selected"]["profile"] == "mcg-scaled"
    next(r for r in data["projections"] if r["profile"] == "mcg-scaled")["rms"] = 0.2
    assert choose_settings(data)["selected"]["profile"] == "mcg-no-down"
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
        assert module.base.DATA == module.DATA / "mcg-trunk-only"
        command = module.base.conversion_command(Path("src"), Path("work"), Path("out"), True)
        assert command[command.index("-cb") + 1] == "mcg" and "--scaled_hessian" in command
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
            module.convert_profile("mcg-trunk-only", {"fixture": b"test"})
    else:
        assert (
            json.loads(module.convert_profile("mcg-trunk-only", {"fixture": b"test"}))[
                "settings_profile"
            ]
            == "mcg-trunk-only"
        )
    assert before == (
        module.base.DATA,
        module.base.volume,
        module.base.conversion_command,
        module.base.run_logged,
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_native",
        "wrong_profile",
        "partial",
        "different_selection",
        "changed_helper",
        "changed_wrapper",
    ],
)
def test_full_conversion_requires_matching_pilot_and_native_approval(
    tmp_path, monkeypatch, failure
):
    import quantize_lfm25_settings_modal as module

    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "preparation", lambda: {"folder": str(tmp_path / "prepared")})
    root = tmp_path / "mcg-trunk-only"
    root.mkdir()
    approval = {
        "profile": "mcg-trunk-only",
        "native_sha256": module.NATIVE_SHA256,
        "settings_source_sha256": module.settings_sources(),
    }
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
    elif failure == "changed_wrapper":
        approval["settings_source_sha256"]["convert_lfm25_settings.py"] = "bad"
    (root / "probe-approved.json").write_text(json.dumps(approval))
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot/report.json").write_text(json.dumps(data))

    def convert(files, candidate):
        assert failure is None and files is None and candidate == "v3"
        command = module.base.conversion_command(Path("src"), Path("work"), Path("out"), False)
        assert command[command.index("--cal_data") + 1] == str(
            tmp_path / "prepared/original.safetensors"
        )
        assert command[command.index("-cb") + 1] == "mcg" and "--scaled_hessian" in command
        output = root / "v3"
        output.mkdir()
        return json.dumps({"status": "complete", "folder": str(output)})

    monkeypatch.setattr(module.base, "convert_model", convert)
    monkeypatch.setattr(module.volume, "commit", lambda: None)
    if failure:
        with pytest.raises(ValueError):
            module.convert_profile("mcg-trunk-only")
    else:
        result = json.loads(module.convert_profile("mcg-trunk-only"))
        assert result["status"] == "complete" and result["settings_profile"] == "mcg-trunk-only"


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


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("kind", ["gate", "up", "down", "dense", "conv"])
def test_projection_flags_follow_the_registered_type_table(profile, kind):
    keys = {
        "gate": "model.layers.2.feed_forward.experts.44.w1",
        "up": "model.layers.2.feed_forward.experts.44.w3",
        "down": "model.layers.2.feed_forward.experts.44.w2",
        "dense": "model.layers.0.feed_forward.w2",
        "conv": "model.layers.0.conv.out_proj",
    }
    enabled = {
        "mcg-standard": set(),
        "mcg-scaled": {"gate", "up", "down", "dense", "conv"},
        "mcg-no-down": {"gate", "up", "dense", "conv"},
        "mcg-trunk-only": {"dense", "conv"},
    }
    assert projection_scaled(profile, keys[kind]) is (kind in enabled[profile])


@pytest.mark.parametrize(
    "profile,key", [("unknown", "a"), ("mcg-scaled", ""), ("mcg-standard", None)]
)
def test_invalid_projection_selectors_fail(profile, key):
    with pytest.raises(ValueError):
        projection_scaled(profile, key)


@pytest.mark.parametrize("entry", ["quantize_linears_single", "quantize_linears_parallel"])
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("fails", [False, True])
def test_official_scalar_and_grouped_entry_keep_inputs_and_restore_on_error(entry, profile, fails):
    from types import SimpleNamespace

    from convert_lfm25_settings import converter_settings

    keys = [
        "model.layers.2.feed_forward.experts.44.w1",
        "model.layers.2.feed_forward.experts.44.w3",
        "model.layers.2.feed_forward.experts.44.w2",
        "model.layers.0.conv.out_proj",
    ]
    linears = [SimpleNamespace(key=k, qmap=("input" if i < 2 else k)) for i, k in enumerate(keys)]
    seen = []
    args, state, capture, sensitivity = {"codebook": "mcg", "other": object()}, object(), {}, {}
    before = args.copy()

    def original(actual, group, config, strategy, idx, devices, ratios, h, s, out_sens=None):
        assert s is state and h is capture and out_sens is sensitivity
        assert config == "config" and strategy == "strategy" and idx == 25
        assert devices == [0] and ratios is None
        assert actual["other"] is before["other"] and actual["codebook"] == "mcg"
        assert len({projection_scaled(profile, l.key) for l in group}) == 1
        assert actual["scaled_hessian"] == projection_scaled(profile, group[0].key)
        seen.extend(group)
        if fails:
            raise RuntimeError("official quantizer error")

    module = SimpleNamespace(quantize_linears_single=original, quantize_linears_parallel=original)
    try:
        with converter_settings(module, profile):
            getattr(module, entry)(
                args,
                linears,
                "config",
                "strategy",
                25,
                [0],
                None,
                capture,
                state,
                out_sens=sensitivity,
            )
    except RuntimeError:
        assert fails
    assert (
        module.quantize_linears_single is original and module.quantize_linears_parallel is original
    )
    assert args == before
    if not fails:
        assert sorted(map(id, seen)) == sorted(map(id, linears))


@pytest.mark.parametrize("failure", ["duplicate", "shared_hessian", "empty_key"])
def test_invalid_mixed_groups_fail_before_any_official_call(failure):
    from types import SimpleNamespace

    from convert_lfm25_settings import converter_settings

    first = SimpleNamespace(key="m.feed_forward.experts.0.w1", qmap="shared")
    second = SimpleNamespace(key="m.feed_forward.experts.0.w2", qmap="down")
    if failure == "duplicate":
        second.key = first.key
    elif failure == "shared_hessian":
        second.qmap = first.qmap
    else:
        second.key = ""
    original = lambda *a, **k: pytest.fail("invalid groups must not quantize")
    module = SimpleNamespace(quantize_linears_single=original, quantize_linears_parallel=original)
    with pytest.raises(ValueError), converter_settings(module, "mcg-no-down"):
        module.quantize_linears_single({}, [first, second], None, None, 0, [0], None, {}, None)
    assert module.quantize_linears_single is original


@pytest.mark.parametrize("failure", [None, "arguments", "codebook", "main"])
def test_wrapper_uses_official_parser_prepare_and_main(failure):
    import argparse
    from types import SimpleNamespace

    from convert_lfm25_settings import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-cb")
    parser.add_argument("-cr", type=int)
    original = lambda *a, **k: None
    seen = []

    def prepare(parsed):
        assert parsed.cr == 250
        return {"codebook": parsed.cb}, "state", failure != "arguments", "bad input"

    def main(args, state):
        assert args == {"codebook": "mcg"} and state == "state"
        assert module.quantize_linears_single is not original
        seen.append(True)
        if failure == "main":
            raise RuntimeError("conversion interrupted")

    module = SimpleNamespace(
        parser=parser,
        prepare=prepare,
        main=main,
        quantize_linears_single=original,
        quantize_linears_parallel=original,
    )
    arguments = ["-cb", "mul1" if failure == "codebook" else "mcg", "-cr", "250"]
    if failure:
        with pytest.raises(RuntimeError if failure == "main" else ValueError):
            run(module, "mcg-no-down", arguments)
    else:
        run(module, "mcg-no-down", arguments)
    assert bool(seen) == (failure in (None, "main"))
    assert (
        module.quantize_linears_single is original and module.quantize_linears_parallel is original
    )
