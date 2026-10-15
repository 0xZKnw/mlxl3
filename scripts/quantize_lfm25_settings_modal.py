"""Bounded H100 experiment using unmodified official EXL3 quantization settings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import modal
import quantize_lfm25_modal as base
from pilot_lfm25_settings import choose_settings, profile_options, settings_args

VOLUME = "mlxl3-lfm25-v3-20261015"
DATA = Path("/data/settings31")
PREPARATION = Path("/data/prepare25-complete.json")
NATIVE_SHA256 = "355b52620cfcf3e8a6d3b053efc9c28eda34963d8f132406e86aa733acdc2cd8"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
if modal.is_local():
    import quantize_lfm25_v3_modal as previous

    image = previous.image
    for name in ("pilot_lfm25_settings", "quantize_lfm25_settings_modal"):
        image = image.add_local_file(
            Path(__file__).with_name(f"{name}.py"), f"/opt/mlxl3/{name}.py", copy=True
        )
else:
    image = base.image
app = modal.App("mlxl3-lfm25-settings31", image=image)


def preparation() -> dict:
    data = json.loads(PREPARATION.read_text())
    if data.get("status") != "complete":
        raise ValueError("Missing completed source/calibration preparation")
    folder = Path(data["folder"])
    for name in ("original.safetensors", "validation.safetensors"):
        if (
            hashlib.sha256((folder / name).read_bytes()).hexdigest()
            != data["files"][name]["sha256"]
        ):
            raise ValueError("Calibration or validation changed")
    package = Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
    base.check_converter_sources(package)
    return data


def settings_command(source, work, output, probe, profile, original_command):
    codebook, scaled = profile_options(profile)
    command = original_command(source, work, output, probe)
    position = command.index("-cb")
    command[position : position + 2] = settings_args(codebook, scaled)
    return command


def convert_profile(profile: str, files: dict[str, bytes] | None = None) -> str:
    profile_options(profile)
    prepared = preparation()
    root = DATA / profile
    root.mkdir(parents=True, exist_ok=True)
    if files is None:
        approval = json.loads((root / "probe-approved.json").read_text())
        if approval.get("profile") != profile or approval.get("native_sha256") != NATIVE_SHA256:
            raise ValueError("Native approval belongs to another settings profile")
        result = json.loads((DATA / "pilot/report.json").read_text())
        selected = (result.get("selection") or {}).get("selected", {})
        reproduced = choose_settings(result)
        helper_sha256 = hashlib.sha256(
            Path(__file__).with_name("pilot_lfm25_settings.py").read_bytes()
        ).hexdigest()
        if (
            result.get("status") != "complete"
            or selected.get("profile") != profile
            or not reproduced
            or reproduced["selected"]["profile"] != profile
            or result.get("script_sha256") != helper_sha256
        ):
            raise ValueError("The complete pilot did not select this profile")
    original_command, original_logged = base.conversion_command, base.run_logged
    original_data, original_volume = base.DATA, base.volume

    def command(source, work, output, probe):
        values = settings_command(source, work, output, probe, profile, original_command)
        if not probe:
            values += ["--cal_data", str(Path(prepared["folder"]) / "original.safetensors")]
        return values

    base.DATA, base.volume = root, volume
    base.conversion_command = command
    base.run_logged = lambda c, log, timeout: original_logged(
        c, log, min(timeout, 180 if files is not None else 6000)
    )
    try:
        value = json.loads(base.convert_model(files, candidate="v3"))
        if value.get("status") != "complete":
            raise ValueError("Incomplete official conversion")
        value["settings_profile"] = profile
        (Path(value["folder"]) / "report.json").write_text(json.dumps(value, indent=2) + "\n")
        return json.dumps(value)
    finally:
        base.conversion_command, base.run_logged = original_command, original_logged
        base.DATA, base.volume = original_data, original_volume
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def pilot() -> str:
    prepared = preparation()
    folder = DATA
    folder.mkdir(exist_ok=True)
    report_path = folder / "pilot/report.json"
    if report_path.parent.exists():
        raise FileExistsError("Preserve the existing pilot attempt before starting another")
    try:
        command = [
            sys.executable,
            "-u",
            "/opt/mlxl3/pilot_lfm25_settings.py",
            prepared["source"],
            str(Path(prepared["folder"]) / "original.safetensors"),
            str(Path(prepared["folder"]) / "validation.safetensors"),
            str(folder / "pilot"),
        ]
        base.run_logged(command, folder / "pilot.log", 780)
        report = json.loads(report_path.read_text())
        if report.get("status") != "complete":
            raise ValueError("Incomplete settings pilot")
        return json.dumps(report)
    except BaseException as error:
        # Preserve detailed child failures; a killed/silent child must not leave a running verdict.
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status="failed", selection=None, launcher_error=repr(error))
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    finally:
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=240,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def probe(profile: str, files: dict[str, bytes]) -> str:
    return convert_profile(profile, files)


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=6300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def convert(profile: str) -> str:
    return convert_profile(profile)


@app.local_entrypoint()
def main(stage: str = "pilot", profile: str = "mul1-standard", probe_source: str = ""):
    if stage == "pilot":
        result = pilot.remote()
    elif stage == "probe" and probe_source:
        source = Path(probe_source)
        files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        result = probe.remote(profile, files)
    elif stage == "convert":
        result = convert.remote(profile)
    else:
        raise ValueError("Choose pilot, probe with source, or convert")
    destination = Path("build") / f"lfm25-settings31-{stage}-{profile}.json"
    destination.write_text(result + "\n")
    print(f"Report: {destination}")
