"""Prepare the registered calibration ablation, then explicitly launch its selected candidate."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import modal
import quantize_lfm25_modal as base

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "mlxl3-lfm25-v3-20261015"
TRACE_SOURCES = {
    "sc_trace.py": "fb268f9c6ab1453475c2b7d32cf7c23b9e9c428ede400679a9dbd07d14022edc",
    "eval/qbench_prompts.py": "83fc2c123fe8bd1af4540db91cd88637b1fb7aba49433aa8f0e14823d567801a",
}


def fetch_trace_sources(destination: str = "/opt/exl3-source"):
    from urllib.request import urlopen

    for name, digest in TRACE_SOURCES.items():
        url = f"https://raw.githubusercontent.com/turboderp-org/exllamav3/{base.CONVERTER}/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Wrong pinned trace source: {name}")
        path = Path(destination) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def trace_runtime_smoke():
    import tempfile

    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="mlxl3-chat-preflight-") as directory:
        tokenizer = AutoTokenizer.from_pretrained(
            base.SOURCE, revision=base.REVISION, trust_remote_code=False, cache_dir=directory
        )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Bonjour."}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if not tokens or any(type(t) is not int or not 0 <= t < len(tokenizer) for t in tokens):
        raise ValueError("Invalid source chat-template preflight")
    if not modal.is_local() and Path("/data").exists() and any(Path("/data").iterdir()):
        raise ValueError("Image preflight populated the reserved Volume mount")
    print(f"Source chat template: {len(tokens)} valid tokens; no GPU/model weights loaded")


volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = base.image.pip_install("transformers==5.16.1").env(
    {
        "PYTHONPATH": "/opt/mlxl3:/opt/exl3-source",
        "HF_HOME": "/tmp/mlxl3-image-hf",
        "HF_XET_CACHE": "/tmp/mlxl3-image-xet",
    }
)
for name in ("quantize_lfm25_modal", "exl3_moe_calibration", "pilot_lfm25_calibration"):
    image = image.add_local_file(
        ROOT / "scripts" / f"{name}.py", f"/opt/mlxl3/{name}.py", copy=True
    )
image = (
    image.run_function(fetch_trace_sources)
    .run_function(trace_runtime_smoke)
    .env({"HF_HOME": "/data/hf", "HF_XET_CACHE": "/data/hf/xet"})
)
app = modal.App("mlxl3-lfm25-v3", image=image)


@app.function(
    cpu=1,
    memory=1024,
    timeout=60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def check_mount():
    import importlib.metadata
    import os

    if not Path("/data").is_dir() or os.environ["HF_HOME"] != "/data/hf":
        raise ValueError("Runtime cache or Volume mount is incorrect")
    package = Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
    return {
        "status": "complete",
        "stage": "mount",
        "gpu_requested": False,
        "transformers": importlib.metadata.version("transformers"),
        "converter_sources": base.check_converter_sources(package),
    }


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ngrams(rows, width=13):
    if width < 1:
        raise ValueError("Positive ngram width required")
    return {gram for row in rows for gram in zip(*(row[i:] for i in range(width)))}


def make_mix(original_path: Path, trace_path: Path, output: Path):
    import torch
    from safetensors.torch import load_file, save_file

    original = load_file(str(original_path))
    if set(original) != {f"tensor.{i}" for i in range(250)}:
        raise ValueError("Expected the exact original 250-row calibration")
    rows = [original[f"tensor.{i}"] for i in range(250)]
    if any(row.shape != (1, 2048) or row.dtype != torch.int64 for row in rows):
        raise ValueError("Invalid original calibration rows")
    old = torch.cat(rows)
    trace = json.loads(trace_path.read_text())["rows"]
    order = list(range(len(trace)))
    random.Random(20261015).shuffle(order)
    replacements = []
    for i in order:
        tokens = trace[i]["input_ids"] + trace[i]["response_ids"]
        if len(tokens) >= 2048:
            replacements.append(torch.tensor(tokens[:2048], dtype=torch.int64)[None])
        if len(replacements) == 39:
            break
    if len(replacements) != 39:
        raise ValueError(f"Only {len(replacements)} full trace rows; expected 39")
    new = torch.cat(rows[:211] + replacements)
    if (
        old.min().item() < 0
        or new.min().item() < 0
        or max(old.max().item(), new.max().item()) >= 125017
    ):
        raise ValueError("Calibration IDs outside the source tokenizer")
    if old.shape != new.shape or not torch.equal(old[:211], new[:211]):
        raise ValueError("Calibration replacement modified retained text rows")
    save_file({"input_ids": old}, str(output / "original.safetensors"))
    save_file({"input_ids": new}, str(output / "trace-mix.safetensors"))
    return old.tolist(), new.tolist()


def build_validation(package: Path, tokenizer_path: Path, forbidden, output: Path):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, domains = [], []
    for name in ("c4", "code", "multilingual", "technical", "wiki", "tiny"):
        text = (package / "conversion/standard_cal_data" / f"{name}.utf8").read_text()
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        found = 0
        for end in range(len(tokens), 1023, -1024):
            row = tokens[end - 1024 : end]
            if not ngrams([row]).isdisjoint(forbidden):
                continue
            rows.append(row)
            domains.append(name)
            found += 1
            if found == 2:
                break
        if found != 2:
            raise ValueError(f"No two independent validation rows for {name}")
    save_file(
        {"input_ids": torch.tensor(rows, dtype=torch.int64)}, str(output / "validation.safetensors")
    )
    return domains


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def prepare(original_ids: bytes):
    import importlib.metadata

    from huggingface_hub import snapshot_download

    complete = Path("/data/prepare25-complete.json")
    if complete.exists():
        raise FileExistsError("A completed calibration preparation already exists")
    folder = Path(f"/data/prepare25-{time.time_ns()}")
    folder.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "publication": False,
        "stage": "prepare",
        "converter": base.CONVERTER,
        "folder": str(folder),
    }
    started = time.monotonic()
    try:
        package = Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
        report["source_hashes"] = base.check_converter_sources(package)
        (folder / "original-ids.safetensors").write_bytes(original_ids)
        if (
            file_sha(folder / "original-ids.safetensors")
            != "10a4a6eec1da405c002dbb29d3ab053d9da061ba6f9ea061461849b0d0b7e391"
        ):
            raise ValueError("Wrong original v2 calibration bytes")
        source = Path(
            snapshot_download(
                base.SOURCE,
                revision=base.REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE*"],
            )
        )
        report["source"] = str(source)
        command = [
            sys.executable,
            "-u",
            "/opt/exl3-source/sc_trace.py",
            "-m",
            str(source),
            "-o",
            str(folder / "trace.json"),
            "-co",
            str(folder / "trace.safetensors"),
            "-cr",
            "96",
            "-cc",
            "2048",
            "--target_tokens",
            "196608",
            "--max_new_tokens",
            "3072",
            "--max_epochs",
            "1",
            "--seed",
            "20261015",
            "-cs",
            "131072",
            "-ambs",
            "8",
        ]
        report["trace_command"] = command
        base.run_logged(command, folder / "trace.log", 1000)
        old, new = make_mix(folder / "original-ids.safetensors", folder / "trace.json", folder)
        report["calibration"] = {
            "rows": 250,
            "tokens": 512000,
            "retained_text_rows": 211,
            "replaced_random_rows": 39,
        }
        report["validation_domains"] = build_validation(
            package, source / "tokenizer.json", ngrams(old + new), folder
        )
        command = [
            sys.executable,
            "-u",
            "/opt/mlxl3/pilot_lfm25_calibration.py",
            str(source),
            str(folder / "original.safetensors"),
            str(folder / "trace-mix.safetensors"),
            str(folder / "validation.safetensors"),
            str(folder / "pilot"),
        ]
        base.run_logged(
            command, folder / "pilot.log", min(600, max(1, 1740 - (time.monotonic() - started)))
        )
        report["pilot"] = json.loads((folder / "pilot/report.json").read_text())
        if report["pilot"]["status"] != "complete":
            raise ValueError("Incomplete pilot")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["files"] = {
            str(p.relative_to(folder)): {"sha256": file_sha(p), "bytes": p.stat().st_size}
            for p in folder.rglob("*")
            if p.is_file() and p.name != "report.json"
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if report["status"] == "complete":
            with complete.open("x") as handle:
                handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()
    return report


def convert_candidate(
    mode: str, corpus: str, trunk8: bool = False, probe_files: dict[str, bytes] | None = None
):
    import os

    from exl3_moe_calibration import MODES

    if (
        mode not in (*MODES, "all")
        or corpus not in ("original", "trace")
        or type(trunk8) is not bool
    ):
        raise ValueError("Unknown registered conversion variant")
    if probe_files is None:
        approval = json.loads(Path("/data/probe-approved.json").read_text())
        if (
            approval.get("mode") != mode
            or approval.get("helper_sha256") != file_sha(Path("/opt/mlxl3/exl3_moe_calibration.py"))
            or approval.get("corpus") != corpus
            or approval.get("trunk8") is not trunk8
        ):
            raise ValueError("Native probe approval does not match these calibration hooks")
        preparation = json.loads(Path("/data/prepare25-complete.json").read_text())
        if preparation["status"] != "complete":
            raise ValueError("Calibration pilot not complete")
        name = ("original" if corpus == "original" else "trace-mix") + ".safetensors"
        if file_sha(Path(preparation["folder"]) / name) != preparation["files"][name]["sha256"]:
            raise ValueError("Calibration file changed after the pilot")
    base.volume = volume
    original_command = base.conversion_command
    original_logged = base.run_logged
    os.environ["MLXL3_MOE_CALIBRATION"] = mode
    os.environ["MLXL3_MOE_MINIMUM"] = "1" if probe_files is not None else "256"
    os.environ["MLXL3_MOE_COVERAGE"] = f"/data/coverage-{time.time_ns()}.jsonl"

    def command(source, work, output, probe):
        values = original_command(source, work, output, probe)
        if mode != "all":
            values[2:4] = ["/opt/mlxl3/exl3_moe_calibration.py"]
        if not probe:
            path = Path(preparation["folder"]) / (
                ("original" if corpus == "original" else "trace-mix") + ".safetensors"
            )
            values += ["--cal_data", str(path)]
        else:
            import torch
            from safetensors.torch import save_file

            if json.loads((source / "config.json").read_text())["vocab_size"] != 128:
                raise ValueError("The native probe requires its 128-token fixture")
            calibration = work.parent / "probe-calibration.safetensors"
            save_file(
                {"input_ids": torch.arange(128, dtype=torch.long).repeat(16, 1)}, str(calibration)
            )
            values += ["--cal_data", str(calibration)]
        if trunk8:
            import yaml
            from exllamav3 import Config, Model
            from exllamav3.modules.linear import Linear

            model = Model.from_config(Config.from_directory(str(source)))
            tensors = {
                linear.key: 4 if ".experts." in linear.key else 8
                for linear in model
                if isinstance(linear, Linear) and linear.qmap and linear.qbits_key == "bits"
            }
            recipe = work.parent / "trunk8.yaml"
            recipe.write_text(
                yaml.safe_dump({"target_bpw": 4.0, "head_bits": 16, "tensors": tensors})
            )
            values += ["--recipe", str(recipe)]
        return values

    base.conversion_command = command
    base.run_logged = lambda cmd, log, timeout: original_logged(cmd, log, min(timeout, 10200))
    result = None
    try:
        result = json.loads(base.convert_model(probe_files, candidate="v3"))
        result["calibration_recipe"] = {"mode": mode, "corpus": corpus, "trunk8": trunk8}
        if mode != "all":
            path = Path(os.environ["MLXL3_MOE_COVERAGE"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            config = json.loads((Path(result["folder"]) / "output/config.json").read_text())
            expected = (
                2
                * config["num_experts"]
                * (config["num_hidden_layers"] - config["num_dense_layers"])
            )
            if len(rows) != expected or len({r["key"] for r in rows}) != expected:
                raise ValueError("Incomplete or duplicate expert coverage audit")
            result["coverage"] = {
                "path": str(path),
                "sha256": file_sha(path),
                "hessians": len(rows),
                "minimum_rows": min(r["count"] for r in rows),
                "minimum_effective_rows": min(r["effective_rows"] for r in rows),
            }
        (Path(result["folder"]) / "report.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return json.dumps(result)
    except BaseException as error:
        if result is not None:
            result.update(status="failed", error=repr(error))
            (Path(result["folder"]) / "report.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        raise
    finally:
        base.conversion_command = original_command
        base.run_logged = original_logged
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=10800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def convert(mode: str, corpus: str, trunk8: bool = False):
    return convert_candidate(mode, corpus, trunk8)


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
def probe(mode: str, corpus: str, trunk8: bool, files: dict[str, bytes]):
    return convert_candidate(mode, corpus, trunk8, files)


@app.local_entrypoint()
def main(stage: str = "prepare", mode: str = "", corpus: str = "", trunk8: bool = False):
    if stage == "mount":
        result = check_mount.remote()
    elif stage == "prepare":
        result = prepare.remote((ROOT / "build/v2-calibration-input-ids.safetensors").read_bytes())
    elif stage == "probe":
        source = ROOT / "build/lfm25-v3-probe-source"
        files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        result = json.loads(probe.remote(mode, corpus, trunk8, files))
    elif stage == "convert":
        result = json.loads(convert.remote(mode, corpus, trunk8))
    else:
        raise ValueError("stage must be mount, prepare, probe, or convert")
    destination = ROOT / "build" / f"lfm25-v3-{stage}-cloud.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result.get("status"), "report": str(destination)}), flush=True)
