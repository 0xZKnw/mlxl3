"""Pinned CUDA conversion on one Modal H100; candidates stay on a private Volume.

modal run scripts/quantize_lfm25_modal.py --probe-source /path/to/tiny/source
Approve the downloaded probe in MLXL3 before running without --probe-source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

SOURCE = "LiquidAI/LFM2.5-24B-A2B"
REVISION = "1769516ac588c4c558089c94f27d96b0dc67b183"
CONVERTER = "16a49792a3c93d8432d72e6c4bce800841566577"
SOURCE_HASHES = {
    "architecture/lfm2_moe.py": "7a2672ec64de9e00d13d535c636a874e8f9fc632f42b7ecc16cb6c130fac9a29",
    "conversion/convert_model.py": "c35bdf05bbc13db12b7016427c730092870909c798b75c55eb63a41c760693fb",
    "modules/quant/exl3_lib/quantize.py": "a951e6707361c592cb1d1d17790e098e1a304b410b7c9225dcaaa1f26117562e",
    "modules/block_sparse_mlp_routing.py": "b878646d14cc4b7cb340de2c6e5e8d14c23c9307fc609b457b53e4b6956b5fa4",
    "modules/block_sparse_mlp.py": "ec05c84fc6213221797256568a0f05d43b07737ef1c36a320f1926b254e3ecd7",
}
WHEEL = (
    "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.4/"
    "exllamav3-1.5.4%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
    "#sha256=32b253a84612fae5bfd3787a3fc8a81450b976c59b3d6d29c2994fe5b451c2bc"
)
VOLUME = "mlxl3-lfm25-v2-20261015"
DATA = Path("/data")


def check_converter_sources(directory: Path) -> dict[str, str]:
    actual = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_HASHES
    }
    if actual != SOURCE_HASHES:
        raise ValueError("Installed converter sources differ from the pinned Git revision")
    return actual


def conversion_command(source: Path, work: Path, output: Path, probe: bool) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "exllamav3.conversion.convert_model",
        "-i",
        str(source),
        "-w",
        str(work),
        "-o",
        str(output),
        "-b",
        "4",
        "-hb",
        "16",
        "-cb",
        "mcg",
        "--out_scales",
        "auto",
        "--hq",
        "-cr",
        "16" if probe else "250",
        "-cc",
        "128" if probe else "2048",
        "-ss",
        "4096",
        "-cpi",
        "120",
    ]


def run_logged(command: list[str], log: Path, timeout: float) -> None:
    """Tee live output, and reap the entire child group on failure or timeout."""
    shell = shlex.join(command) + " 2>&1 | tee " + shlex.quote(str(log))
    with subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", shell], start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        WHEEL,
        "numpy==2.5.2",
        "tokenizers==0.23.1",
        "safetensors==0.8.0",
        "huggingface_hub==1.29.0",
        "rich==15.0.0",
        "typing_extensions==4.15.0",
        "ninja==1.13.0",
        "pillow==12.0.0",
        "pyyaml==6.0.3",
        "marisa-trie==1.3.1",
        "pydantic==2.12.5",
        "llguidance==1.7.5",
    )
    .env({"HF_HOME": "/data/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("mlxl3-lfm25-v2", image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def convert_model(probe_files: dict[str, bytes] | None = None, candidate: str = "v2") -> str:
    if candidate not in ("v2", "v3"):
        raise ValueError("Unknown candidate version")
    probe = probe_files is not None
    folder = DATA / (f"probe-{time.time_ns()}" if probe else candidate)
    folder.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "probe": probe,
        "source": SOURCE,
        "revision": REVISION,
        "converter_commit": CONVERTER,
        "publication": False,
        "folder": str(folder),
    }
    started = time.monotonic()
    try:
        if not probe:
            approval = json.loads((DATA / "probe-approved.json").read_text())
            if approval.get("status") != "passed" or approval.get("converter_commit") != CONVERTER:
                raise ValueError("Native probe approval is missing or for a different converter")
        import importlib.metadata

        import torch
        from huggingface_hub import snapshot_download

        report["converter_sources"] = check_converter_sources(
            Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        report["gpu"] = torch.cuda.get_device_name(0)
        report["versions"] = dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        )
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        volume.commit()
        if probe:
            source = folder / "source"
            source.mkdir()
            for name, data in probe_files.items():
                if Path(name).name != name:
                    raise ValueError("Probe input must contain flat filenames")
                (source / name).write_bytes(data)
        else:
            source = Path(
                snapshot_download(
                    SOURCE,
                    revision=REVISION,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.md",
                        "*.txt",
                        "*.jinja",
                        "LICENSE*",
                    ],
                )
            )
        command = conversion_command(source, folder / "work", folder / "output", probe)
        report["command"] = command
        print(json.dumps(report), flush=True)
        run_logged(command, folder / "convert.log", 840 if probe else 13800)
        output = folder / "output"
        descriptor = json.loads((output / "quantization_config.json").read_text())
        if descriptor.get("quant_method") != "exl3" or not descriptor.get("tensor_storage"):
            raise ValueError("Converter did not produce a complete EXL3 descriptor")
        if probe:
            import numpy as np
            from exllamav3 import Cache, Config, Model

            config = Config.from_directory(str(output))
            model = Model.from_config(config)
            cache = Cache(model, max_num_tokens=256, max_batch_size=1)
            model.load(device="cuda:0", max_output_size=32)
            tokens = list(range(4, 36))
            params = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": 0,
                "batch_shape": (1, 256),
            }
            rows = []
            with torch.inference_mode():
                for token in tokens:
                    logits = model.forward(torch.tensor([[token]]), params)
                    rows.append(logits.float().cpu().numpy().reshape(-1))
                    params["past_len"] += 1
            values = np.stack(rows)
            if values.shape != (32, 128) or not np.isfinite(values).all():
                raise ValueError("Invalid CUDA probe logits")
            np.save(folder / "cuda-logits.npy", values)
            (folder / "tokens.json").write_text(json.dumps(tokens))
            model.unload()
        report["files"] = {}
        for p in sorted(output.iterdir()):
            if p.is_file():
                with p.open("rb") as handle:
                    report["files"][p.name] = {
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                    }
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()
    return json.dumps(report)


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={DATA: volume},
)
def probe(files: dict[str, bytes]) -> str:
    return convert_model(files)


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=14400,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={DATA: volume},
)
def convert() -> str:
    return convert_model()


@app.local_entrypoint()
def main(probe_source: str = ""):
    files = None
    if probe_source:
        files = {p.name: p.read_bytes() for p in Path(probe_source).iterdir() if p.is_file()}
        if not files or sum(map(len, files.values())) > 32 * 1024**2:
            raise ValueError("Probe must be a nonempty fixture of at most32MiB")
    print(probe.remote(files) if files is not None else convert.remote())
