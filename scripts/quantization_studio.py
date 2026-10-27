"""Desktop's bounded JSON interface to the existing EXL3 adapters; no remote code."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

INFERENCE = {"lfm2", "lfm2_moe", "bailing_hybrid", "qwen3_5", "qwen3_5_moe", "qwen4_exp", "gemma4"}
CONVERSION = {
    "lfm2": "quantize_lfm26",
    "lfm2_moe": "quantize_lfm25_moe",
    "bailing_hybrid": "quantize_ling",
}
MAX_JSON = 8 * 1024 * 1024
MAX_FILES = 20_000
_JOB_RECORD: Path | None = None


def watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(0.2)
    if _JOB_RECORD is not None:
        try:
            state = read_json(_JOB_RECORD)
        except (OSError, ValueError, TypeError):
            state = {}
        state.update(status="interrupted", error="Desktop process exited")
        try:
            _JOB_RECORD.write_text(json.dumps(state, indent=2) + "\n")
        except OSError:
            pass
    os._exit(130)


def emit(kind: str, **values) -> None:
    sys.__stdout__.write(json.dumps({"type": kind, **values}, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def read_json(path: Path):
    with path.open("rb") as handle:
        data = handle.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError("JSON metadata exceeds 8 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("Expected JSON object")
    return value


def hub_reference(value: str) -> tuple[str, str]:
    """Accept model pages or owner/repo, never arbitrary download hosts.

    raises: ValueError
    post: len(__return__[0].split('/')) == 2
    """
    value = value.strip()
    revision = "main"
    if "://" in value:
        url = urlparse(value)
        if (
            url.scheme != "https"
            or url.netloc not in {"huggingface.co", "hf.co"}
            or url.query
            or url.fragment
        ):
            raise ValueError("Use a Hugging Face model URL or owner/repository")
        parts = unquote(url.path).strip("/").split("/")
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in {"tree", "resolve"}:
                raise ValueError("Unsupported Hugging Face URL")
            revision = parts[3]
        value = "/".join(parts[:2])
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value)
        or ".." in value
    ):
        raise ValueError("Expected owner/repository")
    if not revision or revision.startswith("-") or ".." in revision or "\x00" in revision:
        raise ValueError("Invalid revision")
    return value, revision


def settings(value: dict) -> dict:
    bits = value.get("bits", 4)
    head = value.get("head_bits", 6)
    rows = value.get("rows", 1024)
    sequence = value.get("sequence", 1024)
    if (
        isinstance(bits, bool)
        or not isinstance(bits, (int, float))
        or not math.isfinite(bits)
        or not 2 <= bits <= 8
    ):
        raise ValueError("BPW must be finite and between 2 and 8")
    for label, number, lower, upper in (
        ("head", head, 2, 8),
        ("rows", rows, 32, 8192),
        ("sequence", sequence, 32, 4096),
    ):
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError(f"Invalid {label}: expected integer in [{lower}, {upper}]")
    excluded = value.get("excluded", [])
    if (
        not isinstance(excluded, list)
        or len(excluded) > 128
        or not all(
            isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_.]+", p) and ".." not in p
            for p in excluded
        )
    ):
        raise ValueError("Exclusions must be module prefixes, one per line")
    keep_head = value.get("keep_head", False)
    if type(keep_head) is not bool:
        raise ValueError("Invalid head setting")
    for key in ("calibration", "validation"):
        path = value.get(key, "")
        if not isinstance(path, str) or "\0" in path:
            raise ValueError(f"Invalid {key} file")
        if path and (not Path(path).is_file() or Path(path).stat().st_size > 8 * 1024 * 1024):
            raise ValueError(f"{key} text must exist and be at most 8 MiB")
    if (
        value.get("validation")
        and value.get("calibration")
        and Path(value["validation"]).read_bytes() == Path(value["calibration"]).read_bytes()
    ):
        raise ValueError("Validation must use a separate held-out text")
    return {
        "bits": float(bits),
        "head_bits": head,
        "rows": rows,
        "sequence": sequence,
        "excluded": excluded,
        "keep_head": keep_head,
        "calibration": value.get("calibration", ""),
        "validation": value.get("validation", ""),
    }


def compatibility(config: dict, size: int, *, already_quantized: bool = False) -> dict:
    kind = config.get("model_type", "unknown")
    convertible = kind in CONVERSION and not already_quantized
    reason = (
        "EXL3 conversion adapter available"
        if convertible
        else "This architecture has no validated EXL3 conversion adapter in Desktop"
    )
    if already_quantized:
        reason = "Source is already quantized; choose original BF16/FP16 Safetensors weights"
    if kind in {"lfm2", "lfm2_moe"} and not config.get(
        "tie_word_embeddings", config.get("tie_embedding", False)
    ):
        convertible, reason = False, "This LFM adapter requires tied embeddings"
    if (
        kind == "lfm2"
        and isinstance(config.get("layer_types"), list)
        and not {"conv", "full_attention"}.issubset(config["layer_types"])
    ):
        convertible, reason = (
            False,
            "LFM calibration requires the native hybrid conv/attention layout",
        )
    return {
        "model_type": kind,
        "inference_supported": kind in INFERENCE,
        "conversion_supported": convertible,
        "reason": reason,
        "source_bytes": size,
        "tied_head": bool(config.get("tie_word_embeddings", config.get("tie_embedding", False))),
    }


def inspect_local(folder: Path) -> dict:
    from safetensors import safe_open

    folder = folder.resolve(strict=True)
    config_path = folder / "config.json"
    if config_path.is_symlink():
        raise ValueError("Linked model configuration refused")
    config = read_json(config_path)
    quantized = bool(
        config.get("quantization")
        or config.get("quantization_config")
        or (folder / "quantization_config.json").exists()
    )
    files = sorted(folder.glob("*.safetensors"))
    if not files or len(files) > MAX_FILES:
        raise ValueError(
            "Choose a complete model folder containing config.json and Safetensors weights"
        )
    tensors, owners, total, parameters = {}, {}, 0, 0
    identity = {"config": hashlib.sha256(config_path.read_bytes()).hexdigest(), "files": {}}
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        asset = folder / name
        if asset.is_symlink():
            raise ValueError("Linked model metadata refused")
        if asset.is_file():
            with asset.open("rb") as handle:
                identity[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    for file in files:
        if file.is_symlink() or not file.is_file():
            raise ValueError("Linked/non-regular weights refused")
        stat = file.stat()
        total += stat.st_size
        identity["files"][file.name] = [stat.st_size, stat.st_mtime_ns]
        with safe_open(file, framework="numpy") as handle:
            for key in handle.keys():  # noqa: SIM118
                if key in tensors or len(tensors) >= 200_000:
                    raise ValueError("Duplicate tensors or too many tensor entries")
                item = handle.get_slice(key)
                shape = item.get_shape()
                if len(shape) > 8 or any(type(x) is not int or x <= 0 for x in shape):
                    raise ValueError(f"Invalid tensor shape: {key}")
                if (
                    not quantized
                    and key.endswith(".weight")
                    and len(shape) == 2
                    and item.get_dtype() not in {"BF16", "F16", "F32"}
                ):
                    raise ValueError("Weights are not unquantized BF16/FP16/FP32")
                count = math.prod(shape)
                if count > 2**48:
                    raise ValueError("Unreasonable tensor shape")
                tensors[key] = shape
                owners[key] = file.name
                parameters += count
    index_path = folder / "model.safetensors.index.json"
    if index_path.exists():
        if index_path.is_symlink():
            raise ValueError("Linked index refused")
        index = read_json(index_path).get("weight_map")
        if (
            not isinstance(index, dict)
            or not index
            or any(
                not isinstance(name, str) or Path(name).name != name or owners.get(key) != name
                for key, name in index.items()
            )
        ):
            raise ValueError("Incomplete, unsafe or inconsistent Safetensors shard index")
        if set(index) != set(tensors):
            raise ValueError("Shard index does not match all source tensors")
    quantized = bool(
        config.get("quantization")
        or config.get("quantization_config")
        or (folder / "quantization_config.json").exists()
    )
    result = compatibility(config, total, already_quantized=quantized)
    if not (folder / "tokenizer.json").is_file():
        result.update(
            conversion_supported=False,
            reason="A local tokenizer.json is required; custom Python tokenizers are not executed",
        )
    result.update(
        source=str(folder),
        revision="local",
        parameters=parameters,
        fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        modules=sorted(
            key.removesuffix(".weight")
            for key, shape in tensors.items()
            if key.endswith(".weight") and len(shape) == 2
        ),
        header_checked=True,
    )
    return result


def probe(source: str) -> dict:
    if Path(source).expanduser().is_file() and Path(source).suffix == ".safetensors":
        return inspect_local(Path(source).expanduser().parent)
    if Path(source).expanduser().is_dir():
        return inspect_local(Path(source).expanduser())
    from huggingface_hub import HfApi, hf_hub_download

    repo, revision = hub_reference(source)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hugging Face returned no immutable source revision")
    config = read_json(Path(hf_hub_download(repo, "config.json", revision=info.sha)))
    files = info.siblings or []
    weights = [f for f in files if f.rfilename.endswith(".safetensors") and "/" not in f.rfilename]
    result = compatibility(
        config,
        sum(f.size or 0 for f in weights),
        already_quantized=bool(
            config.get("quantization")
            or config.get("quantization_config")
            or any(f.rfilename == "quantization_config.json" for f in files)
        ),
    )
    if (
        not weights
        or len(weights) > MAX_FILES
        or not any(f.rfilename == "tokenizer.json" for f in files)
    ):
        result.update(
            conversion_supported=False,
            reason="A complete Safetensors checkpoint with tokenizer.json is required",
        )
    result.update(
        source=repo,
        revision=info.sha,
        parameters=int(getattr(info.safetensors, "total", 0) or 0),
        fingerprint=info.sha,
        modules=[],
        header_checked=False,
    )
    return result


def download(source: dict, cache: Path) -> Path:
    if source["revision"] == "local":
        return Path(source["source"])
    from huggingface_hub import HfApi, hf_hub_download

    repo, revision = source["source"], source["revision"]
    files = HfApi().list_repo_files(repo, revision=revision)
    selected = [
        name
        for name in files
        if "/" not in name
        and (
            name.endswith((".safetensors", ".json", ".model", ".txt", ".jinja", ".md"))
            or name.startswith(("LICENSE", "NOTICE"))
        )
    ]
    if not selected or len(selected) > MAX_FILES:
        raise ValueError("Invalid source file list")
    for index, name in enumerate(selected):
        emit("progress", phase="download", completed=index, total=len(selected), text=name)
        hf_hub_download(repo, name, revision=revision, local_dir=cache)
    emit(
        "progress",
        phase="download",
        completed=len(selected),
        total=len(selected),
        text="Source downloaded",
    )
    return cache


class LogLines(io.TextIOBase):
    def __init__(self):
        self.pending = ""

    def write(self, text):
        self.pending += text.replace("\r", "\n")
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                emit("log", text=line[:4000])
                match = re.match(r"\[e2e:(calib|measure|convert)\].*?(\d+)/(\d+)", line)
                if match:
                    emit(
                        "progress",
                        phase=match[1],
                        completed=int(match[2]),
                        total=int(match[3]),
                        text=line[:500],
                    )
        self.pending = self.pending[-4000:]
        return len(text)

    def flush(self):
        if self.pending.strip():
            emit("log", text=self.pending[:4000])
        self.pending = ""


def validate_quality(source: Path, output: Path, text: Path, work: Path, native: Path) -> dict:
    import mlx.core as mx
    import numpy as np
    from eval_lfm25_quality import native_window
    from mlx_lm.utils import load
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(source / "tokenizer.json"))
    tokens = tokenizer.encode(text.read_text(encoding="utf-8"))[:257]
    if len(tokens) < 33:
        raise ValueError("Held-out validation requires at least 33 tokens")
    vocabulary = read_json(source / "config.json").get("vocab_size", 0)
    if not 1 < vocabulary <= 262_144 or any(
        type(t) is not int or not 0 <= t < vocabulary for t in tokens
    ):
        raise ValueError("Invalid validation vocabulary/tokens")
    emit("progress", phase="validation", completed=0, total=2, text="source teacher")
    model = load(str(source), tokenizer_config={"trust_remote_code": False})[0]
    logits = model(mx.array([tokens[:-1]]))
    mx.eval(logits)
    values = np.asarray(logits[0], dtype=np.float32)
    if values.shape != (len(tokens) - 1, vocabulary) or not np.isfinite(values).all():
        raise ValueError("Incomplete or nonfinite teacher logits")
    teacher_path = work / "teacher.npy"
    np.save(teacher_path, values, allow_pickle=False)
    del values, logits, model
    gc.collect()
    mx.clear_cache()
    emit("progress", phase="validation", completed=1, total=2, text="native EXL3 engine")
    teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    result = native_window(
        native, output, tokens[:-1], tokens[1:], teacher, work / "native-validation.log", 120
    )
    metrics = {
        "tokens": result["tokens"],
        "ppl": math.exp(result["nll"]),
        "teacher_ppl": math.exp(result["bf16_nll"]),
        "kl": result["kl"],
        "text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "scope": "one held-out window, at most 256 predicted tokens; one GPU model at a time",
    }
    if (
        not all(math.isfinite(metrics[key]) for key in ("ppl", "teacher_ppl", "kl"))
        or metrics["kl"] < -1e-7
    ):
        raise ValueError("Invalid quality metrics")
    return metrics


def convert(request: dict) -> dict:
    global _JOB_RECORD
    import importlib

    options = settings(request.get("settings", {}))
    native = Path(request.get("engine", ""))
    if options["validation"] and (not native.is_file() or not os.access(native, os.X_OK)):
        raise ValueError("Native MLXL3 engine is required for KL/perplexity validation")
    if options["validation"] and request["source"]["model_type"] == "bailing_hybrid":
        raise ValueError(
            "Full teacher KL/perplexity validation is not available for this conversion profile"
        )
    expected = request["source"]
    destination = Path(request["output"]).expanduser().absolute()
    if destination.exists() or not destination.parent.is_dir() or destination.is_symlink():
        raise ValueError("Choose a new output folder; existing files are never overwritten")
    physical_memory = (
        os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if sys.platform != "darwin"
        else int(
            __import__("subprocess").check_output(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=3
            )
        )
    )
    if options["validation"] and expected["source_bytes"] > physical_memory * 0.65:
        raise ValueError(
            "Full teacher validation will not fit in this Mac's RAM; leave KL/PPL disabled"
        )
    if expected["model_type"] == "lfm2" and expected["source_bytes"] > physical_memory * 0.65:
        raise ValueError(
            "Dense calibration needs the source model in RAM; this model is too large for this Mac"
        )
    if shutil.disk_usage(destination.parent).free < max(expected["source_bytes"], 64 * 1024 * 1024):
        raise ValueError("Insufficient free disk space for conversion and calibration")
    jobs = Path(request["jobs"])
    jobs.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="conversion-", dir=jobs))
    report = {
        "status": "running",
        "source": expected,
        "settings": options,
        "output": str(destination),
    }

    def save():
        temporary = job / "job.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(job / "job.json")

    _JOB_RECORD = job / "job.json"
    save()
    emit("job", path=str(job))
    try:
        source = download(expected, job / "source")
        actual = inspect_local(source)
        if not actual["conversion_supported"] or actual["model_type"] != expected["model_type"]:
            raise ValueError(actual["reason"])
        if expected["revision"] == "local" and actual["fingerprint"] != expected["fingerprint"]:
            raise ValueError("Source changed after inspection; inspect again")
        if destination.resolve().is_relative_to(
            source.resolve()
        ) or source.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Output and source folders must be separate")
        exclusions = options["excluded"] + (["lm_head", "head"] if options["keep_head"] else [])
        for prefix in options["excluded"]:
            if not any(key == prefix or key.startswith(prefix + ".") for key in actual["modules"]):
                raise ValueError(f"Exclusion matches no source module: {prefix}")
        os.environ["MLXL3_QUANT_EXCLUDE_PREFIXES"] = json.dumps(exclusions)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx_lm.utils

        original_load = mlx_lm.utils.load

        def safe_load(*args, **kwargs):
            kwargs["tokenizer_config"] = {
                **kwargs.get("tokenizer_config", {}),
                "trust_remote_code": False,
            }
            return original_load(*args, **kwargs)

        mlx_lm.utils.load = safe_load
        adapter = importlib.import_module(CONVERSION[actual["model_type"]])
        arguments = [
            "mlxl3-quantize",
            "--in-dir",
            str(source),
            "--out-dir",
            str(destination),
            "--work-dir",
            str(job / "work"),
            "--bits",
            str(options["bits"]),
            "--head-bits",
            str(options["head_bits"]),
            "--calibration-rows",
            str(options["rows"]),
            "--calibration-seq-len",
            str(options["sequence"]),
            "--max-workers",
            "1",
            "--search-backend",
            "metal",
            "--codebook",
            "mcg",
        ]
        if options["calibration"]:
            arguments += ["--calibration-text", options["calibration"]]
        sys.argv = arguments
        emit(
            "progress",
            phase="calibration",
            completed=0,
            total=0,
            text="Capturing calibration; source weights stay unchanged",
        )
        logs = LogLines()
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            result = adapter.main()
            logs.flush()
        if result != 0:
            raise ValueError(f"Converter failed with status {result}")
        from ponyexl3.convert import reuse

        reuse.disable()
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()
        from mlxl3_quantizer import validate_checkpoint_files

        validate_checkpoint_files(destination)
        if inspect_local(source)["fingerprint"] != actual["fingerprint"]:
            raise ValueError("Source changed during conversion; output remains unregistered")
        if options["validation"]:
            report["quality"] = validate_quality(
                source, destination, Path(options["validation"]), job, native
            )
        report.update(
            status="complete",
            output_bytes=sum(p.stat().st_size for p in destination.iterdir() if p.is_file()),
        )
        save()
        return report
    except BaseException as error:
        report.update(
            status="interrupted"
            if isinstance(error, (KeyboardInterrupt, InterruptedError))
            else "failed",
            error=str(error),
            traceback=traceback.format_exc(),
        )
        save()
        raise


def main():
    threading.Thread(
        target=watch_parent,
        args=(int(os.environ.get("MLXL3_OWNER_PID", str(os.getppid()))),),
        daemon=True,
    ).start()

    def interrupted(*_):
        raise InterruptedError("Conversion stopped")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        data = sys.stdin.buffer.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError("Request too large")
        request = json.loads(data)
        if request.get("operation") == "probe":
            emit("result", result=probe(request["source"]))
        elif request.get("operation") == "convert":
            emit("result", result=convert(request))
        else:
            raise ValueError("Unknown quantization operation")
    except BaseException as error:  # noqa: BLE001 - adapter SystemExit must remain a structured failure
        emit("error", text=str(error)[:4000])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
