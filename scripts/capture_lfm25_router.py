"""Capture actual BF16/EXL3 router tensors and normal logits for frozen experiment38."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from contextlib import AbstractContextManager
from pathlib import Path

from diagnose_lfm25_reference import load_source
from diagnose_lfm25_routes import V2_SHARDS, artifact
from lfm25_router_plan import validate_plan

LAYERS = tuple(range(2, 40))
VOCABULARY = 128000
HIDDEN = 2048
EXPERTS = 64
PHASES = ("train", "validation", "final")
V2_CONFIGS = {
    "config.json": "d6cb556687ddd8382a90cbb957b4fbbdf95ed607bbc4800dc85fd5e5510edade",
    "quantization_config.json": "71908578d6be39abe2e2839ae2adfa087f9ea74aaaa51a6812ff5950d48ae61f",
}


def capture_position(index: int, completed: int, total: int) -> int:
    """Accept only the next unfilled position of a nonempty capture.

    raises: ValueError
    post: 0 <= __return__ < total
    post: __return__ == index == completed
    """
    if (
        type(index) is not int
        or type(completed) is not int
        or type(total) is not int
        or not 0 <= index == completed < total
    ):
        raise ValueError("Out-of-order router capture position")
    return index


def cache_capacity(tokens: int) -> int:
    """Round a nonempty sequence up to the actual EXL3 cache page boundary.

    raises: ValueError
    post: tokens <= __return__ < tokens + 256
    post: __return__ % 256 == 0
    """
    if type(tokens) is not int or tokens < 1:
        raise ValueError("Cache requires a positive token count")
    return ((tokens + 255) // 256) * 256


def atomic_report(folder: Path, report: dict) -> None:
    temporary = folder / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(folder / "report.json")


class _Capture(AbstractContextManager):
    def __init__(self, modules: dict, rows: int):
        if tuple(modules) != LAYERS or type(rows) is not int or rows < 1:
            raise ValueError("Expected the ordered 38 actual MoE layers and positive row count")
        self.modules, self.rows = modules, rows
        self.buffers, self.originals, self.handles = {}, {}, []
        self.position, self.completed, self.seen = None, 0, set()

    def begin(self, index: int) -> None:
        if self.position is not None:
            raise ValueError("Previous router token has not finished")
        self.position = capture_position(index, self.completed, self.rows)
        self.seen = set()

    def _observe(self, layer: int) -> int:
        if self.position is None or layer in self.seen:
            raise ValueError("Unexpected or repeated actual router call")
        self.seen.add(layer)
        return self.position

    def finish(self) -> None:
        if self.position is None or self.seen != set(self.modules):
            raise ValueError("Missing actual router layer in token forward")
        self.position = None
        self.completed += 1

    def payload(self, layer: int) -> dict:
        import torch

        if self.position is not None or self.completed != self.rows or layer not in self.buffers:
            raise ValueError("Refusing incomplete router capture")
        result = {key: value.detach().cpu().clone() for key, value in self.buffers[layer].items()}
        if not result or any(not torch.isfinite(value).all().item() for value in result.values()):
            raise ValueError("Nonfinite actual router capture")
        return result

    def __exit__(self, *_):
        for layer, original in self.originals.items():
            self.modules[layer].routing_fn = original
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if _ and _[0] is not None:
            self.buffers.clear()
        return False


class QuantCapture(_Capture):
    """Observe the exact buffers consumed by the fused MoE; never replace outputs."""

    def __init__(self, modules: dict, rows: int, capture_inputs: bool = True):
        super().__init__(modules, rows)
        self.capture_inputs = capture_inputs

    def __enter__(self):
        import torch

        if self.buffers:
            raise ValueError("Router captures are single-use contexts")
        try:
            for layer, module in self.modules.items():
                self.buffers[layer] = {
                    "zq": torch.empty(
                        (self.rows, EXPERTS), dtype=torch.float16, device=module.device
                    )
                }
                if self.capture_inputs:
                    self.buffers[layer]["x"] = torch.empty(
                        (self.rows, HIDDEN), dtype=torch.float16, device=module.device
                    )
                original = module.routing_fn
                self.originals[layer] = original

                def capture(bsz, cfg, hidden, params, *, layer=layer, original=original):
                    position = self._observe(layer)
                    if (
                        bsz != 1
                        or hidden.numel() != HIDDEN
                        or hidden.shape[-1] != HIDDEN
                        or hidden.dtype != torch.float16
                    ):
                        raise ValueError("Expected one actual normalized F16 router input")
                    if self.capture_inputs:
                        self.buffers[layer]["x"][position].copy_(hidden.reshape(HIDDEN).detach())
                    result = original(bsz, cfg, hidden, params)
                    ids, weights = result
                    logits = cfg.router_logits_bsz1
                    if (
                        logits.shape != (1, EXPERTS)
                        or logits.dtype != torch.float16
                        or ids.shape != (1, 4)
                        or ids.dtype != torch.int64
                        or weights.shape != (1, 4)
                        or weights.dtype != torch.float16
                    ):
                        raise ValueError("Actual quantized router output contract changed")
                    self.buffers[layer]["zq"][position].copy_(logits[0].detach())
                    return result

                module.routing_fn = capture
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise


class TeacherCapture(_Capture):
    """Observe the actual Lfm2MoeTopKRouter output, without another gate matmul."""

    def __enter__(self):
        import torch

        if self.buffers:
            raise ValueError("Router captures are single-use contexts")
        try:
            for layer, gate in self.modules.items():
                self.buffers[layer] = {
                    "zsource": torch.empty(
                        (self.rows, EXPERTS), dtype=torch.bfloat16, device=gate.weight.device
                    )
                }

                def capture(module, args, result, *, layer=layer):
                    position = self._observe(layer)
                    if not isinstance(result, tuple) or len(result) != 3:
                        raise ValueError("Expected actual teacher router tuple")
                    logits = result[0]
                    if logits.shape != (1, EXPERTS) or logits.dtype != torch.bfloat16:
                        raise ValueError("Expected actual BF16 full64 teacher router logits")
                    self.buffers[layer]["zsource"][position].copy_(logits[0].detach())

                self.handles.append(gate.register_forward_hook(capture))
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise


def phase_sequences(plan: dict, phase: str) -> list[tuple[str, list[int]]]:
    if phase not in PHASES:
        raise ValueError("Invalid routing experiment phase")
    if phase == "final":
        return [(name, plan[phase][name]["input_ids"]) for name in ("chat-holdout", "cjk-holdout")]
    return [(f"{phase}-{i:02d}", row) for i, row in enumerate(plan[phase]["input_ids"])]


def _logits(logits, rows: int):
    import torch

    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or logits.shape != (1, rows, VOCABULARY)
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("Invalid actual full-vocabulary logits")
    return logits


def decode_teacher_sequence(
    model, tokens, path=None, capture=None, offset=0, progress=None, hash_logits=False
):
    import numpy as np
    import torch

    if not tokens or (path is not None and path.exists()):
        raise ValueError("Empty teacher sequence or existing output")
    values = None
    if path is not None:
        values = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(tokens), VOCABULARY)
        )
    cache, output, logits = None, None, None
    dtypes, digest = set(), hashlib.sha256()
    try:
        for position, token in enumerate(tokens):
            if capture is not None:
                capture.begin(offset + position)
            output = model(
                input_ids=torch.tensor([[token]], device=model.device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=0,
            )
            if capture is not None:
                capture.finish()
            logits = _logits(output.logits, 1)
            dtypes.add(str(logits.dtype))
            if values is not None or hash_logits:
                row = logits.float().cpu().numpy().reshape(VOCABULARY)
                if values is not None:
                    values[position] = row
                if hash_logits:
                    digest.update(row.tobytes())
            cache = output.past_key_values
            if cache is None or cache.get_seq_length() != position + 1:
                raise ValueError("Missing or misaligned actual BF16 decode cache")
            if progress is not None and ((position + 1) % 128 == 0 or position + 1 == len(tokens)):
                if values is not None:
                    values.flush()
                progress(position + 1)
            logits, output = None, None
        record = {"shape": [len(tokens), VOCABULARY], "compute_dtypes": sorted(dtypes)}
        if hash_logits:
            record["logits_payload_sha256"] = digest.hexdigest()
        if values is not None:
            values.flush()
            record.update(storage_dtype="float32", **artifact(path))
        return record
    finally:
        if values is not None:
            values.flush()
        del values, cache, output, logits


def _teacher_control(model, modules: dict, tokens: list[int]) -> dict:
    if len(tokens) != 128:
        raise ValueError("Passive teacher hook control needs128 real training tokens")
    plain = decode_teacher_sequence(model, tokens, hash_logits=True)
    with TeacherCapture({layer: module.gate for layer, module in modules.items()}, 128) as capture:
        hooked = decode_teacher_sequence(model, tokens, capture=capture, hash_logits=True)
        for layer in LAYERS:
            capture.payload(layer)
    return {
        "tokens": 128,
        "values": 128 * VOCABULARY,
        "without_hook": plain,
        "with_hook": hooked,
        "bitexact": plain["logits_payload_sha256"] == hooked["logits_payload_sha256"],
    }


def decode_quantized_sequence(
    model, cache, tokens, path=None, capture=None, offset=0, progress=None, hash_logits=False
):
    import numpy as np
    import torch

    if not tokens or (path is not None and path.exists()):
        raise ValueError("Empty quantized sequence or existing output")
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, len(tokens)),
    }
    values, logits, digest, dtypes = None, None, hashlib.sha256(), set()
    if path is not None:
        values = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(len(tokens), VOCABULARY)
        )
    try:
        for position, token in enumerate(tokens):
            if capture is not None:
                capture.begin(offset + position)
            logits = model.forward(torch.tensor([[token]]), params)
            if capture is not None:
                capture.finish()
            logits = _logits(logits, 1)
            dtypes.add(str(logits.dtype))
            if values is not None or hash_logits:
                row = logits.half().cpu().numpy().reshape(VOCABULARY)
                if not np.isfinite(row).all():
                    raise ValueError("Nonfinite F16 stored logits")
                if values is not None:
                    values[position] = row
                if hash_logits:
                    digest.update(row.tobytes())
            params["past_len"] += 1
            if progress is not None and ((position + 1) % 128 == 0 or position + 1 == len(tokens)):
                if values is not None:
                    values.flush()
                progress(position + 1)
            logits = None
        record = {"shape": [len(tokens), VOCABULARY], "compute_dtypes": sorted(dtypes)}
        if hash_logits:
            record["logits_payload_sha256"] = digest.hexdigest()
        if values is not None:
            values.flush()
            record.update(storage_dtype="float16", **artifact(path))
        return record
    finally:
        if values is not None:
            values.flush()
        params.clear()
        del values, logits, cache


def _save_tensors(path: Path, tensors: dict) -> dict:
    import torch
    from safetensors.torch import save_file

    if (
        path.exists()
        or not tensors
        or any(not torch.isfinite(value).all().item() for value in tensors.values())
    ):
        raise ValueError("Existing tensor output or nonfinite/empty captured tensors")
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        str(path),
        metadata={"format": "pt"},
    )
    return {
        **artifact(path),
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
            for key, value in tensors.items()
        },
    }


def _parameters(modules: dict, teacher: bool) -> dict:
    import torch

    result = {}
    for layer, module in modules.items():
        prefix = f"model.layers.{layer}.feed_forward"
        gate = module.gate.weight if teacher else module.routing_cfg.gate_tensor.T
        bias = module.expert_bias if teacher else module.e_score_correction_bias
        expected = torch.bfloat16 if teacher else torch.float16
        if gate.shape != (EXPERTS, HIDDEN) or gate.dtype != expected:
            raise ValueError("Invalid actual routing gate parameters")
        if bias.shape != (EXPERTS,) or bias.dtype != torch.float32:
            raise ValueError("Invalid actual FP32 routing selection bias")
        result[prefix + ".gate.weight"] = gate.detach().cpu().contiguous().clone()
        result[prefix + ".expert_bias"] = bias.detach().cpu().contiguous().clone()
    return result


def _start(folder: Path, kind: str, plan: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "publication": False,
        "kind": kind,
        "files": {},
        "sequences": {},
        "script": artifact(Path(__file__)),
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }
    atomic_report(folder, report)
    return report


def _progress(folder: Path, report: dict, phase: str, name: str, total: int):
    def progress(tokens):
        report["progress"] = {"phase": phase, "sequence": name, "tokens": tokens, "total": total}
        atomic_report(folder, report)
        print(json.dumps(report["progress"]), flush=True)

    return progress


def run_teacher(source: Path, plan: dict, folder: Path) -> dict:
    report, started = _start(folder, "teacher", plan), time.monotonic()
    model, modules, capture = None, {}, None
    try:
        validate_plan(plan)
        import torch

        model = load_source(source, report)
        modules = {
            layer: model.get_submodule(f"model.layers.{layer}.feed_forward") for layer in LAYERS
        }
        report["files"]["routing-parameters.safetensors"] = _save_tensors(
            folder / "routing-parameters.safetensors", _parameters(modules, True)
        )
        with model._optimize_model_for_decode(), torch.inference_mode():
            implementation = model.get_experts_implementation()
            if not implementation or set(implementation.values()) != {"batched_mm"}:
                raise ValueError("Official BF16 decode must use actual batched_mm experts")
            report["experts_implementation"] = implementation
            report["passive_hook_control"] = _teacher_control(
                model, modules, plan["train"]["input_ids"][0][:128]
            )
            if not report["passive_hook_control"]["bitexact"]:
                raise ValueError("Passive teacher router capture changes normal logits")
            rows = sum(map(len, plan["train"]["input_ids"]))
            with TeacherCapture(
                {layer: module.gate for layer, module in modules.items()}, rows
            ) as capture:
                offset = 0
                for name, tokens in phase_sequences(plan, "train"):
                    report["sequences"][name] = decode_teacher_sequence(
                        model,
                        tokens,
                        capture=capture,
                        offset=offset,
                        progress=_progress(folder, report, "train", name, len(tokens)),
                    )
                    offset += len(tokens)
                for layer in LAYERS:
                    name = f"layer-{layer}.safetensors"
                    report["files"][name] = _save_tensors(folder / name, capture.payload(layer))
                    atomic_report(folder, report)
            capture = None
            for phase in ("validation", "final"):
                for name, tokens in phase_sequences(plan, phase):
                    path = folder / f"{name}.npy"
                    record = decode_teacher_sequence(
                        model,
                        tokens,
                        path,
                        progress=_progress(folder, report, phase, name, len(tokens)),
                    )
                    report["sequences"][name] = record
                    report["files"][path.name] = record
                    atomic_report(folder, report)
        report["status"] = "complete"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        capture = None
        modules.clear()
        model = None
        try:
            gc.collect()
            if "torch" in locals():
                torch.cuda.empty_cache()
        except BaseException as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_report(folder, report)


def _load_quantized(folder: Path, sequences: list, phase: str, report: dict):
    import importlib.metadata

    import torch
    from exllamav3 import Cache, Config, Model
    from quantize_lfm25_modal import check_converter_sources

    report["converter_sources"] = check_converter_sources(
        Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
    )
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("exllamav3", "torch", "safetensors", "numpy")
    }
    report["gpu"] = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report["model_files"] = {
        path.name: artifact(path)
        for path in sorted(folder.iterdir())
        if path.suffix in (".safetensors", ".json")
    }
    if {
        name: report["model_files"].get(name, {}).get("sha256") for name in V2_CONFIGS
    } != V2_CONFIGS:
        raise ValueError("Frozen V2 model configuration SHA mismatch")
    if (
        phase == "train"
        and {name: report["model_files"].get(name, {}).get("sha256") for name in V2_SHARDS}
        != V2_SHARDS
    ):
        raise ValueError("Frozen V2 training checkpoint SHA mismatch")
    config = Config.from_directory(str(folder))
    model = Model.from_config(config)
    try:
        caches = {
            name: Cache(model, max_num_tokens=cache_capacity(len(tokens)), max_batch_size=1)
            for name, tokens in sequences
        }
        controls = (
            [Cache(model, max_num_tokens=cache_capacity(128), max_batch_size=1) for _ in range(2)]
            if phase == "train"
            else []
        )
        model.load(device="cuda:0", max_output_size=1)
        modules = {
            int(module.mlp.key.split(".")[2]): module.mlp
            for module in model.modules
            if getattr(module, "mlp", None) is not None
            and module.mlp.key in {f"model.layers.{layer}.feed_forward" for layer in LAYERS}
        }
        if (
            tuple(modules) != LAYERS
            or config.infer_params.no_reconstruct
            or any(not module.is_quantized or module.bc is None for module in modules.values())
        ):
            raise ValueError("Expected actual fused quantized paths in all38 MoE layers")
        report["execution"] = {
            "normal_forward": True,
            "forced_routes": False,
            "moe_layers": len(modules),
            "fused_bsz1": True,
            "max_output_size": 1,
            "cache_capacities": {name: cache_capacity(len(tokens)) for name, tokens in sequences},
            "control_cache_capacities": [cache_capacity(128)] * len(controls),
        }
        return model, caches, controls, modules
    except BaseException:
        model.unload()
        raise


def run_quantized(model_path: Path, plan: dict, folder: Path, phase: str) -> dict:
    report, started = _start(folder, "quantized", plan), time.monotonic()
    model, caches, controls, modules, capture = None, {}, [], {}, None
    try:
        validate_plan(plan)
        sequences = phase_sequences(plan, phase)
        import torch

        model, caches, controls, modules = _load_quantized(model_path, sequences, phase, report)
        report["phase"] = phase
        with torch.inference_mode():
            if phase == "train":
                report["files"]["routing-parameters.safetensors"] = _save_tensors(
                    folder / "routing-parameters.safetensors", _parameters(modules, False)
                )
                tokens = sequences[0][1][:128]
                if len(tokens) != 128:
                    raise ValueError("Passive-hook control needs128 real training tokens")
                plain = decode_quantized_sequence(model, controls[0], tokens, hash_logits=True)
                with QuantCapture(modules, 128, capture_inputs=False) as control:
                    hooked = decode_quantized_sequence(
                        model, controls[1], tokens, capture=control, hash_logits=True
                    )
                    for layer in LAYERS:
                        control.payload(layer)
                report["passive_hook_control"] = {
                    "tokens": 128,
                    "values": 128 * VOCABULARY,
                    "without_hook": plain,
                    "with_hook": hooked,
                    "bitexact": plain["logits_payload_sha256"] == hooked["logits_payload_sha256"],
                }
                if not report["passive_hook_control"]["bitexact"]:
                    raise ValueError("Passive router capture changes normal logits")
                control = None
                rows = sum(len(tokens) for _, tokens in sequences)
                with QuantCapture(modules, rows) as capture:
                    offset = 0
                    for name, tokens in sequences:
                        report["sequences"][name] = decode_quantized_sequence(
                            model,
                            caches[name],
                            tokens,
                            capture=capture,
                            offset=offset,
                            progress=_progress(folder, report, phase, name, len(tokens)),
                        )
                        offset += len(tokens)
                    for layer in LAYERS:
                        name = f"layer-{layer}.safetensors"
                        report["files"][name] = _save_tensors(folder / name, capture.payload(layer))
                        atomic_report(folder, report)
                capture = None
            else:
                for name, tokens in sequences:
                    path = folder / f"{name}.npy"
                    record = decode_quantized_sequence(
                        model,
                        caches[name],
                        tokens,
                        path,
                        progress=_progress(folder, report, phase, name, len(tokens)),
                    )
                    report["sequences"][name] = record
                    report["files"][path.name] = record
                    atomic_report(folder, report)
        report["status"] = "complete"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        capture = None
        modules.clear()
        caches.clear()
        controls.clear()
        try:
            if model is not None:
                model.unload()
            model = None
            gc.collect()
            if "torch" in locals():
                torch.cuda.empty_cache()
        except BaseException as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_report(folder, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    teacher = subparsers.add_parser("teacher")
    teacher.add_argument("plan", type=Path)
    teacher.add_argument("output", type=Path)
    teacher.add_argument("--source", required=True, type=Path)
    quantized = subparsers.add_parser("quantized")
    quantized.add_argument("plan", type=Path)
    quantized.add_argument("output", type=Path)
    quantized.add_argument("--model", required=True, type=Path)
    quantized.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.kind == "teacher":
        run_teacher(args.source, plan, args.output)
    else:
        run_quantized(args.model, plan, args.output, args.phase)


if __name__ == "__main__":
    main()
