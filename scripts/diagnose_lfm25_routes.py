"""Intervene on actual EXL3 expert routes; this never changes checkpoint weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from contextlib import AbstractContextManager
from pathlib import Path

LAYERS = tuple(f"model.layers.{i}.feed_forward" for i in range(2, 40))
SOURCE_REVISION = "1769516ac588c4c558089c94f27d96b0dc67b183"
WINDOW_HASHES = {
    "window1": "bdafc8c2a7c8ef0322c186013d1b99f7bff96c45fc91f47e9125e9c98fd62061",
    "window5": "23b146a0a8b43a1b0495f9fc9bd6882e7ed628be7cd7c7aa4751082dade7f7a4",
}
V2_SHARDS = {
    "model-00001-of-00004.safetensors": "cce3f431c4a79319bfa10952baaa6db6179971631f5e2a8d60da04bb981c7f8e",
    "model-00002-of-00004.safetensors": "3e87ad7f40ee617dce5d699bb70faaa1d78ea3cac11fd616185ca2a0ab773c77",
    "model-00003-of-00004.safetensors": "b06ea4d40f096d9bedcdc2de00e2c60685feb6da1ca89164824d937bc53576ef",
    "model-00004-of-00004.safetensors": "887707fe6f3c4df422727396f590140ade34b8594b970f1e740e299f8ae0b16e",
}
SOURCE_SHARDS = {
    "model-00001-of-00010.safetensors": "f7b4d9106ff00f1e4529c01bfa2b2e62578cf237cc8c39f792768c510e386b6c",
    "model-00002-of-00010.safetensors": "e33aab029127a29dc3f3ec60e995c3ea37608c517d0e33f0c12bd84ea1031700",
    "model-00003-of-00010.safetensors": "68bd39f975ed9baf0d489772ac302b926f426baf7140d88007c34b64561a6809",
    "model-00004-of-00010.safetensors": "1355d0f20369f5c7697d57eb1d6aa8f954fa12c479e58a688db7817657fc89e4",
    "model-00005-of-00010.safetensors": "a4fec7fbe3c4f3f08afe357abddba837314fe6f220936c515042ea1ed7dfddcd",
    "model-00006-of-00010.safetensors": "d7f59eb54c2c6dab7a54258d1bc3665d35a304301e49cba4a492a430de183ad1",
    "model-00007-of-00010.safetensors": "13f6524d71c8c3190b9d1d083b266a7dff569566a0ab2b14e732e9df08fdb5f7",
    "model-00008-of-00010.safetensors": "51ebc311324fd321baf5d9c12ef0d6f0dc06b88b2e7b95899c63cf33179c2b35",
    "model-00009-of-00010.safetensors": "491ab1cff839626e02c5f996fe90ef044592ce14df6471249567027d4adb4dad",
    "model-00010-of-00010.safetensors": "1d3f7178c0c7d1b146bffe7c5ac41874b06b95ef04538693204502258fe2c1b4",
}


def checked_position(index: int, total: int) -> int:
    """Validate a token position before indexing trace buffers.

    raises: ValueError
    post: 0 <= __return__ < total
    post: __return__ == index
    """
    if type(index) is not int or type(total) is not int or not 0 <= index < total:
        raise ValueError("Invalid route token position")
    return index


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or set(plan) != {"windows", "shards"}:
        raise ValueError("Expected the frozen diagnostic23 plan")
    windows, shards = plan["windows"], plan["shards"]
    if not isinstance(windows, dict) or set(windows) != {"window1", "window5"}:
        raise ValueError("Expected exactly diagnostic windows 1 and 5")
    for name, tokens in windows.items():
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1024
            or any(type(t) is not int or not 0 <= t < 125017 for t in tokens)
        ):
            raise ValueError("Expected 1024 valid imposed token IDs per window")
        if hashlib.sha256(struct.pack("<1024I", *tokens)).hexdigest() != WINDOW_HASHES[name]:
            raise ValueError("Historical diagnostic tokens changed")
    expected = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
    if not isinstance(shards, dict) or set(shards) != expected:
        raise ValueError("Expected the four frozen V2 shards")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        for digest in shards.values()
    ):
        raise ValueError("Invalid frozen V2 shard SHA")
    if shards != V2_SHARDS:
        raise ValueError("Historical V2 shard manifest changed")
    return plan


def artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def validate_routes(routes: dict, keys: tuple[str, ...], steps: int) -> None:
    import torch

    if not keys or len(set(keys)) != len(keys) or set(routes) != set(keys) or steps < 1:
        raise ValueError("Incomplete expert route panel")
    for ids, weights in routes.values():
        if (
            ids.shape != (steps, 4)
            or weights.shape != (steps, 4)
            or ids.dtype != torch.int64
            or weights.dtype != torch.float16
            or not torch.isfinite(weights).all().item()
            or not ((ids >= 0) & (ids < 64)).all().item()
            or not (weights > 0).all().item()
            or not (ids.sort(dim=1).values.diff(dim=1) > 0).all().item()
            or not ((weights.float().sum(dim=1) - 1).abs() <= 0.002).all().item()
        ):
            raise ValueError("Invalid expert IDs, affinities, shape or dtype")


def cuda_selected_weights(cfg, hidden, ids, weights) -> None:
    """Use the pinned CUDA normalization, including its actual FP16 rounding."""
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.block_sparse_mlp_routing import ROUTING_ACT_SIGMOID, _gate_t

    _gate_t(cfg)
    ext.routing_sel_norm(
        hidden,
        cfg.gate_tensor,
        cfg.router_logits_bsz1,
        ids,
        weights,
        cfg.routed_scaling_factor,
        cfg.gate_tensor_t,
        ROUTING_ACT_SIGMOID,
        cfg.gate_i8,
        cfg.gate_sb,
    )


class RouteCapture(AbstractContextManager):
    """Copy/replay the buffers actually passed to each fused MoE call."""

    def __init__(self, modules: dict, steps: int, replay: dict | None = None, ids_only=False):
        if not modules or steps < 1 or (ids_only and replay is None):
            raise ValueError("Nonempty route modules and replay source are required")
        self.modules, self.keys, self.steps = modules, tuple(modules), steps
        if replay is not None:
            validate_routes(replay, self.keys, steps)
        self.replay, self.ids_only = replay, ids_only
        self.originals, self.buffers, self.forced = {}, {}, {}
        self.position, self.completed, self.seen = None, 0, set()

    def __enter__(self):
        import torch

        try:
            for key, module in self.modules.items():
                self.buffers[key] = (
                    torch.empty((self.steps, 4), dtype=torch.long, device=module.device),
                    torch.empty((self.steps, 4), dtype=torch.float16, device=module.device),
                )
                if self.replay is not None:
                    self.forced[key] = tuple(
                        t.to(module.device).contiguous() for t in self.replay[key]
                    )
                original = module.routing_fn
                self.originals[key] = original

                def capture(bsz, cfg, hidden, params, *, key=key, original=original):
                    if bsz != 1 or self.position is None or key in self.seen:
                        raise ValueError("Misaligned or repeated actual routing call")
                    ids, weights = original(bsz, cfg, hidden, params)
                    if ids.shape != (1, 4) or weights.shape != (1, 4):
                        raise ValueError("Actual routing is not single-token top-4")
                    if ids.dtype != torch.long or weights.dtype != torch.float16:
                        raise ValueError("Actual routing dtype changed")
                    if self.replay is not None:
                        # Keep the actual allocations: CUDA graphs can retain their addresses.
                        ids.copy_(self.forced[key][0][self.position : self.position + 1])
                        if self.ids_only:
                            cuda_selected_weights(cfg, hidden, ids, weights)
                        else:
                            weights.copy_(self.forced[key][1][self.position : self.position + 1])
                    self.buffers[key][0][self.position].copy_(ids[0])
                    self.buffers[key][1][self.position].copy_(weights[0])
                    self.seen.add(key)
                    return ids, weights

                module.routing_fn = capture
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def begin(self, index: int) -> None:
        checked_position(index, self.steps)
        if self.position is not None or index != self.completed:
            raise ValueError("Routing token order changed")
        self.position, self.seen = index, set()

    def finish(self) -> None:
        if self.position is None or self.seen != set(self.keys):
            raise ValueError("Missing expert layer in actual forward")
        self.position = None
        self.completed += 1

    def payload(self) -> dict:
        if self.position is not None or self.completed != self.steps:
            raise ValueError("Refusing incomplete route capture")
        result = {key: tuple(t.cpu().clone() for t in pair) for key, pair in self.buffers.items()}
        validate_routes(result, self.keys, self.steps)
        return result

    def __exit__(self, *_):
        for key, original in self.originals.items():
            self.modules[key].routing_fn = original
        return False


def route_comparison(source: dict, candidate: dict, steps: int) -> dict:
    import torch

    keys = tuple(source)
    validate_routes(source, keys, steps)
    validate_routes(candidate, keys, steps)
    return {
        key: {
            "tokens": steps,
            "changed_top4_sets": int(
                (source[key][0].sort(dim=1).values != candidate[key][0].sort(dim=1).values)
                .any(dim=1)
                .sum()
                .item()
            ),
            "same_order": int((source[key][0] == candidate[key][0]).all(dim=1).sum().item()),
            "ordered_weight_max_difference": float(
                (source[key][1].float() - candidate[key][1].float()).abs().max().item()
            ),
            "bitexact": bool(
                torch.equal(source[key][0], candidate[key][0])
                and torch.equal(
                    source[key][1].view(torch.uint8), candidate[key][1].view(torch.uint8)
                )
            ),
        }
        for key in keys
    }


def compare_logits(reference: Path, candidate: Path) -> dict:
    import numpy as np

    a, b = (np.load(p, mmap_mode="r", allow_pickle=False) for p in (reference, candidate))
    if (
        a.shape != b.shape
        or a.ndim != 2
        or not a.size
        or a.dtype != b.dtype
        or not np.issubdtype(a.dtype, np.floating)
    ):
        raise ValueError("Mismatched saved logits")
    total, matches = 0.0, 0
    for start in range(0, len(a), 32):
        x, y = a[start : start + 32].astype(np.float64), b[start : start + 32].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite saved logits")
        matches += int((x.argmax(axis=1) == y.argmax(axis=1)).sum())
        x -= x.max(axis=1, keepdims=True)
        y -= y.max(axis=1, keepdims=True)
        p = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
        q = y - np.log(np.exp(y).sum(axis=1, keepdims=True))
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Logit normalization overflow")
        total += float((np.exp(p) * (p - q)).sum())
        if not math.isfinite(total):
            raise ValueError("Nonfinite logit divergence")
    return {"tokens": len(a), "kl": max(0.0, total / len(a)), "greedy_matches": matches}


def decode_window(
    model, cache, tokens: list[int], modules: dict, path: Path, replay=None, ids_only=False
):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, len(tokens)),
    }
    if path.exists() or path.with_suffix(".routes.safetensors").exists():
        raise FileExistsError("Preserve previous logits and routes")
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(len(tokens), 128000)
    )
    dtypes = set()
    try:
        with (
            torch.inference_mode(),
            RouteCapture(modules, len(tokens), replay, ids_only) as capture,
        ):
            for index, token in enumerate(tokens):
                capture.begin(index)
                logits = model.forward(torch.tensor([[token]]), params)
                capture.finish()
                if not logits.is_floating_point() or logits.numel() != 128000:
                    raise ValueError("Expected floating full-vocabulary logits")
                dtypes.add(str(logits.dtype))
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Nonfinite actual logits")
                row = logits.half().cpu().numpy().reshape(-1)
                if not np.isfinite(row).all():
                    raise ValueError("Nonfinite actual logits")
                values[index] = row
                params["past_len"] += 1
                if (index + 1) % 128 == 0:
                    print(json.dumps({"case": path.stem, "tokens": index + 1}), flush=True)
            routes = capture.payload()
        values.flush()
    finally:
        del values
    route_path = path.with_suffix(".routes.safetensors")
    save_file(
        {
            f"{key}.{kind}": t.contiguous()
            for key, pair in routes.items()
            for kind, t in zip(("ids", "weights"), pair)
        },
        str(route_path),
    )
    return routes, {
        **artifact(path),
        "shape": [len(tokens), 128000],
        "dtype": "float16",
        "compute_dtypes": sorted(dtypes),
        "token_sha256": hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest(),
        "routes": {
            **artifact(route_path),
            "layers": len(routes),
            "tokens_per_layer": len(tokens),
            "top_k": 4,
        },
    }


def run(plan: dict, output: Path, data: Path = Path("/data")) -> dict:
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "publication": False,
        "causal_interpretation": False,
        "arms": {},
        "ids_only": "not_executed",
    }
    model = None

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        validate_plan(plan)
        import importlib.metadata

        import torch
        from exllamav3 import Cache, Config, Model
        from exllamav3.ext import exllamav3_ext as ext
        from quantize_lfm25_modal import check_converter_sources

        report["script"] = artifact(Path(__file__))
        report["converter_sources"] = check_converter_sources(
            Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
        )
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("exllamav3", "torch", "numpy", "safetensors")
        }
        report["gpu"] = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        quantized = data / "v2/output"
        report["v2_shards"] = {}
        for name, expected in plan["shards"].items():
            record = artifact(quantized / name)
            if record["sha256"] != expected:
                raise ValueError("Frozen V2 weight SHA mismatch: " + name)
            report["v2_shards"][name] = record
        source = data / "hf/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots" / SOURCE_REVISION
        source_index = json.loads((source / "model.safetensors.index.json").read_text())
        source_names = sorted(set(source_index["weight_map"].values()))
        if set(source_names) != set(SOURCE_SHARDS):
            raise ValueError("Invalid source shard index")
        report["source_revision"] = SOURCE_REVISION
        report["source_shards"] = {name: artifact(source / name) for name in source_names}
        if any(
            record["sha256"] != SOURCE_SHARDS[name]
            for name, record in report["source_shards"].items()
        ):
            raise ValueError("Pinned source weight SHA mismatch")
        report["source_config"] = artifact(source / "config.json")
        save()
        source_routes, normal_routes = {}, {}
        for kind, folder in (("source", source), ("v2", quantized)):
            config = Config.from_directory(str(folder))
            model = Model.from_config(config)
            # Every arm/window owns a cache, created before load allocates recurrent/KV tensors.
            arm_names = (
                ["source"] if kind == "source" else ["normal", "own_routes", "source_routes"]
            )
            ids_api = callable(getattr(ext, "routing_sel_norm", None))
            if kind == "v2" and ids_api:
                arm_names = ["normal", "own_routes", "own_ids", "source_routes", "source_ids"]
            caches = {
                (arm, name): Cache(model, max_num_tokens=1024, max_batch_size=1)
                for arm in arm_names
                for name in plan["windows"]
            }
            model.load(device="cuda:0", max_output_size=1)
            modules = {
                m.mlp.key: m.mlp
                for m in model.modules
                if getattr(m, "mlp", None) is not None and m.mlp.key in LAYERS
            }
            if tuple(modules) != LAYERS:
                raise ValueError("Missing or reordered actual 38 MoE layers")
            if kind == "v2" and (
                config.infer_params.no_reconstruct
                or any(not m.is_quantized or m.bc is None for m in modules.values())
            ):
                raise ValueError("V2 is not using the actual fused quantized MoE path")
            report.setdefault("model_paths", {})[kind] = {
                "layers": len(modules),
                "quantized": all(m.is_quantized for m in modules.values()),
                "fused_bsz1": all(m.bc is not None for m in modules.values()),
                "cache_count": len(caches),
            }
            ids_approved = ids_api
            for arm in arm_names:
                if arm == "source_ids" and not ids_approved:
                    continue
                report["arms"][arm] = {}
                for name, tokens in plan["windows"].items():
                    replay = (
                        normal_routes.get(name)
                        if arm in ("own_routes", "own_ids")
                        else source_routes.get(name)
                        if arm in ("source_routes", "source_ids")
                        else None
                    )
                    trace, record = decode_window(
                        model,
                        caches[arm, name],
                        tokens,
                        modules,
                        output / f"{arm}-{name}.npy",
                        replay,
                        arm.endswith("_ids"),
                    )
                    if arm == "source":
                        source_routes[name] = trace
                    elif arm == "normal":
                        normal_routes[name] = trace
                        record["source_route_comparison"] = route_comparison(
                            source_routes[name], trace, len(tokens)
                        )
                    elif arm == "own_routes":
                        normal = report["arms"]["normal"][name]
                        record["replay_bitexact"] = normal["sha256"] == record["sha256"] and all(
                            r["bitexact"]
                            for r in route_comparison(
                                normal_routes[name], trace, len(tokens)
                            ).values()
                        )
                        if not record["replay_bitexact"]:
                            report["arms"][arm][name] = record
                            raise ValueError("Actual V2 own-route replay is not bitexact")
                    elif arm == "own_ids":
                        control = compare_logits(
                            output / f"normal-{name}.npy", Path(record["path"])
                        )
                        record["normal_comparison"] = control
                        ids_approved &= (
                            control["kl"] <= 1e-6
                            and control["greedy_matches"] / control["tokens"] >= 0.999
                        )
                    elif arm == "source_routes" and not all(
                        r["bitexact"]
                        for r in route_comparison(source_routes[name], trace, len(tokens)).values()
                    ):
                        raise ValueError("Source routes/affinities were not actually imposed")
                    elif arm == "source_ids" and any(
                        not torch.equal(source_routes[name][key][0], trace[key][0])
                        for key in modules
                    ):
                        raise ValueError("Source expert IDs were not actually imposed")
                    report["arms"][arm][name] = record
                    save()
                if arm == "own_ids":
                    report["ids_only"] = (
                        "approved" if ids_approved else "not_executed_control_failed"
                    )
                    save()
            model.unload()
            model = None
            del caches, modules, config
            torch.cuda.empty_cache()
        report.update(
            status="complete",
            historical_agreement="pending independent CPU KL/NLL comparison to diagnostic23",
            ids_only=report["ids_only"] if ids_api else "not_executed_binding_absent",
        )
        save()
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error), causal_interpretation=False)
        save()
        raise
    finally:
        if model is not None:
            model.unload()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    args = parser.parse_args(argv)
    return run(json.loads(args.plan.read_text()), args.output, args.data)


if __name__ == "__main__":
    main()
