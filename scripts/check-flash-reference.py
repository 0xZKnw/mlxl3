"""Independent Flash Next oracle: mlx-vlm architecture and upstream PonyExl3.

Only selected experts and PLE rows are read. Never instantiate a second target
checkpoint in the same process. Outputs are correctness fixtures, not benchmarks.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np


class Store:
    def __init__(self, root):
        self.entries = {}
        self.files = []
        try:
            self.open(root)
        except BaseException:
            self.close()
            raise

    def open(self, root):
        for path in sorted(root.glob("*.safetensors")):
            if not (
                path.name.startswith("model")
                or path.name
                in ("ngram_embedding.safetensors", "mtp_hyper_connection_mixer_patch.safetensors")
            ):
                continue
            fd = os.open(path, os.O_RDONLY)
            self.files.append(fd)
            size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            if size > 100_000_000:
                raise ValueError("invalid checkpoint header")
            header = json.loads(os.pread(fd, size, 8))
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if name in self.entries:
                    raise ValueError(f"duplicate tensor {name}")
                self.entries[name] = (fd, size + 8, info)

    def read(self, name, row=None):
        fd, base, info = self.entries[name]
        dtype = {
            "F16": "<f2",
            "BF16": "<u2",
            "F32": "<f4",
            "I16": "<u2",
            "I32": "<i4",
            "I64": "<i8",
        }[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        if row is not None:
            if not 0 <= row < shape[0]:
                raise ValueError("row outside tensor")
            width = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
            start, end = start + row * width, start + (row + 1) * width
            shape = shape[1:]
        raw = os.pread(fd, end - start, base + start)
        if len(raw) != end - start:
            raise ValueError(f"truncated tensor {name}")
        value = np.frombuffer(raw, dtype).reshape(shape).copy()
        if info["dtype"] == "BF16":
            value = (value.astype(np.uint32) << 16).view(np.float32)
        return value

    def close(self):
        for fd in self.files:
            os.close(fd)
        self.files.clear()


def mul1(words):
    """CUDA half FMA codebook, evaluated independently in float64 then half."""
    bits = words.astype(np.uint32) * np.uint32(0x83DCD12D)
    sums = np.full(words.shape, 0x6400, np.uint32)
    for shift in (0, 8, 16, 24):
        sums += (bits >> shift) & 255
    values = sums.astype(np.uint16).view(np.float16).astype(np.float64)
    inv, bias = np.array([0x1EEE, 0xC931], np.uint16).view(np.float16).astype(np.float64)
    return (values * inv + bias).astype(np.float16)


def decode_ngram(words):
    if words.shape != (41,) or words.dtype != np.uint16:
        raise ValueError("invalid ngram record")
    symbols = ((words[1:, None] >> np.arange(0, 16, 4)) & 15).reshape(160)
    codes = symbols.astype(np.uint16)
    for previous in range(1, 4):
        codes |= np.roll(symbols, previous).astype(np.uint16) << (4 * previous)
    scale = words[:1].view(np.float16).astype(np.float32)
    values = mul1(codes).astype(np.float32) * scale
    if not np.isfinite(values).all():
        raise ValueError("nonfinite ngram row")
    return values


def parse_tokens(value: str) -> list[int]:
    """Bound disposable oracle workloads and reject invalid vocabulary IDs.

    raises: ValueError
    post: 1 <= len(__return__) <= 32
    post: all(0 <= token < 248320 for token in __return__)
    """
    tokens = [int(token) for token in value.split(",")]
    if not tokens or len(tokens) > 32 or any(not 0 <= token < 248320 for token in tokens):
        raise ValueError("oracle requires 1..32 valid tokens")
    return tokens


def probe_input(root: Path, entry: dict, width: int) -> np.ndarray:
    if entry.get("count") != width or entry.get("shape") != [1, 1, width]:
        raise ValueError("invalid probe input shape/count")
    with (root / entry["file"]).open("rb") as handle:
        payload = handle.read(width * 4 + 1)
    if len(payload) != width * 4:
        raise ValueError("invalid probe input bytes")
    data = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(data).all():
        raise ValueError("nonfinite probe input")
    return data.reshape(1, 1, width)


def draft_head_input(root: Path):
    """Read the bounded real backbone fixture before loading the independent head."""
    data = json.loads((root / "oracle.json").read_text())
    kind = data.get("kind")
    count = data.get("proposals")
    if (
        data.get("status") != "complete_cpu_oracle"
        or kind not in ("dflash2", "dspark")
        or type(count) is not int
        or not 1 <= count <= 7
        or data.get("residual_norm") != "sglang-3c560089-fp32-variance-bf16-residual"
    ):
        raise ValueError("incomplete or foreign draft oracle")
    rows = count + int(kind == "dflash2")
    entry = data["arrays"]["hidden"]
    filename = entry["file"]
    if (
        type(filename) is not str
        or Path(filename).name != filename
        or entry.get("shape") != [rows, 2560]
        or entry.get("count") != rows * 2560
    ):
        raise ValueError("invalid draft head input metadata")
    with (root / filename).open("rb") as stream:
        payload = stream.read(rows * 2560 * 4 + 1)
    if len(payload) != rows * 2560 * 4 or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError("incomplete or changed draft hidden")
    hidden = np.frombuffer(payload, dtype="<f4").reshape(rows, 2560)
    if not np.isfinite(hidden).all():
        raise ValueError("nonfinite draft hidden")
    return data, hidden[int(kind == "dflash2") :]


def norm_reference(model: Path, output: Path):
    """Small oracle using the actual upstream norms, including grouped FP32 HC."""
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm, _qwen4_inject

    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "in_progress", "cases": []}
    store = Store(model)
    try:
        names = [
            ("layers.0.attn_hyper_connection.hc_norm", 2560, 4, True),
            ("layers.0.linear_attn.norm", 128, 1, False),
            ("layers.3.self_attn.q_norm", 256, 1, True),
            ("layers.3.self_attn.k_norm", 256, 1, True),
            ("layers.3.self_attn.indexer.q_layernorm", 128, 1, True),
            ("layers.3.self_attn.indexer.k_layernorm", 128, 1, True),
            ("layers.1.ple.norm_key", 2560, 4, True),
            ("layers.1.ple.norm_query", 2560, 4, True),
            ("layers.1.ple.norm_conv", 2560, 4, True),
            ("hyper_connection_mixer.hc_norm", 2560, 4, True),
        ]
        for name, width, groups, centered in names:
            name = "model.language_model." + name + ".weight"
            weight = mx.array(store.read(name))
            norm = Qwen4ExpRMSNorm(
                width * groups, group_size=width if groups > 1 else None, eps=1e-6
            )
            norm.weight = weight
            for dtype in (mx.float16, mx.float32):
                for rows in (1, 3):
                    values = (
                        (np.arange(rows * width * groups, dtype=np.int64) * 3571) % 20011 - 10005
                    ) / 37.0
                    x = mx.array(values.astype(np.float32).reshape(rows, width * groups)).astype(
                        dtype
                    )
                    y = norm(x) if centered else mx.fast.rms_norm(x, weight, 1e-6)
                    case = {
                        "weight": name,
                        "width": width,
                        "groups": groups,
                        "centered": centered,
                        "dtype": str(dtype),
                        "rows": rows,
                    }
                    for kind, value in (("input", x), ("output", y)):
                        data = np.asarray(value.astype(mx.float32))
                        if data.size != rows * width * groups or not np.isfinite(data).all():
                            raise ValueError("invalid norm oracle")
                        path = f"norm-{len(report['cases'])}-{kind}.f32"
                        data.astype("<f4").tofile(output / path)
                        case[kind] = {
                            "file": path,
                            "shape": list(data.shape),
                            "count": int(data.size),
                        }
                    report["cases"].append(case)
        report["hc_write"] = []
        for branch_dtype in (mx.float16, mx.float32):
            for scale in (0.001, 1.0, 100.0):
                values = ((np.arange(10240) * 3571) % 20011 - 10005).astype(np.float32) / 37.0
                streams = mx.array(values.reshape(1, 10240)) * scale
                branch = mx.array(values[:2560].reshape(1, 2560)).astype(branch_dtype)
                gate = mx.array([[0.073891, 0.528147, 1.473211, 1.873529]])
                case = {"branch_dtype": str(branch_dtype)}
                for kind, value in (
                    ("streams", streams),
                    ("branch", branch),
                    ("gate", gate),
                    ("output", _qwen4_inject(branch, streams, gate)),
                ):
                    data = np.asarray(value.astype(mx.float32))
                    if data.size == 0 or not np.isfinite(data).all():
                        raise ValueError("invalid HC oracle")
                    path = f"hc-{len(report['hc_write'])}-{kind}.f32"
                    data.astype("<f4").tofile(output / path)
                    case[kind] = {"file": path, "shape": list(data.shape), "count": int(data.size)}
                report["hc_write"].append(case)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        store.close()
        (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pony", type=Path, default=Path("build/research-ponyexl3"))
    parser.add_argument("--tokens", default="248044,1,42,248044,7")
    parser.add_argument("--norm-only", action="store_true")
    parser.add_argument("--projections-only", action="store_true")
    parser.add_argument("--attention-only", type=Path)
    parser.add_argument("--mtp-only", type=Path)
    parser.add_argument("--draft-head-only", type=Path)
    args = parser.parse_args()
    if (
        sum(
            (
                args.norm_only,
                args.projections_only,
                args.attention_only is not None,
                args.mtp_only is not None,
                args.draft_head_only is not None,
            )
        )
        > 1
    ):
        parser.error("choose a single probe mode")
    if args.norm_only:
        norm_reference(args.model, args.output)
        return
    tokens = parse_tokens(args.tokens)
    sys.path.insert(0, str(args.pony.resolve()))
    import mlx.core as mx
    from mlx import nn
    from mlx_vlm.models.activations import swiglu
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta
    from mlx_vlm.models.qwen4_exp.language import (
        ArraysCache,
        QSAKVCache,
        Qwen4ExpDecoderLayer,
        Qwen4ExpGatedResidual,
        TextConfig,
    )
    from ponyexl3.mlx.exl3_linear import EXL3Linear
    from ponyexl3.mlx.layer_state import clear_layer_caches
    from ponyexl3.ref.layer import EXL3Layer

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "in_progress",
        "tokens": tokens,
        "steps": [],
        "reference": "mlx-vlm 0.7.3 + PonyExl3 8e7fa6b",
        "branch_dtype": "float16",
        "mode": "eval",
        "activation": "upstream compiled float16 swiglu",
    }
    report_path = args.output / "reference.json"
    store = Store(args.model)

    def save_report():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def save(name, value):
        data = np.asarray(value.astype(mx.float32))
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError(f"invalid reference output {name}")
        data.astype("<f4").tofile(args.output / (name + ".f32"))
        return {"file": name + ".f32", "shape": list(data.shape), "count": int(data.size)}

    class Packed(nn.Module):
        def __init__(self, prefix, logical_output=None):
            super().__init__()
            t = store.read(prefix + ".trellis")
            layer = EXL3Layer(
                prefix,
                t.shape[0] * 16,
                t.shape[1] * 16,
                t.shape[2] // 16,
                t,
                store.read(prefix + ".suh"),
                store.read(prefix + ".svh"),
                mul1=prefix + ".mul1" in store.entries,
                mcg=prefix + ".mcg" in store.entries,
            )
            self._linear = EXL3Linear(layer).release_source()
            self._in = layer.in_features
            self._out = logical_output or layer.out_features

        def __call__(self, x):
            if x.shape[-1] < self._in:
                x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self._in - x.shape[-1])])
            return self._linear(x)[..., : self._out]

    class Routed(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self._prefix = prefix
            self._experts = {}

        def __call__(self, x, indices):
            flat = indices.tolist()[0][0]
            self._indices = flat
            outputs = []
            for expert in flat:
                p = f"{self._prefix}.{expert}"
                if expert not in self._experts:
                    self._experts[expert] = [
                        Packed(p + "." + part) for part in ("gate_proj", "up_proj", "down_proj")
                    ]
                gate, up, down = self._experts[expert]
                g, u = gate(x), up(x)
                activation = swiglu(g, u)
                outputs.append(down(activation))
            result = mx.stack(outputs, axis=-2)
            mx.eval(result)
            # Only keep this exact ten-expert cohort; no unbounded GPU cache.
            self._experts = {key: self._experts[key] for key in flat}
            clear_layer_caches()
            return result

    class NgramRows(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self._prefix = prefix
            self._bias = store.read(prefix + ".head_bias").astype(np.float32)

        def __call__(self, ids):
            output = []
            for head, row in enumerate(ids.tolist()[0][0]):
                shard, local = divmod(row, 2500012)
                decoded = decode_ngram(store.read(f"{self._prefix}.shard_{shard}.trellis", local))
                output.append((decoded + self._bias[head]).astype(np.float16))
            return mx.array(np.stack(output)[None, None])

    class HalfGdn(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self._layer = layer

        def __call__(self, inputs, **kwargs):
            return self._layer(inputs.astype(mx.float16), **kwargs)

    class Observe(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self._layer = layer
            self._input = None
            self._output = None

        def __call__(self, inputs, *arguments, **kwargs):
            self._input = inputs
            self._output = self._layer(inputs, *arguments, **kwargs)
            return self._output

    def load(module, prefix):
        for name, child in list(module.children().items()):
            path = prefix + "." + name
            if path + ".trellis" in store.entries:
                module[name] = Packed(path, child.weight.shape[0])
            elif isinstance(child, nn.Linear):
                child.weight = mx.array(store.read(path + ".weight").astype(np.float16))
            elif name == "switch_mlp":
                module[name] = Routed(prefix + ".experts")
            elif name == "ple_embedding":
                child.ngram_embedding = NgramRows(path + ".ngram_embedding")
                child.layer_multipliers = store.read(
                    path + ".ngram_embedding.layer_multipliers"
                ).tolist()
                child.ngram_heads_vocab_sizes = mx.array(
                    store.read(path + ".ngram_embedding.head_vocab_sizes")
                )
                child.ngram_heads_offsets = mx.array(
                    store.read(path + ".ngram_embedding.head_offsets")
                )
            elif isinstance(child, nn.Conv1d):
                child.weight = mx.array(
                    store.read(path + ".weight").transpose(0, 2, 1).astype(np.float16)
                )
            elif isinstance(child, nn.Module):
                load(child, path)
        for name in ("weight", "A_log", "dt_bias"):
            if prefix + "." + name in store.entries:
                value = store.read(prefix + "." + name)
                if name == "dt_bias":
                    value = value.astype(np.float16)
                module[name] = mx.array(value)

    try:
        save_report()
        if args.draft_head_only is not None:
            from check_flash_drafts import Weights, bf16, select_cpu

            draft_data, hidden = draft_head_input(args.draft_head_only)
            lm_head = Packed("lm_head", 248320)
            logits = lm_head(mx.array(hidden).astype(mx.float16)).astype(mx.bfloat16)
            mx.eval(logits)
            logits_cpu = np.asarray(logits.astype(mx.float32))
            selected, confidence = select_cpu(
                Weights(Path(draft_data["draft_directory"]) / "model.safetensors"),
                draft_data["kind"],
                bf16(hidden),
                logits_cpu,
                7,
            )
            report["draft_head"] = {
                "kind": draft_data["kind"],
                "revision": draft_data["revision"],
                "hidden_sha256": draft_data["arrays"]["hidden"]["sha256"],
                "logits": save("draft-head-logits", logits),
                "tokens": selected,
                "confidence": confidence,
                "anchor": 7,
            }
            report["status"] = "complete"
            return
        if args.projections_only:
            report["projections"] = []
            specs = [
                ("layers.0.linear_attn.in_proj_a", 2560, 48),
                ("layers.0.linear_attn.in_proj_b", 2560, 48),
                ("layers.0.linear_attn.in_proj_qkv", 2560, 10240),
                ("layers.0.linear_attn.in_proj_z", 2560, 6144),
                ("layers.0.linear_attn.out_proj", 6144, 2560),
                ("layers.0.mlp.experts.0.gate_proj", 2560, 640),
                ("layers.0.mlp.experts.0.up_proj", 2560, 640),
                ("layers.0.mlp.experts.0.down_proj", 640, 2560),
                ("layers.3.self_attn.q_proj", 2560, 12288),
                ("lm_head", 2560, 248320),
            ]
            for suffix, width, out in specs:
                prefix = suffix if suffix == "lm_head" else "model.language_model." + suffix
                if prefix + ".trellis" in store.entries:
                    module = Packed(prefix, out)
                else:
                    module = nn.Linear(width, out, bias=False)
                    module.weight = mx.array(store.read(prefix + ".weight").astype(np.float16))
                for dtype in (mx.float16, mx.float32):
                    values = ((np.arange(width) * 3571) % 20011 - 10005).astype(np.float32) / 4000.0
                    x = mx.array(values.reshape(1, width)).astype(dtype)
                    y = module(x)
                    index = len(report["projections"])
                    report["projections"].append(
                        {
                            "prefix": prefix,
                            "input_width": width,
                            "output_width": out,
                            "dtype": str(dtype),
                            "input": save(f"p{index}-input", x),
                            "output": save(f"p{index}-output", y),
                        }
                    )
                    print(f"projection {prefix} {dtype}", flush=True)
                del module
                clear_layer_caches()
            report["status"] = "complete"
            return
        config = TextConfig.from_dict(
            json.loads((args.model / "config.json").read_text())["text_config"]
        )
        if args.mtp_only is not None:
            from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
            from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
                Qwen4ExpMTPDraftModel,
            )

            original = json.loads((args.mtp_only / "reference.json").read_text())
            steps = original.get("steps", [])
            if (
                original.get("status") != "complete"
                or original.get("mode") != "eval"
                or not 5 <= len(steps) <= 32
            ):
                raise ValueError("invalid MTP input fixture")
            head = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig(text_config=config))
            load(head, "mtp")
            load(head.layers[0], "mtp.layers.0")
            head.eval()
            if head.training:
                raise ValueError("MTP oracle is not in inference mode")
            for name in ("attn_hyper_connection", "self_attn", "mlp_hyper_connection", "mlp"):
                head.layers[0][name] = Observe(head.layers[0][name])
            lm_head = Packed("lm_head", 248320)

            def embed(ids):
                row = store.read("model.language_model.embed_tokens.weight", int(ids.item()))
                return mx.array(row).astype(mx.float16).reshape(1, 1, 2560)

            head._input_embed = embed
            head._cache = head.make_cache()
            report["embedding_dtype"] = "float16, same target embedding boundary"
            report["mtp_steps"] = []
            for index, (previous, current) in enumerate(pairwise(steps)):
                token = current["token"]
                if not isinstance(token, int) or not 0 <= token < 248320:
                    raise ValueError("invalid MTP token")
                hidden = mx.array(probe_input(args.mtp_only, previous["layers"][-1], 10240))
                ids = mx.array([[token]], dtype=mx.int32)
                mixed = head.fuse_inputs(embed(ids), hidden)
                output, raw = head._forward_tokens(ids, hidden, mx.int32)
                logits = lm_head(output)
                cache = head._cache[0]
                keys, values = cache.state[:2]
                tail = cache.index_keys[:, -(cache.offset % 4) :] if cache.offset % 4 else None
                entry = {"token": token, "position": index}
                for name, value in (
                    ("input", hidden),
                    ("mixed", mixed),
                    ("layer", raw),
                    ("output", output),
                    ("logits", logits),
                    ("keys", keys),
                    ("values", values),
                    ("index_tail", tail),
                ):
                    entry[name] = save(f"mtp-{index}-{name}", value) if value is not None else None
                entry["argmax"] = int(mx.argmax(logits).item())
                for name, component, field in (
                    ("attn_mixed", "attn_hyper_connection", 0),
                    ("attn_gate", "attn_hyper_connection", 2),
                    ("attention", "self_attn", None),
                    ("mlp_mixed", "mlp_hyper_connection", 0),
                    ("mlp_gate", "mlp_hyper_connection", 2),
                    ("moe", "mlp", None),
                ):
                    value = head.layers[0][component]._output
                    entry[name] = save(
                        f"mtp-{index}-{name}", value if field is None else value[field]
                    )
                report["mtp_steps"].append(entry)
                save_report()
                print(f"MTP step {index + 1}/{len(steps) - 1}", flush=True)
            report["status"] = "complete"
            return
        if args.attention_only is not None:
            original = json.loads((args.attention_only / "reference.json").read_text())
            steps = original.get("steps", [])
            if original.get("status") != "complete" or not 5 <= len(steps) <= 32:
                raise ValueError("invalid input attention fixture")
            report["attention"] = {}
            for index in (1, 2, 15, 35):
                parent = Qwen4ExpDecoderLayer(config, index)
                is_ple = index == 1
                is_gdn = parent.is_linear and not is_ple
                name = "ple" if is_ple else "linear_attn" if is_gdn else "self_attn"
                module = parent[name]
                load(module, f"model.language_model.layers.{index}.{name}")
                module.eval()
                if module.training:
                    raise ValueError("attention oracle is not in inference mode")
                names = (
                    (
                        "ple_embedding",
                        "key_proj",
                        "value_proj",
                        "norm_key",
                        "norm_query",
                        "norm_conv",
                        "conv1d",
                    )
                    if is_ple
                    else ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "norm")
                    if is_gdn
                    else ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
                )
                for child in names:
                    module[child] = Observe(module[child])
                captured = {}
                if is_gdn:
                    conv = module._causal_conv1d_decode
                    normalize = module._normalize_qk

                    def capture_conv(x, _conv=conv, _captured=captured):
                        value = _conv(x)
                        _captured["conv"] = nn.silu(value)
                        return value

                    def capture_qk(q, k, _normalize=normalize, _captured=captured):
                        result = _normalize(q, k)
                        _captured["q"], _captured["k"] = result
                        return result

                    module._causal_conv1d_decode = capture_conv
                    module._normalize_qk = capture_qk
                if not parent.is_linear:
                    rotary = module.rotary_emb.apply_rotary

                    def capture_rotary(
                        q, k, *arguments, _rotary=rotary, _captured=captured, **kwargs
                    ):
                        result = _rotary(q, k, *arguments, **kwargs)
                        _captured["q_rope"], _captured["k_rope"] = result
                        return result

                    module.rotary_emb.apply_rotary = capture_rotary
                cache = ArraysCache(size=4) if parent.is_linear else QSAKVCache()
                report["attention"][str(index)] = []
                for step, old in enumerate(steps):
                    entry = old["layers"][0] if is_ple else old["components"][index][name]["input"]
                    x = mx.array(probe_input(args.attention_only, entry, 10240 if is_ple else 2560))
                    captured.clear()
                    if is_ple:
                        y = x + module(x, mx.array([[old["token"]]]), cache, None)
                    else:
                        y = module(
                            x.astype(mx.float16) if is_gdn else x,
                            mask=None,
                            cache=cache,
                            **({} if is_gdn else {"position_ids": mx.array([[step]], mx.int32)}),
                        )
                    if is_ple:
                        key = module.norm_key._output.reshape(1, 1, 4, 2560)
                        query = module.norm_query._output.reshape(1, 1, 4, 2560)
                        raw = mx.sum(key * query, axis=-1, keepdims=True) / (2560**0.5)
                        gate = mx.sigmoid(mx.sign(raw) * mx.sqrt(mx.maximum(mx.abs(raw), 1e-6)))
                        captured.update(
                            embedding=module.ple_embedding._output,
                            key=key,
                            query=query,
                            raw=raw,
                            gate=gate,
                            gated=module.norm_conv._input,
                            normed=module.norm_conv._output,
                            conv_raw=module.conv1d._output,
                            convolution=nn.silu(module.conv1d._output),
                        )
                    elif is_gdn:
                        captured["decay"], captured["beta"] = _compute_g_beta(
                            module.A_log,
                            module.in_proj_a._output,
                            module.in_proj_b._output,
                            module.dt_bias,
                        )
                        captured.update(
                            qkv=module.in_proj_qkv._output,
                            a=module.in_proj_a._output,
                            b=module.in_proj_b._output,
                            z=module.in_proj_z._output,
                            delta=module.norm._input,
                            normed=mx.fast.rms_norm(
                                module.norm._input, module.norm._layer.weight, 1e-6
                            ),
                            gated=module.norm._output,
                            v=captured["conv"][..., 4096:].reshape(1, 1, 48, 128),
                        )
                    else:
                        captured.update({child: module[child]._output for child in names[:-1]})
                        captured["k_norm"] = module.k_norm._output
                        captured["gated"] = module.o_proj._input
                    captured["out"] = y
                    prefix = f"a{index}-s{step}"
                    report["attention"][str(index)].append(
                        {
                            "token": old["token"],
                            "input": save(prefix + "-input", x),
                            "stages": {
                                key: save(prefix + "-" + key, value)
                                for key, value in captured.items()
                            },
                        }
                    )
                    print(f"attention {index} step {step + 1}/{len(steps)}", flush=True)
                del parent, module
                clear_layer_caches()
            report["status"] = "complete"
            return
        layers, caches = [], []
        for i in range(48):
            layer = Qwen4ExpDecoderLayer(config, i)
            load(layer, f"model.language_model.layers.{i}")
            layer.eval()
            if layer.training or (layer.is_linear and layer.linear_attn.training):
                raise ValueError("oracle layer is not in inference mode")
            if layer.is_linear:
                layer.linear_attn = HalfGdn(layer.linear_attn)
            for name in (
                "attn_hyper_connection",
                "mlp_hyper_connection",
                "mlp",
                "linear_attn" if layer.is_linear else "self_attn",
            ):
                layer[name] = Observe(layer[name])
            layers.append(layer)
            caches.append(ArraysCache(size=4) if layer.is_linear else QSAKVCache())
        mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        load(mixer, "model.language_model.hyper_connection_mixer")
        head = Packed("lm_head")
        for step, token in enumerate(tokens):
            hidden = mx.array(
                store.read("model.language_model.embed_tokens.weight", token).astype(np.float16)
            )[None, None]
            hidden = mx.tile(hidden, (1, 1, 4)).astype(mx.float32)
            trace = {"token": token, "layers": []}
            trace["components"] = []
            trace["route_ids"] = []
            for i, (layer, cache) in enumerate(zip(layers, caches)):
                hidden = layer(
                    hidden,
                    mx.array([[token]]),
                    mask=None,
                    cache=cache,
                    position_ids=mx.array([[step]], mx.int32),
                )
                mx.eval(hidden)
                trace["layers"].append(save(f"s{step}-l{i}", hidden))
                trace["route_ids"].append(sorted(layer.mlp._layer.switch_mlp._indices))
                observed = {}
                for name in (
                    "attn_hyper_connection",
                    "mlp_hyper_connection",
                    "mlp",
                    "linear_attn" if layer.is_linear else "self_attn",
                ):
                    component = layer[name]
                    observed[name] = {"input": save(f"s{step}-l{i}-{name}-input", component._input)}
                    value = component._output
                    if isinstance(value, tuple):
                        observed[name]["mixed"] = save(f"s{step}-l{i}-{name}-mixed", value[0])
                        observed[name]["injection"] = save(
                            f"s{step}-l{i}-{name}-injection", value[2]
                        )
                    else:
                        observed[name]["output"] = save(f"s{step}-l{i}-{name}-output", value)
                trace["components"].append(observed)
                if i in (0, 1):
                    trace[f"gdn_state_{i}"] = save(f"s{step}-state{i}", cache[1])
                print(f"oracle step {step + 1}/{len(tokens)} layer {i + 1}/48", flush=True)
            logits = head(mixer(hidden))
            trace["logits"] = save(f"s{step}-logits", logits)
            trace["argmax"] = int(mx.argmax(logits).item())
            report["steps"].append(trace)
            save_report()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        store.close()
        save_report()


if __name__ == "__main__":
    main()
