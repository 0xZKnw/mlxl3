"""Independent Flash Next oracle: mlx-vlm architecture and upstream PonyExl3.

Only selected experts and PLE rows are read. Never instantiate a second target
checkpoint in the same process. Outputs are correctness fixtures, not benchmarks.
"""

import argparse
import json
import os
import struct
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pony", type=Path, default=Path("build/research-ponyexl3"))
    parser.add_argument("--tokens", default="248044,1,42,248044,7")
    args = parser.parse_args()
    tokens = parse_tokens(args.tokens)
    sys.path.insert(0, str(args.pony.resolve()))
    import mlx.core as mx
    from mlx import nn
    from mlx_vlm.models.activations import swiglu
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
        config = TextConfig.from_dict(
            json.loads((args.model / "config.json").read_text())["text_config"]
        )
        layers, caches = [], []
        for i in range(48):
            layer = Qwen4ExpDecoderLayer(config, i)
            load(layer, f"model.language_model.layers.{i}")
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
