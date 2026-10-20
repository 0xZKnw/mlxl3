"""Independent NumPy/BF16 oracle for the two pinned Flash Next drafts.

Equations: SGLang 3c560089 DFlash2 residual norm and e00930c5 DSpark (Apache-2.0).
No MLX, torch, target weights in RAM, or remote model code is imported.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

CAPTURES = {"dflash2": [5, 15, 25, 35, 45], "dspark": [1, 12, 23, 34, 45]}
MASKS = {"dflash2": 248077, "dspark": 248070}
REVISION = {
    "dflash2": "a22a1744e85c17364dcc54a6a420af7135d4c7f6",
    "dspark": "aafaf1bf59fdc4639183600d7747474990623171",
}


def native_diagnostic(log, kind, category, oracle, width=3, coupled=None):
    """Reject partial, nonfinite and oversized native qualification reports."""
    if oracle.get("status") != "complete_cpu_oracle":
        raise ValueError("incomplete independent oracle")
    markers = {
        "numpy": "FLASH_DRAFT_NUMPY ",
        "prefix": "FLASH_DRAFT_PREFIX ",
        "greedy": "FLASH_DRAFT_GREEDY ",
    }
    marker = markers[category]
    lines = [line.split(marker, 1)[1] for line in log.splitlines() if marker in line]
    if len(lines) != 1:
        raise ValueError("missing or duplicate native diagnostic")
    data = json.loads(lines[0])
    expected_kind = {"dflash2": "Dflash2", "dspark": "Dspark"}[kind]
    if data["kind"] != expected_kind:
        raise ValueError("foreign native draft kind")
    peak = data["memory"]["process_lifetime_peak_bytes"]
    if type(peak) is not int or not 0 < peak <= 10 << 30:
        raise ValueError("invalid native process peak")

    def number(value, lower=0):
        if type(value) not in [int, float] or not math.isfinite(value) or value < lower:
            raise ValueError("invalid native numeric value")

    def integer(value):
        if type(value) is not int or value < 0:
            raise ValueError("invalid native counter")

    def comparison(entry, name, tolerance):
        if (
            entry["name"] != name
            or type(entry["count"]) is not int
            or entry["count"] != oracle["arrays"][name]["count"]
        ):
            raise ValueError("incomplete native numerical comparison")
        number(entry["relative_rmse"])
        number(entry["tolerance"])
        if entry["tolerance"] != tolerance or entry["relative_rmse"] > tolerance:
            raise ValueError("native numerical mismatch")

    if category == "numpy":
        if data["status"] != "passed":
            raise ValueError("native backbone did not pass")
        names = (
            [name for i in range(5) for name in [f"keys-{i}", f"values-{i}"]]
            + [f"layer-{i}" for i in range(5)]
            + ["hidden"]
        )
        if len(data["checks"]) != len(names):
            raise ValueError("incomplete backbone/cache checks")
        for entry, name in zip(data["checks"], names, strict=True):
            comparison(entry, name, 0.01 if name.startswith(("keys", "values")) else 0.02)
        if data["selector_tokens"] != oracle["selector_tokens"] or any(
            type(v) is not int or not 0 <= v < 248320 for v in data["selector_tokens"]
        ):
            raise ValueError("native selector diverged")
        if len(data["confidence"]) != len(oracle["selector_confidence"]):
            raise ValueError("incomplete confidence head")
        for value, expected in zip(data["confidence"], oracle["selector_confidence"], strict=True):
            number(value)
            if value > 1 or abs(value - expected) > 1e-4:
                raise ValueError("native confidence differs")
        if coupled is not None:
            actual = data["coupled"]
            count = oracle.get("proposals")
            if (
                type(count) is not int
                or not 1 <= count <= 7
                or coupled["kind"] != kind
                or coupled["revision"] != oracle["revision"]
                or coupled["hidden_sha256"] != oracle["arrays"]["hidden"]["sha256"]
                or len(coupled["tokens"]) != count
                or any(
                    type(token) is not int or not 0 <= token < 248320 for token in coupled["tokens"]
                )
            ):
                raise ValueError("incomplete or foreign coupled reference")
            if (
                actual["tokens"] != coupled["tokens"]
                or actual["cpu_hidden_tokens"] != coupled["tokens"]
                or any(
                    type(token) is not int
                    for key in ("tokens", "cpu_hidden_tokens")
                    for token in actual[key]
                )
            ):
                raise ValueError("real coupled proposals diverged")
            check = actual["head"]
            if (
                check["name"] != "coupled_logits"
                or check["count"] != oracle["proposals"] * 248320
                or check["tolerance"] != 0.004
            ):
                raise ValueError("incomplete real head comparison")
            number(check["relative_rmse"])
            if check["relative_rmse"] > 0.004:
                raise ValueError("real head logits diverged")
            expected_count = oracle["proposals"] if kind == "dspark" else 0
            if len(actual["confidence"]) != expected_count:
                raise ValueError("incomplete coupled confidences")
            for value in actual["confidence"]:
                number(value)
                if value > 1:
                    raise ValueError("invalid coupled confidence")
    elif category == "prefix":
        if (
            data["status"] != "passed"
            or type(data["prefix_cases"]) is not int
            or data["prefix_cases"] != 15
            or data["widths"] != [1, 2, 4, 8]
            or any(type(v) is not int for v in data["widths"])
            or data["target_logits_and_states"] != "byte_exact"
        ):
            raise ValueError("incomplete prefix state verification")
        comparison(data["captures"], "captured", 0.004)
    else:
        if (
            data["status"] != "passed_quality_not_speed"
            or type(data["width"]) is not int
            or data["width"] != width
            or [r["budget"] for r in data["results"]] != [1, 2, 3, 17, 64]
        ):
            raise ValueError("incomplete greedy generation")
        for record in data["results"]:
            for key in ["budget", "accepted", "proposed", "blocks", "target_offset"]:
                integer(record[key])
            if (
                record["accepted"] > record["proposed"]
                or not record["budget"] - 1 <= record["target_offset"] <= 4096
            ):
                raise ValueError("invalid acceptance or context counters")
            for key in ["draft_seconds", "verify_seconds", "decode_seconds"]:
                number(record[key])
            if (record["budget"] == 1 and record["decode_seconds"] != 0) or (
                record["budget"] > 1 and record["decode_seconds"] == 0
            ):
                raise ValueError("empty or measured decode duration disagrees with budget")
    return data


def bf16(values):
    """Round finite float32 to BF16, ties to even, retaining float32 storage."""
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite BF16 input")
    bits = array.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16) << 16
    result = rounded.view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 overflow")
    return result


class Weights:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= 1 << 20:
                raise ValueError("invalid draft header size")
            self.header = json.loads(stream.read(size))
        self.header.pop("__metadata__", None)
        self.offset = 8 + size
        self.data = np.memmap(self.path, mode="r", dtype=np.uint16, offset=self.offset)
        end = 0
        for name, info in sorted(self.header.items(), key=lambda item: item[1]["data_offsets"][0]):
            count = int(np.prod(info["shape"], dtype=np.int64))
            first, last = info["data_offsets"]
            if info["dtype"] != "BF16" or count <= 0 or first != end or last - first != count * 2:
                raise ValueError(f"invalid BF16 tensor {name}")
            end = last
        if end != self.data.nbytes:
            raise ValueError("incomplete tensor data")

    def __call__(self, name, indices=None):
        info = self.header[name]
        first, last = info["data_offsets"]
        raw = self.data[first // 2 : last // 2].reshape(info["shape"])
        if indices is not None:
            raw = raw[indices]
        return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)

    def finite(self):
        for first in range(0, self.data.size, 1 << 20):
            chunk = (self.data[first : first + (1 << 20)].astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("nonfinite weight payload")
        return int(self.data.size)


def capture(reference, ids):
    reference = Path(reference)
    data = json.loads((reference / "reference.json").read_text())
    if data["status"] != "complete" or data["mode"] != "eval" or len(data["steps"]) < 5:
        raise ValueError("incomplete target oracle")
    rows = []
    for step in data["steps"]:
        layers = []
        for layer in ids:
            entry = step["layers"][layer]
            array = np.fromfile(reference / entry["file"], dtype="<f4")
            if entry["count"] != 10240 or array.size != 10240 or not np.isfinite(array).all():
                raise ValueError("invalid target capture")
            # Published capture: FP32 mean of four completed HC streams.
            layers.append(bf16(array.reshape(4, 2560).mean(axis=0, dtype=np.float32)))
        rows.append(np.concatenate(layers))
    return np.stack(rows), [step["token"] for step in data["steps"]]


def embeddings(model, ids):
    key = "model.language_model.embed_tokens.weight"
    for path in sorted(Path(model).glob("*.safetensors")):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
            if key not in header:
                continue
            info = header[key]
            if info["dtype"] != "BF16" or info["shape"] != [248320, 2560]:
                raise ValueError("invalid target embedding")
            result = []
            for token in ids:
                if not 0 <= token < 248320:
                    raise ValueError("embedding token outside vocabulary")
                stream.seek(8 + length + info["data_offsets"][0] + token * 5120)
                raw = np.frombuffer(stream.read(5120), dtype="<u2")
                if raw.size != 2560:
                    raise ValueError("truncated embedding row")
                result.append((raw.astype(np.uint32) << 16).view(np.float32))
            return np.stack(result)
    raise ValueError("target embedding missing")


def rope(array, theta, start):
    rows = array.shape[-2]
    frequency = np.float32(theta) ** (-np.arange(0, 128, 2, dtype=np.float32) / 128)
    angle = np.arange(start, start + rows, dtype=np.float32)[:, None] * frequency
    cosine, sine = np.cos(angle), np.sin(angle)
    left, right = array[..., :64], array[..., 64:]
    return bf16(
        np.concatenate([left * cosine - right * sine, left * sine + right * cosine], axis=-1)
    )


class Oracle:
    def __init__(self, weights, kind):
        self.w = weights
        self.kind = kind
        self.theta = 10000 if kind == "dflash2" else 10000000

    def linear(self, x, prefix):
        return bf16(x @ self.w(prefix + ".weight").T)

    def norm(self, x, prefix):
        inverse = 1 / np.sqrt(
            np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6)
        )
        return bf16(x * inverse * self.w(prefix + ".weight"))

    def residual_norm(self, x, residual, prefix):
        total = np.asarray(x, dtype=np.float32) + np.asarray(residual, dtype=np.float32)
        if (
            total.ndim != 2
            or not 1 <= total.shape[0] <= 8
            or total.shape != x.shape
            or total.shape != residual.shape
            or total.shape[-1] != 2560
        ):
            raise ValueError("invalid residual norm shape")
        retained = bf16(total)
        inverse = 1 / np.sqrt(
            np.mean(total * total, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6)
        )
        return bf16(retained * self.w(prefix + ".weight") * inverse), retained

    def convolve(self, x, dynamic, prefix, side):
        rows = x.shape[0]
        delta = dynamic.reshape(rows, 2, 2, 160, 1)[:, side]
        coefficient = bf16(
            self.w(prefix + ".base_kernel")[side]
            + np.broadcast_to(delta, (rows, 2, 160, 16)).reshape(rows, 2, 2560)
        )
        shifted = np.concatenate([np.zeros((1, 2560), dtype=np.float32), x[:-1]])
        return bf16(bf16(x * coefficient[:, 0]) + bf16(shifted * coefficient[:, 1]))

    def prepare(self, x, prefix):
        if self.kind == "dspark":
            return x, None
        dynamic = self.linear(x, prefix + ".kernel_projection")
        return self.convolve(x, dynamic, prefix, 0), dynamic

    def finish(self, x, dynamic, prefix):
        if dynamic is not None:
            x = self.convolve(x, dynamic, prefix, 1)
        return x

    def append(self, captured):
        rows = captured.shape[0]
        context = self.norm(self.linear(captured, "fc"), "hidden_norm")
        cache = []
        for layer in range(5):
            p = f"layers.{layer}.self_attn"
            k = self.norm(
                self.linear(context, p + ".k_proj").reshape(1, rows, 8, 128), p + ".k_norm"
            ).transpose(0, 2, 1, 3)
            k = rope(k, self.theta, 0)
            v = self.linear(context, p + ".v_proj").reshape(1, rows, 8, 128).transpose(0, 2, 1, 3)
            cache.append((k, v))
        return cache

    def attention(self, x, cache, layer, position):
        rows = x.shape[0]
        p = f"layers.{layer}.self_attn"
        q = self.norm(
            self.linear(x, p + ".q_proj").reshape(1, rows, 32, 128), p + ".q_norm"
        ).transpose(0, 2, 1, 3)
        k = self.norm(
            self.linear(x, p + ".k_proj").reshape(1, rows, 8, 128), p + ".k_norm"
        ).transpose(0, 2, 1, 3)
        v = self.linear(x, p + ".v_proj").reshape(1, rows, 8, 128).transpose(0, 2, 1, 3)
        q, k = rope(q, self.theta, position), rope(k, self.theta, position)
        old_k, old_v = cache[layer]
        k = np.repeat(np.concatenate([old_k, k], axis=2), 4, axis=1)
        v = np.repeat(np.concatenate([old_v, v], axis=2), 4, axis=1)
        scores = (q @ k.swapaxes(-1, -2)) * np.float32(128**-0.5)
        if self.kind == "dflash2":
            past = old_k.shape[2]
            for row in range(rows):
                scores[:, :, row, : max(0, past + row + 1 - 2048)] = -np.inf
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attended = bf16(probabilities @ v).transpose(0, 2, 1, 3).reshape(rows, 4096)
        return self.linear(attended, p + ".o_proj")

    def forward(self, x, cache, position):
        trace = []
        normalized = self.norm(x, "layers.0.input_layernorm")
        for layer in range(5):
            p = f"layers.{layer}"
            prepared, dynamic = self.prepare(normalized, p + ".attention_conv")
            attention = self.finish(
                self.attention(prepared, cache, layer, position), dynamic, p + ".attention_conv"
            )
            mlp_input, residual = self.residual_norm(attention, x, p + ".post_attention_layernorm")
            prepared, dynamic = self.prepare(mlp_input, p + ".mlp_conv")
            gate = self.linear(prepared, p + ".mlp.gate_proj")
            up = self.linear(prepared, p + ".mlp.up_proj")
            activated = bf16(gate / (1 + np.exp(-gate)) * up)
            mlp = self.finish(
                self.linear(activated, p + ".mlp.down_proj"), dynamic, p + ".mlp_conv"
            )
            next_norm = "norm" if layer == 4 else f"layers.{layer + 1}.input_layernorm"
            normalized, x = self.residual_norm(mlp, residual, next_norm)
            trace.append(x)
        return normalized, trace


def save_array(directory, name, value):
    value = np.asarray(value, dtype="<f4")
    if not value.size or not np.isfinite(value).all():
        raise ValueError("empty/nonfinite oracle output")
    path = directory / (name + ".f32")
    value.tofile(path)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "count": int(value.size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def select_cpu(weights, kind, hidden, logits, anchor):
    """Reference equations on real backbone output and independent EXL3 logits."""
    if (
        kind not in CAPTURES
        or type(anchor) is not int
        or not 0 <= anchor < 248320
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= 7
        or logits.shape[1] != 248320
        or hidden.shape != (logits.shape[0], 2560)
        or not np.isfinite(hidden).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid coupled selector inputs")
    tokens, confidence = [], []
    previous = anchor
    if kind == "dflash2":
        projected = bf16(hidden @ weights("candidate_selector.hidden_projection.weight").T)
        vocabulary = np.arange(248320)
        for row in range(logits.shape[0]):
            ids = np.lexsort((vocabulary, -logits[row]))[:16]
            pred = weights("candidate_selector.predecessor_codebook", [previous])[0]
            keys = weights("candidate_selector.successor_codebook", ids)
            edges = bf16(bf16(pred * projected[row]) @ keys.T)
            previous = int(ids[np.argmax(logits[row, ids] + edges)])
            tokens.append(previous)
    else:
        matrix = weights("markov_head.markov_w2.weight")
        for row in range(logits.shape[0]):
            latent = weights("markov_head.markov_w1.weight", [previous])[0]
            corrected = bf16(logits[row] + bf16(latent @ matrix.T))
            features = np.concatenate([hidden[row], latent])
            raw = float(
                features @ weights("confidence_head.proj.weight")[0]
                + weights("confidence_head.proj.bias")[0]
            )
            if not math.isfinite(raw):
                raise ValueError("nonfinite coupled confidence")
            probability = math.exp(-abs(raw))
            confidence.append(
                1 / (1 + probability) if raw >= 0 else probability / (1 + probability)
            )
            previous = int(corrected.argmax())
            tokens.append(previous)
    return tokens, confidence


def selector_fixture(weights, kind, rows):
    """Choose nontrivial logits that expose legacy scaling or a stale Markov predecessor."""
    hidden = bf16(((np.arange(rows * 2560, dtype=np.float32) % 127) - 63).reshape(rows, 2560) / 64)
    logits = np.full((rows, 248320), -10000, dtype=np.float32)
    tokens, confidences = [], []
    previous = 7
    if kind == "dflash2":
        state = bf16(hidden @ weights("candidate_selector.hidden_projection.weight").T)
        ids = np.arange(17, 33)
        keys = weights("candidate_selector.successor_codebook", ids)
        for row in range(rows):
            pred = weights("candidate_selector.predecessor_codebook", [previous])[0]
            edge = bf16(bf16(pred * state[row]) @ keys.T)
            low, high = int(edge.argmin()), int(edge.argmax())
            gap = float(edge[high] - edge[low])
            if not gap > 1e-5:
                raise ValueError("selector fixture has no transition separation")
            unary = np.full(16, -1000, dtype=np.float32)
            unary[high], unary[low] = 0, gap / 2
            unary = bf16(unary)
            order = np.lexsort((ids, -unary))
            correct = int(order[np.argmax(unary[order] + edge[order])])
            legacy = int(order[np.argmax(unary[order] + edge[order] * 0.25)])
            if correct == legacy or correct != high:
                raise ValueError("selector scale mutation would be invisible")
            logits[row, ids] = unary
            previous = int(ids[correct])
            tokens.append(previous)
    else:
        matrix = weights("markov_head.markov_w2.weight")
        anchor = weights("markov_head.markov_w1.weight", [7])[0]
        for row in range(rows):
            latent = weights("markov_head.markov_w1.weight", [previous])[0]
            bias = bf16(latent @ matrix.T)
            desired = 17 + 13 * row
            delta = 2.0
            if row:
                wrong = bf16(anchor @ matrix.T) - bias
                desired = int(wrong.argmin())
                delta = max(0.001, float(wrong.max() - wrong[desired]) / 2)
            base = -bias.copy()
            base[desired] += delta
            logits[row] = bf16(base)
            corrected = bf16(logits[row] + bias)
            selected = int(corrected.argmax())
            if selected != desired:
                raise ValueError("Markov fixture lacks a unique intended winner")
            if row and int(bf16(logits[row] + bf16(anchor @ matrix.T)).argmax()) == selected:
                raise ValueError("stale predecessor mutation would be invisible")
            features = np.concatenate([hidden[row], latent])
            raw = float(
                features @ weights("confidence_head.proj.weight")[0]
                + weights("confidence_head.proj.bias")[0]
            )
            confidences.append(float(1 / (1 + np.exp(-np.float64(raw)))))
            previous = selected
            tokens.append(previous)
    return hidden, bf16(logits), tokens, confidences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=list(CAPTURES), required=True)
    parser.add_argument("--proposals", type=int, choices=[1, 3, 7], default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "kind": args.kind,
        "proposals": args.proposals,
        "revision": REVISION[args.kind],
        "arrays": {},
    }
    path = args.output / "oracle.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        weights = Weights(args.draft / "model.safetensors")
        report["finite_parameters"] = weights.finite()
        captured, tokens = capture(args.reference, CAPTURES[args.kind])
        rows = args.proposals + int(args.kind == "dflash2")
        input_ids = [7] + [MASKS[args.kind]] * (rows - 1)
        noise = embeddings(args.model, input_ids)
        oracle = Oracle(weights, args.kind)
        cache = oracle.append(captured)
        hidden, trace = oracle.forward(noise, cache, len(tokens))
        if (
            len(trace) != 5
            or len(cache) != 5
            or hidden.shape != (rows, 2560)
            or any(value.shape != hidden.shape for value in trace)
            or any(k.shape != (1, 8, len(tokens), 128) or v.shape != k.shape for k, v in cache)
        ):
            raise ValueError("incomplete draft backbone/cache output")
        selector_hidden, selector_logits, selector_tokens, selector_confidence = selector_fixture(
            weights, args.kind, args.proposals
        )
        arrays = {"captured": captured, "input": noise, "hidden": hidden}
        arrays.update(selector_hidden=selector_hidden, selector_logits=selector_logits)
        arrays.update({f"layer-{i}": value for i, value in enumerate(trace)})
        arrays.update({f"keys-{i}": pair[0] for i, pair in enumerate(cache)})
        arrays.update({f"values-{i}": pair[1] for i, pair in enumerate(cache)})
        report["arrays"] = {
            name: save_array(args.output, name, array) for name, array in arrays.items()
        }
        report.update(
            status="complete_cpu_oracle",
            tokens=tokens,
            input_ids=input_ids,
            target_reference=str(args.reference),
            draft_directory=str(args.draft.resolve()),
            residual_norm="sglang-3c560089-fp32-variance-bf16-residual",
            norm_relative_rmse_limit=0.02,
            cache_relative_rmse_limit=0.01,
            selector_tokens=selector_tokens,
            selector_confidence=selector_confidence,
        )
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=type(error).__name__, reason=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "oracle": str(path),
                "parameters": report["finite_parameters"],
            }
        )
    )


if __name__ == "__main__":
    main()
