"""Replay Q04 with evolving native LFU caches; frozen gates do not predict quality or speed."""

import argparse
import array
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def choose_swaps(
    ids, probabilities, weights, ready, tau, starved, limit=1, protect=3, weight_cap=0.1
):
    """Return substitutions and miss-only recovery state, leaving inputs untouched."""
    if not math.isfinite(tau) or not 0 <= tau <= 0.4:
        raise ValueError("tolerance must be finite and within 0..0.4")
    if len(ids) != 10 or len(weights) != 10 or len(probabilities) != 512 or len(starved) != 512:
        raise ValueError("invalid router dimensions")
    if (
        type(limit) is not int
        or not 1 <= limit <= 10
        or type(protect) is not int
        or not 0 <= protect <= 10
    ):
        raise ValueError("invalid substitution limits")
    if not math.isfinite(weight_cap) or not 0 < weight_cap <= 1:
        raise ValueError("invalid weight cap")
    protected = set(sorted(range(10), key=lambda i: (-probabilities[ids[i]], ids[i]))[:protect])
    taken = set(ids)
    candidates = sorted(set(ready) - taken, key=lambda e: (-probabilities[e], e))
    updated, picks = starved.copy(), []
    # Source order is descending miss score, stable in the original route on ties.
    for rank in sorted(range(10), key=lambda i: -probabilities[ids[i]]):
        expert = ids[rank]
        if expert in ready:
            continue
        updated[expert] = 0  # An unchanged miss is loaded, including forced recovery.
        if (
            tau == 0
            or rank in protected
            or starved[expert] >= 3
            or len(picks) >= limit
            or not 0 < weights[rank] <= f32(weight_cap)
        ):
            continue
        if (
            candidates
            and probabilities[expert] > 0
            and probabilities[candidates[0]] >= f32(f32(1 - f32(tau)) * probabilities[expert])
        ):
            picks.append((rank, candidates.pop(0)))
            updated[expert] = starved[expert] + 1
    return picks, updated


class Cache:
    """Captured native LFU/age policy, with global frequency halving every 16 rows."""

    def __init__(self, capacity=53):
        if type(capacity) is not int or not 10 <= capacity <= 512:
            raise ValueError("invalid cache capacity")
        self.keys, self.ages = [None] * capacity, [0] * capacity
        self.counts, self.tick = [0] * 512, 0

    def ready(self):
        return {e for e in self.keys if e is not None}

    def reserve(self, ids):
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(ids) > len(self.keys)
            or any(type(e) is not int or not 0 <= e < 512 for e in ids)
        ):
            raise ValueError("invalid cache route")
        ready = self.ready()
        self.tick += 1
        if self.tick % 16 == 0:
            self.counts = [count // 2 for count in self.counts]
        for expert in sorted(ids):
            if expert in self.keys:
                slot = self.keys.index(expert)
            else:
                slot = min(
                    (s for s, e in enumerate(self.keys) if e not in ids),
                    key=lambda s: (
                        self.keys[s] is not None,
                        self.counts[self.keys[s]] if self.keys[s] is not None else 0,
                        self.ages[s],
                        s,
                    ),
                )
            self.keys[slot] = expert
            self.ages[slot] = self.tick
            self.counts[expert] += 1
        return len(set(ids) - ready)


def load_trace(folder):
    metadata = folder / "trace.json"
    payload = folder / "probabilities.f32"
    if metadata.stat().st_size > 128 * 1024**2 or payload.stat().st_size > 32768 * 512 * 4:
        raise ValueError("trace exceeds capture budget")
    trace = json.loads(metadata.read_text())
    rows, docs = trace.get("samples"), trace.get("documents")
    if (
        trace.get("status") != "complete_diagnostic_only"
        or not rows
        or not docs
        or len(rows) > 32768
        or trace.get("sample_count") != len(rows)
    ):
        raise ValueError("incomplete or empty trace")
    if any(type(d.get("input_rows")) is not int or d["input_rows"] < 16 for d in docs):
        raise ValueError("invalid document length")
    if sum(d["input_rows"] for d in docs) * 48 != len(rows):
        raise ValueError("incomplete layer coverage")
    data = payload.read_bytes()
    if len(data) != len(rows) * 512 * 4:
        raise ValueError("missing or trailing probability data")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    positions = [[0] * 48 for _ in docs]
    record_bytes = None
    for index, row in enumerate(rows):
        document, layer, position = (row.get(k) for k in ("document", "layer", "position"))
        if (
            any(type(v) is not int for v in (document, layer, position))
            or not 0 <= document < len(docs)
            or not 0 <= layer < 48
        ):
            raise ValueError("invalid document/layer")
        if position != positions[document][layer] or position >= docs[document]["input_rows"]:
            raise ValueError("duplicate, missing or reordered route")
        expected_document = next(
            (
                d
                for d, positions_by_layer in enumerate(positions)
                if min(positions_by_layer) < docs[d]["input_rows"]
            ),
            None,
        )
        if (
            document != expected_document
            or layer != index % 48
            or position != min(positions[document])
        ):
            raise ValueError("invalid route execution order")
        positions[document][layer] += 1
        ids, ready, weights = (row.get(k) for k in ("ids", "ready", "weights"))
        for collection, size in ((ids, 10), (ready, None)):
            if (
                not isinstance(collection, list)
                or (size is not None and len(collection) != size)
                or len(collection) > 53
                or len(set(collection)) != len(collection)
                or any(type(e) is not int or not 0 <= e < 512 for e in collection)
            ):
                raise ValueError("invalid expert IDs")
        probabilities = values[index * 512 : (index + 1) * 512]
        if (
            not isinstance(weights, list)
            or len(weights) != 10
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights)
            or not 0.98 <= sum(weights) <= 1.02
        ):
            raise ValueError("invalid mixture weights")
        if (
            any(not math.isfinite(v) or v < 0 for v in probabilities)
            or not 0.98 <= sum(probabilities) <= 1.02
        ):
            raise ValueError("invalid probabilities")
        hits, read_bytes = row.get("actual_hits"), row.get("actual_read_bytes")
        if (
            type(hits) is not int
            or hits != sum(e in ready for e in ids)
            or type(read_bytes) is not int
            or read_bytes < 0
        ):
            raise ValueError("invalid observed read counts")
        cold = 10 - hits
        if cold:
            if not read_bytes or read_bytes % cold:
                raise ValueError("invalid expert record size")
            size = read_bytes // cold
            if record_bytes is not None and size != record_bytes:
                raise ValueError("inconsistent expert record size")
            record_bytes = size
        elif read_bytes:
            raise ValueError("reads reported without a miss")
    return trace, values, record_bytes or 0


def analyse(folder):
    trace, values, record_bytes = load_trace(folder)
    rows = trace["samples"]
    caches = [Cache() for _ in range(48)]
    baseline_cold = 0
    for row in rows:
        cache = caches[row["layer"]]
        if cache.ready() != set(row["ready"]):
            raise ValueError(
                f"native LFU baseline mismatch at document/layer/position {row['document']}/{row['layer']}/{row['position']}"
            )
        cold = cache.reserve(row["ids"])
        if cold != 10 - row["actual_hits"]:
            raise ValueError("native LFU miss mismatch")
        if row["position"] >= 16:
            baseline_cold += cold
    decode = [(i, r) for i, r in enumerate(rows) if r["position"] >= 16]
    if not decode:
        raise ValueError("no decode routes")
    results = []
    for limit, protect, cap in ((1, 3, 0.1), (3, 3, 0.1), (10, 0, 1.0)):
        for tau in (0, 0.05, 0.1, 0.2, 0.3, 0.4) if protect else (0.2, 0.4):
            caches = [Cache() for _ in range(48)]
            states = {}
            stats = {
                "max_swaps": limit,
                "protected": protect,
                "weight_cap": cap,
                "tau": tau,
                "swaps": 0,
                "decode_cold_reads": 0,
                "moved_weight": 0.0,
            }
            for index, row in enumerate(rows):
                cache = caches[row["layer"]]
                starved = states.setdefault((row["document"], row["layer"]), [0] * 512)
                probabilities = values[index * 512 : (index + 1) * 512]
                ids = row["ids"].copy()
                if row["position"] >= 16:
                    picks, updated = choose_swaps(
                        ids,
                        probabilities,
                        row["weights"],
                        cache.ready(),
                        tau,
                        starved,
                        limit,
                        protect,
                        cap,
                    )
                    states[row["document"], row["layer"]] = updated
                    for rank, candidate in picks:
                        ids[rank] = candidate
                        stats["moved_weight"] += row["weights"][rank]
                    stats["swaps"] += len(picks)
                cold = cache.reserve(ids)
                if row["position"] >= 16:
                    stats["decode_cold_reads"] += cold
            saved = baseline_cold - stats["decode_cold_reads"]
            stats.update(
                net_modeled_saved_reads=saved,
                net_modeled_saved_logical_bytes=saved * record_bytes,
                net_modeled_cold_read_fraction=saved / baseline_cold if baseline_cold else None,
                mean_moved_weight_per_layer_row=stats.pop("moved_weight") / len(decode),
            )
            results.append(stats)
    return {
        "status": "complete_replay_only",
        "baseline_ready_sets_exact": True,
        "route_samples": len(rows),
        "decode_layer_rows": len(decode),
        "observed_cold_reads": baseline_cold,
        "logical_bytes_per_expert": record_bytes,
        "cache": "native LFU,53slots/layer,persistent across documents",
        "variants": results,
        "limitations": "Counterfactual cache state but frozen exact-model gates: no approximate-target quality, physical SSD bytes, or application gain measured. Probability thresholds are not bit-exact Sushi logit thresholds.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new result path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "running"}))
    try:
        result = analyse(args.trace)
        result["input_sha256"] = {
            name: hashlib.sha256((args.trace / name).read_bytes()).hexdigest()
            for name in ("trace.json", "probabilities.f32")
        }
    except BaseException as error:
        args.output.write_text(json.dumps({"status": "failed", "error": repr(error)}))
        raise
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
