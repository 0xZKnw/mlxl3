"""Offline AR opportunity only: reconstruct every physical LFU observation before scoring."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

LAYERS, EXPERTS, ROUTES, SLOTS = 48, 512, 10, 2544
EXPERT_BYTES = 1248012
CAPTURE_SHA256 = "fd8cee874b73379db02a6f2f7e22fe8426870331fba79ac167e166e36f4eb884"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def far_key(layers, current, layer, probability):
    """Jit EMA64 expected layer steps; zero probability is infinitely distant."""
    if not (
        type(layers) is int
        and layers > 0
        and 0 <= current < layers
        and 0 <= layer < layers
        and math.isfinite(probability)
        and 0 <= probability <= 1.000000000001
    ):
        raise ValueError("invalid cycle geometry or probability")
    return (
        math.inf
        if probability == 0
        else (layer - current - 1) % layers + 1 + layers * (1 - probability) / probability
    )


class Pool:
    """Demand-only slots; completed cohorts, no IO/GPU duration model.

    Per-layer farthest heads avoid scanning the whole shared pool per miss.
    Leased/pending slots are never victims; the historical replay has neither.
    """

    def __init__(self, layers, experts, capacity, cycle=False):
        if not all(type(v) is int and v > 0 for v in (layers, experts, capacity)):
            raise ValueError("positive integer dimensions required")
        self.layers, self.experts, self.capacity, self.cycle = layers, experts, capacity, cycle
        self.keys = [None] * capacity
        self.where = {}
        self.resident = [set() for _ in range(layers)]
        self.counts = [[0] * experts for _ in range(layers)]
        self.visits = [0] * layers
        self.weights = [[0.0] * experts for _ in range(layers)]
        self.scale = [1.0] * layers
        self.ages = [0] * capacity
        self.generations = [0] * capacity
        self.clock = 0
        self.free = list(range(capacity))
        self.heads = [None] * layers
        self.mode = self.previous_layer = None
        self.had_leases = False
        self.cycle_observations = 0

    def _head(self, layer, protected, blocked, cyclic):
        if self.heads[layer] is None:
            row = self.weights[layer] if cyclic else self.counts[layer]
            eligible = (
                slot
                for slot in self.resident[layer]
                if slot not in blocked and self.keys[slot] not in protected
            )
            self.heads[layer] = min(
                eligible,
                key=lambda slot: (row[self.keys[slot][1]], self.ages[slot], slot),
                default=-1,
            )
        return self.heads[layer]

    def reserve(self, layer, ids, decode, leased=(), pending=()):
        if type(layer) is not int or not 0 <= layer < self.layers:
            raise ValueError("layer outside pool")
        if not ids or any(type(e) is not int or not 0 <= e < self.experts for e in ids):
            raise ValueError("invalid expert IDs")
        unique = sorted(set(ids))
        if len(unique) > self.capacity:
            raise ValueError("cohort exceeds pool")
        protected = {(layer, e) for e in unique}
        blocked = set(leased) | set(pending)
        if any(type(s) is not int or not 0 <= s < self.capacity for s in blocked):
            raise ValueError("invalid lease slot")
        if blocked:
            misses = sum(key not in self.where for key in protected)
            available = sum(
                slot not in blocked and key not in protected for slot, key in enumerate(self.keys)
            )
            if available < misses:
                raise ValueError("all potential victims are protected")
        cyclic = self.cycle and decode
        if cyclic != self.mode or blocked or self.had_leases:
            self.heads = [None] * self.layers
        self.mode, self.had_leases = cyclic, bool(blocked)
        if self.previous_layer is not None:
            self.heads[self.previous_layer] = None  # its previous cohort is no longer pinned
        self.heads[layer] = None
        self.previous_layer = layer
        self.clock += 1
        self.visits[layer] += 1
        if self.visits[layer] % 16 == 0:
            self.counts[layer] = [n // 2 for n in self.counts[layer]]
        if cyclic:
            self.scale[layer] *= 63 / 64
            if self.scale[layer] < 1e-100:
                self.weights[layer] = [w * self.scale[layer] for w in self.weights[layer]]
                self.scale[layer] = 1.0
            for expert in unique:
                self.weights[layer][expert] += (1 / 64) / self.scale[layer]
            self.cycle_observations += 1
        mapped = {}
        for expert in unique:
            key = (layer, expert)
            slot = self.where.get(key)
            hit = slot is not None
            if not hit:
                held = []
                while self.free and self.free[0] in blocked:
                    held.append(heapq.heappop(self.free))
                slot = heapq.heappop(self.free) if self.free else None
                for free in held:
                    heapq.heappush(self.free, free)
                if slot is not None:
                    pass
                else:
                    candidates = []
                    for other in range(self.layers):
                        head = self._head(other, protected, blocked, cyclic)
                        if head < 0:
                            continue
                        old_expert = self.keys[head][1]
                        rank = (
                            -far_key(
                                self.layers,
                                layer,
                                other,
                                self.weights[other][old_expert] * self.scale[other],
                            )
                            if cyclic
                            else self.counts[other][old_expert]
                        )
                        candidates.append((rank, self.ages[head], head))
                    if not candidates:
                        raise ValueError("no unleased victim")
                    slot = min(candidates)[2]
                    old = self.keys[slot]
                    del self.where[old]
                    self.resident[old[0]].remove(slot)
                    self.heads[old[0]] = None
                self.keys[slot] = key
                self.where[key] = slot
                self.resident[layer].add(slot)
                self.generations[slot] += 1
            self.counts[layer][expert] = min(2**32 - 1, self.counts[layer][expert] + 1)
            self.ages[slot] = self.clock
            self.heads[layer] = None
            mapped[expert] = (slot, hit)
        return [mapped[e] for e in ids]


def validate_capture(capture):
    if set(capture) != {"status", "initial_generations", "cases"}:
        raise ValueError("wrong capture fields")
    if capture["status"] != "complete_ar_routes" or capture["initial_generations"] != 2:
        raise ValueError("incomplete or wrong initialization")
    if len(capture["cases"]) != 2:
        raise ValueError("MAIN and HOLD required")
    for case, name in zip(capture["cases"], ("MAIN", "HOLD"), strict=True):
        if set(case) != {"name", "prompt_tokens", "phases"} or case["name"] != name:
            raise ValueError("invalid case")
        if type(case["prompt_tokens"]) is not int or not 1 <= case["prompt_tokens"] <= 512:
            raise ValueError("invalid prompt length")
        if len(case["phases"]) != 3:
            raise ValueError("three observed phases required")
        for phase, label in zip(case["phases"], ("trace_1", "trace_2", "trace_3"), strict=True):
            if set(phase) != {"name", "before", "after_prefill", "after", "prefill", "decode"}:
                raise ValueError("invalid phase fields")
            if phase["name"] != label:
                raise ValueError("wrong phase order")
            for key in ("before", "after_prefill", "after"):
                counts = phase[key]
                if len(counts) != LAYERS or any(
                    len(row) != 3
                    or any(type(n) is not int or not 0 <= n < 2**64 for n in row)
                    or row[2] != row[1] * EXPERT_BYTES
                    for row in counts
                ):
                    raise ValueError("invalid physical counters")
            for part, rows in (("prefill", case["prompt_tokens"]), ("decode", 127)):
                if len(phase[part]) != LAYERS * rows:
                    raise ValueError("incomplete layer/row sequence")
                for observation in phase[part]:
                    if set(observation) != {"ids", "slots_hit"}:
                        raise ValueError("wrong observation fields")
                    ids, slots = observation["ids"], observation["slots_hit"]
                    if (
                        len(ids) != ROUTES
                        or len(set(ids)) != ROUTES
                        or any(type(e) is not int or not 0 <= e < EXPERTS for e in ids)
                        or len(slots) != ROUTES
                        or any(
                            len(pair) != 2
                            or type(pair[0]) is not int
                            or not 0 <= pair[0] < 53
                            or type(pair[1]) is not bool
                            for pair in slots
                        )
                    ):
                        raise ValueError("invalid routes or physical slots")


def observe(pool, layer, observation, counters, decode, check=False):
    mapped = pool.reserve(layer, observation["ids"], decode)
    if check and mapped != [tuple(pair) for pair in observation["slots_hit"]]:
        raise ValueError("physical slot/hit reconstruction mismatch")
    misses = sum(not hit for _, hit in mapped)
    for i, delta in enumerate((len(mapped) - misses, misses, misses * EXPERT_BYTES)):
        counters[i] += delta
    return len(mapped) - misses


def replay(case, policy, deadline):
    local = policy == "physical_per_layer_LFU53"
    pools = (
        [Pool(1, EXPERTS, 53) for _ in range(LAYERS)]
        if local
        else [Pool(LAYERS, EXPERTS, SLOTS, cycle=policy == "global_EMA64_cycle")]
    )
    counters = [[0, 0, 0] for _ in range(LAYERS)]
    phases = []
    for generation in range(5):
        phase = case["phases"][max(0, generation - 2)]
        if local and generation >= 2 and counters != phase["before"]:
            raise ValueError("initial/cumulative physical state differs")
        for part, expected in (("prefill", "after_prefill"), ("decode", "after")):
            before = [row.copy() for row in counters]
            histogram = [0] * 11
            eligible = [0] * 11
            for index, observation in enumerate(phase[part]):
                if index % 48 == 0 and time.monotonic() >= deadline:
                    raise TimeoutError("bounded offline replay deadline")
                layer = index % LAYERS
                hits = observe(
                    pools[layer] if local else pools[0],
                    0 if local else layer,
                    observation,
                    counters[layer],
                    part == "decode",
                    check=local and generation >= 2,
                )
                histogram[hits] += 1
                # Source eligibility only: successor regular GDN, neither PLE(index2) nor QSA.
                if layer + 1 < LAYERS and (layer + 1) % 4 != 3 and layer + 1 != 2:
                    eligible[hits] += 1
            if local and generation >= 2 and counters != phase[expected]:
                raise ValueError("physical phase/layer counters differ")
            if generation >= 2:
                delta = [
                    [a - b for a, b in zip(row, old, strict=True)]
                    for row, old in zip(counters, before, strict=True)
                ]
                totals = [sum(row[j] for row in delta) for j in range(3)]
                if totals[0] + totals[1] != len(phase[part]) * ROUTES:
                    raise ValueError("empty or incomplete score")
                resident = (
                    [len(p.resident[0]) for p in pools]
                    if local
                    else [len(s) for s in pools[0].resident]
                )
                if sum(resident) > SLOTS:
                    raise ValueError("shared capacity exceeded")
                phases.append(
                    {
                        "phase": phase["name"],
                        "part": part,
                        "cohorts": len(phase[part]),
                        "hits": totals[0],
                        "misses": totals[1],
                        "logical_read_bytes": totals[2],
                        "per_layer_hits_misses_bytes": delta,
                        "hit_count_histogram": histogram,
                        "regular_GDN_successor_hit_histogram": eligible,
                        "resident_slots_by_layer": resident,
                    }
                )
    return {
        "name": case["name"],
        "phases": phases,
        "cycle_observations": sum(p.cycle_observations for p in pools),
    }


def analyze(capture, deadline):
    validate_capture(capture)
    variants = []
    for policy in ("physical_per_layer_LFU53", "global_LFU_control", "global_EMA64_cycle"):
        cases = [replay(case, policy, deadline) for case in capture["cases"]]
        variants.append({"policy": policy, "cases": cases})
    return {
        "status": "complete_simulation_not_speed",
        "confirmed_tps_gain": False,
        "numerical_parity": False,
        "baseline_physical_slots_hits_counters": "exact_both_cases_all_three_phases",
        "total_slots": SLOTS,
        "expert_bank_bytes": SLOTS * EXPERT_BYTES,
        "initial_generations": 2,
        "cycle_period": 64,
        "cycle_armed": "decode_only; prefill uses per-layer-decayed LFU global victims",
        "initial_state": "each policy empty; two full first-phase generations before scoring",
        "concurrency": "demand-only completed cohorts; no timed IO/GPU leases in historical data",
        "variants": variants,
    }


def run(source, output, expected_sha=CAPTURE_SHA256, seconds=45.0):
    report = {"status": "incomplete", "confirmed_tps_gain": False, "numerical_parity": False}
    with output.open("x") as stream:
        json.dump(report, stream)
    started = time.monotonic()
    try:
        if not math.isfinite(seconds) or not 0 < seconds <= 55:
            raise ValueError("replay budget must be finite in (0,55]")
        if not 0 < source.stat().st_size <= 32 << 20:
            raise ValueError("invalid input extent")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha:
            raise ValueError("capture SHA256 differs")
        capture = json.loads(raw, object_pairs_hook=unique_object)
        report = analyze(capture, started + seconds)
        report.update(
            capture_sha256=digest,
            input=str(source),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
    except BaseException as error:
        report.update(
            status="failed", confirmed_tps_gain=False, error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=CAPTURE_SHA256)
    parser.add_argument("--seconds", type=float, default=45.0)
    args = parser.parse_args()
    result = run(args.capture, args.out, args.expected_sha256, args.seconds)
    print(json.dumps({key: result[key] for key in ("status", "seconds", "capture_sha256")}))
