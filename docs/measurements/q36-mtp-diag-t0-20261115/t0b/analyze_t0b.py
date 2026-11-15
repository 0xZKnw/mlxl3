"""Q36-MTP-CYCLE-TIMELINE-T0B: attribute GPU idle time of MTP cycles to host synchronizations.

Inputs: run directories with diag.json (mtp_cycle_diagnostics, confidence 0) and cb.bin (P4 dylib
22c83ffa..., 80-byte records, CACurrentMediaTime seconds; host marks use CLOCK_UPTIME_RAW ns, the
same mach base). Every check fails closed and the summary keeps status "failed" with the reasons.
Usage: python -I analyze_t0b.py <output.json> <nodylib run> <run1> [<run2> ...]
"""

import json
import math
import struct
import sys
from pathlib import Path

FORMAT = "<QQddddddqQ"
SIZE = struct.calcsize(FORMAT)
FIELDS = (
    "magic",
    "sequence",
    "commit",
    "kernel_start",
    "kernel_end",
    "gpu_start",
    "gpu_end",
    "handler",
    "status",
    "pad",
)
COMPLETED = 4  # MTLCommandBufferStatusCompleted
SPECULATIVE = [
    "cycle_start",
    "draft_graph",
    "draft_ready",
    "verify_graph",
    "verify_ready",
    "greedy_ready",
    "commit_graph",
    "repair_graph",
    "repair_ready",
    "cycle_end",
]
TARGET_STEP = ["cycle_start", "target_step_end"]
GROUPS = {
    "S1 draft to_u32 (wait + verify graph build)": [
        "draft_graph>draft_ready",
        "draft_ready>verify_graph",
    ],
    "S2 verify logits.eval": ["verify_graph>verify_ready"],
    "S3 chat_greedy_ids (+ commit, repair graph)": [
        "verify_ready>greedy_ready",
        "greedy_ready>commit_graph",
        "commit_graph>repair_graph",
    ],
    "S4 eval_cache (+ inter-cycle, next draft graph)": [
        "repair_graph>repair_ready",
        "repair_ready>cycle_end",
        "inter_cycle",
        "cycle_start>draft_graph",
    ],
    "target step (width 0)": ["cycle_start>target_step_end"],
}
REL = {  # (input tokens, proposed, accepted, blocks) from REL compare_native, 128 tokens
    ("a-short-fr", 1): (27, 69, 58, 69),
    ("b-code", 1): (43, 70, 57, 70),
    ("a-short-fr", 2): (27, 108, 72, 54),
    ("b-code", 2): (43, 107, 73, 54),
    ("a-short-fr", 3): (27, 138, 81, 46),
    ("b-code", 3): (43, 139, 80, 47),
}


def read_records(path):
    data = Path(path).read_bytes()
    if not data or len(data) % SIZE:
        raise ValueError(f"empty or truncated record file: {len(data)} bytes")
    rows = [
        dict(zip(FIELDS, struct.unpack_from(FORMAT, data, o), strict=True))
        for o in range(0, len(data), SIZE)
    ]
    if any(r["magic"] != 0x31544243 for r in rows):
        raise ValueError("bad magic")
    rows.sort(key=lambda r: r["sequence"])
    if [r["sequence"] for r in rows] != list(range(len(rows))):
        raise ValueError("missing or duplicate sequence numbers")
    bad = [
        r["sequence"]
        for r in rows
        if not all(math.isfinite(r[k]) for k in ("commit", "gpu_start", "gpu_end", "handler"))
        or r["status"] != COMPLETED
        or not (0 < r["gpu_start"] <= r["gpu_end"] <= r["handler"])
    ]
    if bad:
        raise ValueError(f"{len(bad)} unordered, non-finite or failed records, first {bad[:5]}")
    return rows


def split_cycles(marks):
    cycles, current = [], None
    for mark in marks:
        if mark["name"] == "cycle_start":
            current = [mark]
            cycles.append(current)
        elif current is None:
            raise ValueError(f"mark {mark['name']} before the first cycle_start")
        else:
            current.append(mark)
    for cycle in cycles:
        names = [m["name"] for m in cycle]
        if names not in (SPECULATIVE, TARGET_STEP):
            raise ValueError(f"unexpected mark sequence {names}")
    return cycles


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def overlap(intervals, a, b):
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in intervals)


def analyze_case(case, records):
    marks = case["trace"]["marks"]
    cycles = split_cycles(marks)
    if len(cycles) != len(case["cycles"]) or len(cycles) != sum(map(sum, case["histogram"])):
        raise ValueError("cycle count differs from the histogram")
    for cycle, (width, _) in zip(cycles, case["cycles"], strict=True):
        if (len(cycle) == len(TARGET_STEP)) != (width == 0):
            raise ValueError("mark sequence does not match the cycle width")
    phases = []  # (start s, end s, phase name, cycle index)
    for index, cycle in enumerate(cycles):
        times = [m["uptime_ns"] / 1e9 for m in cycle]
        for i in range(len(cycle) - 1):
            phases.append(
                (times[i], times[i + 1], f"{cycle[i]['name']}>{cycle[i + 1]['name']}", index)
            )
        if index + 1 < len(cycles):
            phases.append(
                (times[-1], cycles[index + 1][0]["uptime_ns"] / 1e9, "inter_cycle", index)
            )
    start, end = phases[0][0], phases[-1][1]
    if any(b < a for a, b, _, _ in phases):
        raise ValueError("non-monotonic marks")
    window = [r for r in records if r["gpu_end"] > start and r["gpu_start"] < end]
    if not window:
        raise ValueError("no command buffer in the case window")
    busy = union([(max(r["gpu_start"], start), min(r["gpu_end"], end)) for r in window])
    gaps, cursor = [], start
    for a, b in busy:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append((cursor, end))

    def phase_at(t):
        for a, b, name, index in phases:
            if a <= t < b:
                return name, index
        return phases[-1][2], phases[-1][3]

    per_phase_idle, per_phase_host, cb_by_phase = {}, {}, {}
    for a, b, name, _ in phases:
        per_phase_host[name] = per_phase_host.get(name, 0.0) + (b - a)
    for a, b in gaps:
        name, _ = phase_at(a)
        per_phase_idle[name] = per_phase_idle.get(name, 0.0) + (b - a)
    for r in window:
        if start <= r["commit"] < end:
            name, _ = phase_at(r["commit"])
            cb_by_phase[name] = cb_by_phase.get(name, 0) + 1
    spec = sum(1 for width, _ in case["cycles"] if width > 0)
    wall = end - start
    gpu = sum(b - a for a, b in busy)
    groups = {}
    for group, names in GROUPS.items():
        idle = sum(per_phase_idle.get(n, 0.0) for n in names)
        denominator = spec if not group.startswith("target") else len(cycles) - spec
        groups[group] = {
            "idle_ms_total": idle * 1e3,
            "idle_ms_per_cycle": idle * 1e3 / denominator if denominator else None,
        }
    unclassified = set(per_phase_idle) - {n for names in GROUPS.values() for n in names}
    if unclassified:
        raise ValueError(f"unclassified idle phases {unclassified}")
    if abs(sum(per_phase_idle.values()) - (wall - gpu)) > 1e-6:
        raise ValueError("idle attribution does not add up")
    return {
        "prompt": case["prompt"],
        "depth": case["depth"],
        "budget": case["budget"],
        "input_tokens": case["input_tokens"],
        "cycles": len(cycles),
        "speculative_cycles": spec,
        "histogram": case["histogram"],
        "proposed": case["proposed"],
        "accepted": case["accepted"],
        "wall_ms": wall * 1e3,
        "gpu_busy_ms": gpu * 1e3,
        "gpu_idle_ms": (wall - gpu) * 1e3,
        "gpu_active_fraction": gpu / wall,
        "wall_ms_per_cycle": wall * 1e3 / len(cycles),
        "idle_ms_per_speculative_cycle": (wall - gpu) * 1e3 / spec if spec else None,
        "command_buffers": len(window),
        "groups": groups,
        "phase_idle_ms_per_speculative_cycle": {
            k: v * 1e3 / max(spec, 1) for k, v in sorted(per_phase_idle.items())
        },
        "phase_host_ms_per_speculative_cycle": {
            k: v * 1e3 / max(spec, 1) for k, v in sorted(per_phase_host.items())
        },
        "command_buffers_by_commit_phase": dict(sorted(cb_by_phase.items())),
    }


def main():
    output, nodylib, runs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    errors, result = [], {"runs": {}, "nodylib": str(nodylib)}
    try:
        reference = json.loads((nodylib / "diag.json").read_text())
        if (nodylib / "cb.bin").exists():
            errors.append("no-dylib run has CB records")
        ref_cases = {(c["prompt"], c["depth"]): c for c in reference["cases"]}
        if len(ref_cases) != 6:
            errors.append(f"expected 6 no-dylib cases, got {len(ref_cases)}")
        for key, case in ref_cases.items():
            rel = REL.get(key)
            got = (case["input_tokens"], case["proposed"], case["accepted"], case["blocks"])
            if rel != got:
                errors.append(f"no-dylib {key}: {got} != REL bridge {rel}")
        for run in runs:
            diag = json.loads((run / "diag.json").read_text())
            records = read_records(run / "cb.bin")
            cases = {(c["prompt"], c["depth"]): c for c in diag["cases"]}
            if cases.keys() != ref_cases.keys():
                errors.append(f"{run.name}: case set differs")
            rows = []
            for key, case in sorted(cases.items()):
                ref = ref_cases.get(key)
                if (
                    ref is None
                    or ref["tokens"] != case["tokens"]
                    or ref["cycles"] != case["cycles"]
                ):
                    errors.append(
                        f"{run.name} {key}: tokens or cycles differ from the no-dylib run"
                    )
                row = analyze_case(case, records)
                row["seconds_with_dylib"] = case["seconds_not_a_benchmark"]
                row["seconds_without_dylib"] = ref["seconds_not_a_benchmark"] if ref else None
                rows.append(row)
            result["runs"][run.name] = {"records": len(records), "cases": rows}
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    output.write_text(json.dumps(result, indent=2))
    for name, run in result["runs"].items():
        for r in run["cases"]:
            groups = "; ".join(
                f"{g.split(' ')[0]} {v['idle_ms_per_cycle']:.3f}"
                for g, v in r["groups"].items()
                if v["idle_ms_per_cycle"] is not None
            )
            print(
                f"{name} {r['prompt']:11s} MTP{r['depth']} cycles {r['cycles']:3d} "
                f"wall/cycle {r['wall_ms_per_cycle']:.3f} ms active {r['gpu_active_fraction']:.3f} "
                f"idle/spec {r['idle_ms_per_speculative_cycle']:.3f} ms | {groups} | "
                f"dylib {r['seconds_with_dylib']:.3f}s vs {r['seconds_without_dylib']:.3f}s"
            )
    print("\n".join(errors) if errors else "all T0b checks passed")
    print("t0b_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
