"""Read-only attribution of interrupted r2; sample weights are not durations."""

import bisect
import collections
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

p = Path(sys.argv[1])
sys.argv = ["analysis", str(p)]
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(str(p / "analyze-partial.py"))
ev, n = ns["ev"], ns["n"]
by = collections.defaultdict(list)
for item in ns["groups"]["host_async_submit"]:
    by[item[1]["parent"]].append(item)
assert len(by) == n * 48 and all(len(items) == 2 for items in by.values())
role_spans = collections.defaultdict(list)
for parent, items in by.items():
    assert ev[parent]["name"] == "gpu_resident_hot_prepare"
    for role, (_, _, a, b) in zip(
        ("resident_slot_indices_async_eval", "resident_hot_output_async_eval"),
        sorted(items, key=lambda item: item[2]),
        strict=True,
    ):
        role_spans[role].append((a, b))
roles = {role: ns["Measure"](spans) for role, spans in role_spans.items()}
eligible = ns["Measure"](sum(role_spans.values(), []))


def contains(measure, timestamp):
    return measure.overlap(timestamp, timestamp + 1) == 1


def exposure(spans, offset=0):
    return sum(
        ns["cpus"]["Running"].overlap(a + offset, b + offset)
        - ns["cpugpu"]["Running"].overlap(a + offset, b + offset)
        for a, b in spans
    )


exact = {}
halfwidth = ns["result"]["clock_fit_halfwidth_ns"]
for role, spans in role_spans.items():
    exact[role] = {
        "calls": len(spans),
        "wall_ms_per_output": sum(b - a for a, b in spans) / n / 1e6,
        "cpu_running_gpu_idle_ms_per_output": exposure(spans) / n / 1e6,
        "clock_fit_endpoint_sensitivity_ms_per_output": {
            str(offset): exposure(spans, offset) / n / 1e6
            for offset in (-halfwidth, 0, halfwidth)
        },
    }

reports = {}
for label, profile_name, raw_name in (
    ("high", "time-profile.xml", "time-sample-100us-user-kernel.xml"),
    ("standard", "time-profile-standard-user.xml", "time-sample-1000us-user.xml"),
):
    raw, resolve = ns["table"](p / raw_name)
    raw_types = collections.defaultdict(list)
    all_raw_types = collections.Counter()
    for row in raw:
        thread = resolve(row["thread"])
        process = next(resolve(child) for child in thread if child.tag == "process")
        if ns["identity"](process, resolve, "pid") != ns["pid"]:
            continue
        tid = ns["identity"](thread, resolve, "tid")
        timestamp = ns["number"](row["time"])
        kind = (row["sample-type"].text or "", row["sample-type"].get("fmt", ""))
        raw_types[timestamp, tid].append(kind)
        all_raw_types[str(kind)] += 1
    del raw, resolve
    rows, resolve = ns["table"](p / profile_name)
    leaf = collections.Counter()
    near = collections.Counter()
    user_leaf = collections.Counter()
    mapping = collections.Counter()
    weights = collections.Counter()
    selected_kinds = collections.Counter()
    eligible_kinds = collections.Counter()
    role_counts = collections.Counter()
    other_threads = {}
    binaries = {}
    stacks = []
    main_samples = 0
    for row in rows:
        if ns["identity"](row["process"], resolve, "pid") != ns["pid"]:
            continue
        tid = ns["identity"](row["thread"], resolve, "tid")
        timestamp = ns["number"](row["time"])
        if not ns["lo"] <= timestamp < ns["hi"]:
            continue
        frames = [resolve(frame) for frame in row["stack"] if frame.tag == "frame"]
        names = [frame.get("name", "<unknown>") for frame in frames]
        assert names
        for frame in frames:
            for binary in frame:
                if binary.tag == "binary":
                    item = resolve(binary)
                    if item.get("name") in ("prototype-engine", "libmlx.dylib"):
                        binaries[item.get("name"), item.get("UUID")] = dict(item.attrib)
        kind = raw_types.get((timestamp, tid), [])
        kind_label = str(kind) if kind else "no_exact_raw_time_tid_match"
        nearest = next(
            (name for name in names if name.startswith(("mlx::", "std::__1::__function::__func<mlx::"))),
            "<no MLX frame>",
        )
        weight = ns["number"](row["weight"])
        assert weight > 0
        if tid != ns["threads"][0]:
            item = other_threads.setdefault(
                tid,
                {"name": ns["names"].get(tid, row["thread"].get("fmt", "")),
                 "samples": 0, "weight_ns": 0, "leaf": collections.Counter(),
                 "nearest_mlx": collections.Counter(), "raw_kinds": collections.Counter()},
            )
            item["samples"] += 1
            item["weight_ns"] += weight
            item["leaf"][names[0]] += 1
            item["nearest_mlx"][nearest] += 1
            item["raw_kinds"][kind_label] += 1
            continue
        main_samples += 1
        selected_kinds[kind_label] += 1
        weights[weight] += 1
        original = timestamp
        if not contains(ns["cpus"]["Running"], timestamp):
            irq = ns["cpus"]["Interrupted"]
            index = bisect.bisect_right(irq.s, timestamp) - 1
            if index < 0 or not irq.x[index][0] <= timestamp < irq.x[index][1]:
                mapping["not_running_or_interrupt"] += 1
                continue
            timestamp = irq.x[index][0] - 1
            if not contains(ns["cpus"]["Running"], timestamp):
                mapping["interrupt_without_running_predecessor"] += 1
                continue
            mapping["running_predecessor_of_interrupt_derived"] += 1
        else:
            mapping["direct_running"] += 1
        if not contains(eligible, timestamp) or contains(ns["gall"], timestamp):
            continue
        which = [role for role, measure in roles.items() if contains(measure, timestamp)]
        assert len(which) == 1
        role = which[0]
        role_counts[role] += 1
        eligible_kinds[kind_label] += 1
        leaf[names[0]] += 1
        near[nearest] += 1
        first_user = "<no user frame>"
        for frame in frames:
            binary = next((resolve(item) for item in frame if item.tag == "binary"), None)
            if binary is not None and not (
                binary.get("path", "").startswith("/System/Library/Kernels/")
                or binary.get("name", "").startswith("com.apple.")
            ):
                first_user = frame.get("name", "<unknown>")
                break
        user_leaf[first_user] += 1
        stacks.append({"time_ns": timestamp, "sample_time_ns": original, "tid": tid,
                       "role": role, "weight_ns": weight, "raw_kinds": kind,
                       "mapping": "direct" if timestamp == original else "interrupt_predecessor",
                       "frames": names})
    assert stacks and sum(role_counts.values()) == len(stacks)
    thread_reports = []
    for tid, item in sorted(other_threads.items(), key=lambda pair: -pair[1]["samples"]):
        thread_reports.append({**item, "tid": tid,
                               "leaf": item["leaf"].most_common(10),
                               "nearest_mlx": item["nearest_mlx"].most_common(10),
                               "raw_kinds": dict(item["raw_kinds"])})
    report = {
        "status": "sampled_partial_observer_biased_diagnostic_not_release",
        "full_run_parity": False, "series": label, "inference_tid": ns["threads"][0],
        "steps": [ns["steps"][0]["data"]["step"], ns["steps"][-1]["data"]["step"]],
        "complete_outputs": n, "exact_role_intervals": exact,
        "all_raw_types_for_target_process": dict(all_raw_types),
        "main_samples_in_selection": main_samples,
        "main_selected_raw_types": dict(selected_kinds),
        "main_weight_histogram_ns": dict(weights), "mapping_counts": dict(mapping),
        "eligible_samples": len(stacks), "samples_by_role": dict(role_counts),
        "eligible_raw_types": dict(eligible_kinds),
        "top_leaf_samples": leaf.most_common(30), "nearest_mlx_frame_samples": near.most_common(30),
        "first_user_frame_samples": user_leaf.most_common(30),
        "other_engine_threads_separate": thread_reports,
        "resolved_engine_mlx_binaries": list(binaries.values()),
        "limits": ["Weights and frame counts are statistical sample evidence, not exact exclusive CPU durations.",
                   "HF/LF are separate providers; never add their samples or XML references.",
                   "Derived interrupted points are labeled separately from direct Running points.",
                   "Other-thread frame presence does not prove it extends the main-thread critical path.",
                   "Interrupted workflow and cfg(test) stage retention prevent release-app performance attribution."]}
    (p / ("async-" + label + "-stacks.json")).write_text(json.dumps(stacks, indent=2) + "\n")
    (p / ("async-" + label + "-report.json")).write_text(json.dumps(report, indent=2) + "\n")
    reports[label] = {"eligible_samples": len(stacks), "samples_by_role": dict(role_counts),
                      "main_samples": main_samples, "raw_types": dict(eligible_kinds),
                      "nearest_mlx": near.most_common(8)}
    del rows, resolve
print(json.dumps({"exact": exact, "sample_reports": reports}, indent=2))
