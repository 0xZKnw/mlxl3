"""P09 interval attribution; XML/interval helpers retained from P08 analyze-partial.py."""

import bisect
import collections
import hashlib
import json
import sys
import tempfile
import xml.etree.ElementTree as E
from decimal import Decimal
from pathlib import Path

q = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and sys.argv[1] != "--self-check"
    else Path("docs/measurements/ultra-profile-20261027/attribution/AR-MAIN")
)


def table(path):
    root = E.parse(path).getroot()
    ids = {}
    for e in root.iter():
        if e.get("id"):
            assert e.get("id") not in ids and not e.get("ref"), "duplicate/ref definition"
            ids[e.get("id")] = e

    def v(e):
        if e.get("ref"):
            assert e.get("ref") in ids, "unknown XML reference"
            a = ids[e.get("ref")]
            assert a.tag == e.tag, "reference type mismatch"
            return a
        return e

    schemas = root.findall(".//schema")
    assert len(schemas) == 1, "export must contain exactly one schema"
    schema = schemas[0]
    cols = [e.findtext("mnemonic") for e in schema.findall("col")]
    assert len(set(cols)) == len(cols)
    rows = []
    for e in root.findall(".//row"):
        assert len(e) == len(cols)
        rows.append({k: v(ei) for k, ei in zip(cols, e)})
    assert rows, "empty table"
    return rows, v


def text(e):
    return e.text or ""


def number(e):
    assert e.tag != "sentinel"
    n = int(text(e))
    assert n >= 0
    return n


def identity(e, v, kind):
    if e.tag == "sentinel":
        return None
    n = next((v(x) for x in e if x.tag == kind), None)
    assert n is not None, ("missing identity", kind, e.tag)
    return number(n)


def intersect(xs, ys):
    a = merge(xs)
    b = merge(ys)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def merge(xs):
    out = []
    for a, b in sorted(xs):
        assert type(a) is int and type(b) is int and 0 <= a <= b
        if a == b:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


class Measure:
    def __init__(self, xs):
        self.x = merge(xs)
        self.s = [a for a, b in self.x]
        self.prefix = [0]
        for a, b in self.x:
            self.prefix.append(self.prefix[-1] + b - a)

    def until(self, t):
        i = bisect.bisect_right(self.s, t) - 1
        return 0 if i < 0 else self.prefix[i] + min(t - self.x[i][0], self.x[i][1] - self.x[i][0])

    def overlap(self, a, b):
        assert a <= b
        return self.until(b) - self.until(a)


for a in range(5):
    for b in range(a, 5):
        for c in range(5):
            for d in range(c, 5):
                m = Measure([(a, b), (c, d)])
                oracle = set(range(a, b)) | set(range(c, d))
                for x in range(5):
                    for y in range(x, 5):
                        assert m.overlap(x, y) == len(oracle & set(range(x, y)))
                        assert Measure(intersect([(a, b)], [(c, d)])).overlap(x, y) == len(
                            set(range(a, b)) & set(range(c, d)) & set(range(x, y))
                        )
for bad in [[(-1, 2)], [(2, 1)], [(False, 2)]]:
    try:
        merge(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid interval accepted")

if "--self-check" in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "table.xml"
        good = '<trace-query-result><node><schema name="test"><col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col></schema><row><start-time id="1">2</start-time><duration id="2">3</duration></row><row><start-time ref="1"/><duration ref="2"/></row></node></trace-query-result>'
        f.write_text(good)
        r, v = table(f)
        assert len(r) == 2 and number(r[1]["start"]) == 2 and number(r[1]["duration"]) == 3
        for bad in [
            good.replace('duration id="2"', 'duration id="1"'),
            good.replace('duration ref="2"', 'duration ref="missing"'),
            good.replace('duration ref="2"', 'duration ref="1"'),
            good.replace('<duration ref="2"/>', ""),
            good.replace("<row>", "<notrow>").replace("</row>", "</notrow>"),
        ]:
            f.write_text(bad)
            try:
                table(f)
            except (AssertionError, E.ParseError):
                pass
            else:
                raise AssertionError("invalid XML accepted")
    print(
        "3375 union +3375 intersection finite lattice checks; 3 invalid interval cases; valid XML references and5 corrupt/empty XML cases passed"
    )
    sys.exit(0)

assert not (q / "attribution.json").exists(), "refusing to overwrite attribution"
capture = json.loads((q / "capture.json").read_text())
assert capture["status"] == "native_validated_captures_pending_inspection"
assert capture["parity"] is True and capture["target_reaped"] is True
pid = capture["target_pid"]
gpu, v = table(q / "metal-gpu-intervals.xml")
gm, ga, unknown = [], [], []
for row in gpu:
    assert text(row["state"]) == "Active"
    a = number(row["start"])
    interval = (a, a + number(row["duration"]))
    owner = identity(row["process"], v, "pid")
    ga.append(interval)
    if owner == pid:
        gm.append(interval)
    if owner is None:
        unknown.append(interval)
assert gm, "no model GPU intervals"
gmodel, gall, gunknown = Measure(gm), Measure(ga), Measure(unknown)
del gpu, v
rows, v = table(q / "thread-state.xml")
states = collections.defaultdict(list)
names = set()
for row in rows:
    if identity(row["process"], v, "pid") != pid:
        continue
    if "qwen4::profile::flash_detailed" not in row["thread"].get("fmt", ""):
        continue
    names.add(identity(row["thread"], v, "tid"))
    a = number(row["start"])
    states[text(row["state"])].append((a, a + number(row["duration"])))
assert len(names) == 1 and states, "missing or ambiguous inference thread"
cpus = {key: Measure(xs) for key, xs in states.items()}
cpugpu = {key: Measure(intersect(m.x, gall.x)) for key, m in cpus.items()}
del rows, v
rows, v = table(q / "syscall.xml")
pread = []
for row in rows:
    if identity(row["process"], v, "pid") != pid or text(row["call"]) != "BSC_sys_preadv":
        continue
    a = number(row["start"])
    assert number(row["errno"]) == 0
    pread.append({"start": a, "end": a + number(row["duration"]), "bytes": number(row["return"])})
assert pread, "no POSIX expert reads"
del rows, v
ti, v = table(q / "time-info.xml")
assert len(ti) == 1
origin = int(Decimal(text(ti[0]["trace-start-time"])) * 1_000_000_000)
s = json.loads((q / "profile/trace_1-summary.json").read_text())
ev = json.loads((q / "profile/trace_1-decode.json").read_text())
assert s["full_logits_target_head_states_byte_exact"] is True
assert len(s["tokens"]) == 128 and s["logit_count"] == 248320
assert ev[0]["name"] == "decode"
shift = int(Decimal(str(s["decode_epoch"])) * 1_000_000_000) - origin - ev[0]["start_ns"]


def span(event):
    return event["start_ns"] + shift, event["end_ns"] + shift


first = max(min(a for a, _ in gm), min(c["start"] for c in pread))
last = max(b for xs in states.values() for _, b in xs)


def select_steps():
    return [
        e
        for e in ev
        if e["name"] == "output_step"
        and e["data"]["step"] >= 16
        and span(e)[0] >= first
        and span(e)[1] <= last
    ][:40]


steps = select_steps()
assert len(steps) >= 16, "fewer than 16 complete transitions"
n = len(steps)
lo, hi = span(steps[0])[0], span(steps[-1])[1]
batches = [
    e
    for e in ev
    if e["name"] in ("disk_read_batch", "cold_ready_batch") and lo <= span(e)[0] < span(e)[1] <= hi
]
assert batches, "no complete expert batches"


def matched_calls(event):
    a, b = span(event)
    calls = [c for c in pread if a - 30000 <= c["start"] < b and a < c["end"] <= b + 30000]
    data = event["data"]
    cold = event["name"] == "cold_ready_batch"
    assert data["io_ok" if cold else "ok"] is True
    jobs = data["jobs_mapping" if cold else "jobs"]
    assert jobs and len(calls) == len(jobs), (event["name"], a, b, len(calls), len(jobs))
    if cold:
        assert data["status"] == "finished" and data["jobs"] == len(jobs)
    assert all(c["bytes"] == 1248012 for c in calls), "expert syscall byte count"
    assert len({(c["start"], c["end"]) for c in calls}) == len(calls)
    return calls


fit_low, fit_high = -(10**9), 10**9
calibration = [e for e in batches if e["name"] == "disk_read_batch"]
assert calibration, "no synchronous batch for independent clock calibration"
for event in calibration:
    a, b = span(event)
    calls = matched_calls(event)
    fit_low = max(fit_low, max(c["end"] - b for c in calls))
    fit_high = min(fit_high, min(c["start"] - a for c in calls))
assert fit_low <= fit_high, "inconsistent clock containment"
correction = (fit_low + fit_high) // 2
shift += correction
lo += correction
hi += correction
assert first <= lo < hi <= last and select_steps() == steps, "clock fit changed window"


def summarize(intervals):
    intervals = merge(intervals)
    total = sum(b - a for a, b in intervals)
    out = {"wall_ns": total}
    for key, meter in [
        ("model_gpu_active_ns", gmodel),
        ("any_gpu_active_ns", gall),
        ("unknown_owner_gpu_ns", gunknown),
    ]:
        out[key] = sum(meter.overlap(a, b) for a, b in intervals)
    for state, meter in cpus.items():
        out["cpu_" + state + "_ns"] = sum(meter.overlap(a, b) for a, b in intervals)
        out["cpu_" + state + "_gpu_idle_ns"] = sum(
            meter.overlap(a, b) - cpugpu[state].overlap(a, b) for a, b in intervals
        )
    residual = total - sum(out["cpu_" + state + "_ns"] for state in cpus)
    assert 0 <= residual <= len(intervals) * 100 + 100, "CPU state partition incomplete"
    out["cpu_partition_residual_ns"] = residual
    out["per_token_ms"] = {key: value / n / 1e6 for key, value in out.items()}
    return out


groups, exclusive = collections.defaultdict(list), collections.defaultdict(list)
children = collections.defaultdict(list)
for i, event in enumerate(ev):
    parent = event["parent"]
    assert parent is None or type(parent) is int and 0 <= parent < i
    a, b = span(event)
    assert a <= b
    if parent is not None:
        pa, pb = span(ev[parent])
        assert pa <= a <= b <= pb, "child outside parent"
        children[parent].append((a, b))
for i, event in enumerate(ev):
    a, b = span(event)
    a, b = max(a, lo), min(b, hi)
    if a >= b:
        continue
    groups[event["name"]].append((a, b))
    cursor = a
    for ca, cb in intersect(children[i], [(a, b)]):
        if cursor < ca:
            exclusive[event["name"]].append((cursor, ca))
        cursor = cb
    if cursor < b:
        exclusive[event["name"]].append((cursor, b))
exclusive_total = sum(sum(b - a for a, b in merge(xs)) for xs in exclusive.values())
assert exclusive_total == hi - lo, "exclusive scopes do not partition selected window"

io = []
seen_calls = set()
for event in batches:
    calls = matched_calls(event)
    keys = {(c["start"], c["end"]) for c in calls}
    assert not keys & seen_calls, "syscall assigned twice"
    seen_calls.update(keys)
    intervals = [(c["start"], c["end"]) for c in calls]
    io.append(
        {
            "operation": event["name"],
            "start_ns": span(event)[0],
            "end_ns": span(event)[1],
            "jobs": len(calls),
            "bytes": sum(c["bytes"] for c in calls),
            "intervals": intervals,
        }
    )
all_calls = [c for row in io for c in row["intervals"]]
selected_calls = {(c["start"], c["end"]) for c in pread if lo <= c["start"] < c["end"] <= hi}
assert seen_calls == selected_calls, "unmatched syscall in selected window"
waits = groups["cold_ready_wait"] + groups["cold_ready_finish"] + groups["disk_read_batch"]
io_summary = summarize(all_calls)
io_summary["jobs"] = sum(row["jobs"] for row in io)
io_summary["bytes"] = sum(row["bytes"] for row in io)
io_summary["batch_types"] = dict(collections.Counter(row["operation"] for row in io))
io_summary["read_and_host_wait"] = summarize(intersect(all_calls, waits))
result = {
    "status": "completed_interval_attribution_of_selected_diagnostic_window",
    "full_run_parity": True,
    "pid": pid,
    "inference_tid": next(iter(names)),
    "complete_steps": n,
    "step_first": steps[0]["data"]["step"],
    "step_last": steps[-1]["data"]["step"],
    "selection_policy": "first up to40 complete steps>=16 inside coverage; minimum16; stable after clock fit",
    "selected_start_ns": lo,
    "selected_end_ns": hi,
    "clock_correction_ns": correction,
    "clock_fit_halfwidth_ns": (fit_high - fit_low + 1) // 2,
    "clock_calibration_batches": len(calibration),
    "decode_selected": summarize([(lo, hi)]),
    "output_steps_only": summarize([span(e) for e in steps]),
    "components_inclusive": {key: summarize(xs) for key, xs in groups.items()},
    "components_exclusive": {key: summarize(xs) for key, xs in exclusive.items()},
    "io": io_summary,
    "limits": [
        "CPU and GPU axes overlap; component-inclusive spans overlap.",
        "GPU activity is command execution, not ALU utilization; other/unknown owners reported.",
        "POSIX reads may hit OS cache; no physical NAND attribution.",
        "Read intersect host wait and GPU idle is exposed time, not a causal SSD saving guarantee.",
        "Cold W1 depends on a subset of reads; no independent-GPU tail bound reused.",
        "Diagnostic overhead must be compared against before/after controls; not a release speedup.",
    ],
}
stack_paths = [q / "time-profile.xml", q / "time-sample-standard.xml"]
assert all(p.exists() for p in stack_paths) or not any(p.exists() for p in stack_paths), (
    "incomplete sampling exports"
)
if all(p.exists() for p in stack_paths):
    raw, resolve = table(stack_paths[1])
    kinds = collections.defaultdict(list)
    for row in raw:
        thread = resolve(row["thread"])
        process = next(resolve(c) for c in thread if c.tag == "process")
        if identity(process, resolve, "pid") == pid:
            kinds[number(row["time"]), identity(thread, resolve, "tid")].append(
                text(row["sample-type"])
            )
    del raw, resolve
    rows, resolve = table(stack_paths[0])
    count, leaf, near, scopes, mapping, raw_kinds = (collections.Counter() for _ in range(6))
    saved = []
    active = cpus.get("Running", Measure([]))
    interrupted = cpus.get("Interrupted", Measure([]))
    exclusive_meters = {key: Measure(xs) for key, xs in exclusive.items()}
    for row in rows:
        if identity(row["process"], resolve, "pid") != pid or identity(
            row["thread"], resolve, "tid"
        ) != next(iter(names)):
            continue
        original = timestamp = number(row["time"])
        if not lo <= timestamp < hi:
            continue
        count["main_samples_in_window"] += 1
        if active.overlap(timestamp, timestamp + 1) == 0:
            index = bisect.bisect_right(interrupted.s, timestamp) - 1
            if index < 0 or not interrupted.x[index][0] <= timestamp < interrupted.x[index][1]:
                mapping["not_running_or_interrupted"] += 1
                continue
            timestamp = interrupted.x[index][0] - 1
            if active.overlap(timestamp, timestamp + 1) == 0:
                mapping["interrupt_without_running_predecessor"] += 1
                continue
            mapping["running_predecessor_of_interrupt_derived"] += 1
        else:
            mapping["direct_running"] += 1
        if not lo <= timestamp < hi or gall.overlap(timestamp, timestamp + 1):
            continue
        frames = [
            resolve(frame).get("name", "<unknown>")
            for frame in row["stack"]
            if frame.tag == "frame"
        ]
        if not frames:
            count["eligible_missing_backtrace"] += 1
            frames = ["<no captured backtrace>"]
        count["eligible_cpu_running_gpu_idle_samples"] += 1
        scope = [
            key
            for key, meter in exclusive_meters.items()
            if meter.overlap(timestamp, timestamp + 1)
        ]
        assert len(scope) == 1, "sample not in exactly one exclusive scope"
        kind = kinds.get((original, next(iter(names))), [])
        raw_kinds[str(kind) if kind else "no_exact_raw_match"] += 1
        scopes[scope[0]] += 1
        leaf[frames[0]] += 1
        nearest = next(
            (
                name
                for name in frames
                if name.startswith(("mlx::", "std::__1::__function::__func<mlx::"))
            ),
            "<no MLX frame>",
        )
        near[nearest] += 1
        saved.append(
            {
                "sample_ns": original,
                "mapped_ns": timestamp,
                "scope": scope[0],
                "raw_kinds": kind,
                "frames": frames,
            }
        )
    result["cpu_idle_sampling"] = {
        "counts": dict(count),
        "mapping": dict(mapping),
        "raw_kinds": dict(raw_kinds),
        "exclusive_scope_counts": scopes.most_common(),
        "leaf_counts": leaf.most_common(30),
        "nearest_mlx_counts": near.most_common(30),
        "samples": saved,
        "limits": "Statistical frame counts, not exact CPU milliseconds; interrupt predecessors are derived and explicit.",
    }
    del rows, resolve
else:
    result["cpu_idle_sampling"] = {"status": "not_measured_no_sampling_exports"}
inputs = [
    Path(__file__),
    *(
        q / name
        for name in (
            "capture.json",
            "metal-gpu-intervals.xml",
            "thread-state.xml",
            "syscall.xml",
            "time-info.xml",
            "profile/trace_1-summary.json",
            "profile/trace_1-decode.json",
        )
    ),
]
inputs.extend(p for p in stack_paths if p.exists())
result["input_sha256"] = {}
for path in inputs:
    with path.open("rb") as stream:
        result["input_sha256"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
(q / "io-batches.json").write_text(json.dumps(io) + "\n")
(q / "attribution.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            key: result[key]
            for key in [
                "status",
                "complete_steps",
                "clock_fit_halfwidth_ns",
                "decode_selected",
                "io",
            ]
        },
        indent=2,
    )
)
