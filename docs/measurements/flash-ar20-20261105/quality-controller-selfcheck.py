"""Small synthetic checks of the frozen production controller; no main/model runs."""
import copy
import hashlib
import json
import runpy
import tempfile
from pathlib import Path

SOURCE = Path("/Users/justin/.codex/worktrees/flash-ar20-20261105/mix-stq1_0/build/ar20/quality.py")
fingerprint = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
q = runpy.run_path(str(SOURCE))
checks = []


def check(name, action, rejects=False):
    try:
        action()
    except (AssertionError, ValueError, TypeError, KeyError):
        assert rejects, name
    else:
        assert not rejects, name
    checks.append(name)


def fixture(selected):
    bits = sum(1 << i for i, flag in enumerate(q["MACROS"]) if flag in selected)
    rows = []
    for token in [248044, 1]:
        before = {name: 100 for name in [*q["PATH_FLAGS"], "cross_layer_triples", "moe_deferred", "expert_axis_qmv_dispatches", "reused_destinations", "split_records", "split_pieces"]}
        before.update(kernel_bits=bits, readers_in_flight=0, spin_wait=[0, 0, 100, 0], readahead_fds=0, readahead_records=0, slru_layers=0, slru_reservations=0)
        after = copy.deepcopy(before)
        for name, flag in q["PATH_FLAGS"].items():
            after[name] += q["PATH_USES"][name] if flag in selected else 0
        after.update(cross_layer_triples=134, moe_deferred=134, expert_axis_qmv_dispatches=244)
        after["spin_wait"] = [0, 0, 103, 7 if "MLXL3_NATIVE_GPU_SPIN_WAIT" in selected else 0]
        if "MLXL3_NATIVE_EXPERT_READAHEAD_OFF" in selected:
            after.update(readahead_fds=4, readahead_records=12)
        if "MLXL3_NATIVE_CACHE_SLRU" in selected:
            after.update(slru_layers=48, slru_reservations=11)
        rows.append({"token": token, "before": before, "after": after})
    return {**q["COUNTS"], "kernel_bits": bits, "actual_paths": rows}


selected = [flag for flag in q["NEW"] if flag != q["HEAD"]]
check("selection-default11", lambda: q["check_selected"](selected))
for name, value in [("duplicate", selected + selected[:1]), ("unknown", ["MLXL3_NATIVE_UNKNOWN"]), ("scalar", selected[0]), ("boolean", [True])]:
    check("selection-" + name, lambda value=value: q["check_selected"](value), True)
for name, flags in [("reference", []), ("candidate11", selected), ("head-only", [q["HEAD"]]), ("qsa-only", ["MLXL3_NATIVE_QSA_CHAINS"])]:
    check("paths-" + name, lambda flags=flags: q["paths"](fixture(flags), flags))


def changed(name, value, *, before=False, flags=selected):
    evidence = fixture(flags)
    record = evidence["actual_paths"][0]["before" if before else "after"]
    if value is None:
        del record[name]
    else:
        record[name] = value
    return lambda: q["paths"](evidence, flags)


for name, value in [("qsa_chains", None), ("qsa_prefetch", 111), ("head_gates", 101), ("resident_bundle", 135), ("finish_mix", True), ("cross_layer_triples", 133), ("moe_deferred", 133), ("expert_axis_qmv_dispatches", 243), ("reused_destinations", 99), ("split_pieces", -1), ("readers_in_flight", 1), ("kernel_bits", 0), ("spin_wait", [0, 0, 103, 0]), ("readahead_records", 0), ("slru_layers", 0), ("slru_reservations", 0)]:
    check("counter-" + name, changed(name, value), True)
for name, value in [("qsa_chains", 112), ("spin_wait", [1, 0, 103, 0]), ("readahead_fds", 1), ("slru_reservations", 1)]:
    check("off-activation-" + name, changed(name, value, flags=[]), True)
for name, value in [("case_count", 0), ("compared_state_words", True), ("all_readers_joined", False)]:
    evidence = {**q["COUNTS"], name: value}
    check("count-" + name, lambda evidence=evidence: q["OLD"]["exact_fields"](evidence, q["COUNTS"]), True)
with tempfile.TemporaryDirectory(prefix="ar20-quality-json-") as folder:
    path = Path(folder) / "oracle.json"
    for name, text, rejects in [("valid", '{"cases":[1,2],"parity":false}', False), ("duplicate", '{"schema":1,"schema":1}', True), ("nested-duplicate", '{"cases":[{"token":1,"token":2}]}', True), ("nan", '{"value":NaN}', True), ("infinite", '{"value":1e999}', True), ("incomplete", '{"cases":[', True)]:
        path.write_text(text)
        check("json-" + name, lambda: q["read_json"](path), rejects)
marker = "AR20_MODEL_COMPARE "
check("marker-duplicate-lines", lambda: q["OLD"]["marker"](marker + '{}\n' + marker + '{}', marker.strip()), True)
# Retain a concrete accepted malformed marker; parent must approve any frozen-source fix.
ambiguous = marker + '{"parity":false,"parity":true}'
assert q["OLD"]["marker"](ambiguous, marker.strip())["parity"] is True
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == fingerprint
print(json.dumps({"status": "selfchecks_passed_with_audit_finding", "checks": len(checks), "names": checks, "controller_sha256": fingerprint, "finding": "OLD marker accepts duplicate object keys (last wins); strict read_json rejects them", "scope": "deterministic synthetic Python checks only; no model, subprocess or formal proof"}, indent=2))
