"""Exercise P2's actual inspector using disposable inherited bridge fixtures."""

import contextlib
import copy
import json
import runpy
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
adapter = runpy.run_path(str(BASE / "inspection.py"))
old = runpy.run_path(str(BASE.parent / "inspection_selftest.py"))
execute = old["execute"]
original_fixture = old["fixture"]


def fixture(output, context, mode, budget, position):
    data = original_fixture(output, context, mode, budget, 2)
    entry, manifest, identity, plan, report, supervisor, _, _ = data
    order = context["ORDER"]["pilot128"]
    entry.update(arm=order[position - 1], position_within_mode=position)
    entry["cold_ready_flag"] = adapter["FLAGS"][entry["arm"]]
    manifest["plans"] = []
    for i, arm in enumerate(order, 1):
        row = copy.deepcopy(entry)
        row.update(arm=arm, position_within_mode=i, cold_ready_flag=adapter["FLAGS"][arm])
        if i != position:
            row["name"] += str(i)
        manifest["plans"].append(row)
    for env in (plan[0]["env"], report["environment"], supervisor["commands"][0]["env"]):
        env["MLXL3_SSD_COLD_READY"] = entry["cold_ready_flag"]
    for document in (manifest, identity):
        document["maximum_new_swapin_bytes"] = 1 << 30
    identity["sources"][str(BASE / "inspection.py")] = context["file_sha"](BASE / "inspection.py")
    return data


def main():
    execute.__globals__["fixture"] = fixture
    cases = [
        (mode, position, None)
        for mode in ("AR", "MTP", "DF")
        for position in range(1, 7 if mode == "AR" else 5)
    ]
    negative = [
        lambda v: v["manifest"]["plans"].reverse(),
        old["replace"]("report", ["environment", "MLXL3_SSD_COLD_READY"], "invalid"),
        old["replace"]("report", ["runs", 0, "token_hash"], "invalid"),
        old["replace"]("report", ["reader_joined"], False),
        old["replace"]("report", ["runs"], []),
        old["replace"]("manifest", ["maximum_new_swapin_bytes"], (1 << 30) + 1),
        old["replace"]("supervisor", ["commands", 0, "maximum_live_swapin_bytes"], (1 << 30) + 1),
        old["replace"]("report", ["binary_sha256"], "0" * 64),
    ]
    cases.extend(("AR", 3, mutation) for mutation in negative)
    with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
        for i, (mode, position, mutation) in enumerate(cases):
            context = adapter["configured"](mode)
            error, _ = execute(context, Path(temporary) / str(i), mode, 128, position, mutation)
            assert (error is not None) == (mutation is not None), (i, error)
    policy = {"swap_anchor_vm_bytes": {"Swapins": 100, "Swapouts": 200}}
    process = {
        "before": {"vm_bytes": {"Swapins": 100 + (1 << 30) - (256 << 20), "Swapouts": 200}},
        "after": {"vm_bytes": {"Swapins": 100 + (1 << 30), "Swapouts": 200 + (32 << 20)}},
        "swapins_bytes": 256 << 20,
        "swapouts_bytes": 32 << 20,
        "maximum_live_swapin_bytes": 1 << 30,
        "maximum_live_swapout_bytes": 32 << 20,
    }
    adapter["check_swap"](process, policy)
    for counter in ("Swapins", "Swapouts"):
        bad = copy.deepcopy(process)
        bad["after"]["vm_bytes"][counter] += 1
        bad[counter.lower() + "_bytes"] += 1
        try:
            adapter["check_swap"](bad, policy)
        except AssertionError:
            pass
        else:
            raise AssertionError("swap overflow accepted")
    print(
        json.dumps(
            {
                "positive_inspections": 14,
                "negative_inspections": len(negative),
                "swap_boundary_positive": 1,
                "swap_overflow_rejected": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
