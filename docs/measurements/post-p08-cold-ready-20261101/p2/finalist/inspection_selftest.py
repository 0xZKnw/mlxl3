"""Reuse the P2 failure fixtures for the fixed P/U/D ablation schedule."""

import contextlib
import copy
import json
import runpy
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    shared = runpy.run_path(str(BASE.parent / "inspection_selftest.py"))
    adapter = runpy.run_path(str(BASE / "inspection.py"))
    shared["fixture"].__globals__.update(adapter=adapter, BASE=BASE)
    execute = shared["execute"]
    original_fixture = shared["fixture"]

    def fixture(*args):
        data = original_fixture(*args)
        _, manifest, identity, plan, _, _, _, _ = data
        context = args[1]
        amendment = json.loads((BASE.parent / "ablation/swap64-amendment.json").read_text())
        amendment.update(
            anchor_path=manifest["swap_anchor_path"],
            anchor_sha256=manifest["swap_anchor_sha256"],
            previous_policy_sha256=context["file_sha"](Path(manifest["policy_amendment"])),
        )
        path = Path(args[0]) / "swap64-amendment.json"
        path.write_text(json.dumps(amendment))
        manifest["swapout_amendment"] = str(path)
        for document in (manifest, identity, plan[0]):
            document["maximum_new_swapout_bytes"] = 64 << 20
        for source in (
            path,
            BASE.parent / "ablation/gpu-runner.py",
            BASE.parent / "ablation/compose-inspection.py",
            BASE.parent.parent / "diagnostic-swapin-amendment.json",
        ):
            identity["sources"][str(source)] = context["file_sha"](source)
        return data

    execute.__globals__["fixture"] = fixture
    replace = shared["old"]["replace"]
    changes = [
        lambda v: v["manifest"]["plans"].reverse(),
        replace("report", ["environment", "MLXL3_SSD_COLD_READY"], "0"),
        replace("report", ["binary_sha256"], "0" * 64),
        replace("supervisor", ["commands", 0, "maximum_live_swapin_bytes"], (1 << 30) + 1),
    ]
    cases = [
        (mode, i, change)
        for mode in ("MTP", "DF")
        for i, change in ([(j, None) for j in range(1, 7)] + [(2, change) for change in changes])
    ]
    with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
        for i, (mode, position, change) in enumerate(cases):
            context = adapter["configured"](mode)
            error, _ = execute(context, Path(folder) / str(i), mode, 128, position, change)
            assert (error is not None) == (change is not None), (i, error)
    policy = {"swap_anchor_vm_bytes": {"Swapins": 100, "Swapouts": 200}}
    process = {
        "before": {"vm_bytes": {"Swapins": 100 + (768 << 20), "Swapouts": 200 + (32 << 20)}},
        "after": {"vm_bytes": {"Swapins": 100 + (1 << 30), "Swapouts": 200 + (64 << 20)}},
        "swapins_bytes": 256 << 20,
        "swapouts_bytes": 32 << 20,
        "maximum_live_swapin_bytes": 1 << 30,
        "maximum_live_swapout_bytes": 64 << 20,
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
        "12 positive finalist inspections, 8 negative cases rejected; swap boundary + 2 overflows checked"
    )


if __name__ == "__main__":
    main()
