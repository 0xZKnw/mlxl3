"""Ablation reuses Cold's complete inspector with preregistered orders and frozen per-arm binaries."""

import json
import runpy
from pathlib import Path

BASE = Path(__file__).resolve().parent
FLAGS = {"P": "1", "U": "1", "D": "1"}


def check_swap(process, policy):
    result = {}
    for counter, cap in (("Swapouts", 64 << 20), ("Swapins", 1 << 30)):
        delta = process[counter.lower() + "_bytes"]
        maximum = process["maximum_live_" + counter.lower()[:-1] + "_bytes"]
        before, after = (process[moment]["vm_bytes"][counter] for moment in ("before", "after"))
        start = policy["swap_anchor_vm_bytes"][counter]
        assert all(type(value) is int for value in (delta, maximum, before, after, start))
        assert 0 <= start <= before <= after and after - before == delta, "VM counters"
        assert 0 <= delta <= (32 << 20 if counter == "Swapouts" else 256 << 20), "per-process swap"
        assert 0 <= maximum <= after - start <= cap, "cumulative swap"
        result.update(
            {
                "cumulative_" + counter.lower() + "_before_bytes": before - start,
                "cumulative_" + counter.lower() + "_after_bytes": after - start,
                "maximum_cumulative_" + counter.lower() + "_bytes": max(maximum, after - start),
            }
        )
    return result


def configured(mode):
    assert mode == "AR"
    context = runpy.run_path(str(BASE.parent.parent / "benchmark-inspection.py"))
    order = {"pilot128": list("PUDDUP")}
    context["preflight"].__globals__["COMPOSE"] = BASE / "compose-inspection.py"

    def amended_policy(manifest, identity, frozen):
        runner = runpy.run_path(str(BASE / "gpu-runner.py"))
        original_policy = context["swap_policy"]
        # The inherited immutable policy still validates the original anchor and provenance.
        documents = [
            dict(document, maximum_new_swapout_bytes=32 << 20) for document in (manifest, identity)
        ]
        policy = original_policy(*documents, frozen)
        for document in (manifest, identity):
            assert type(document["maximum_new_swapout_bytes"]) is int
            assert document["maximum_new_swapout_bytes"] == 64 << 20
        amendment = Path(manifest["swapout_amendment"])
        assert manifest["swapout_amendment"] == str(amendment)
        for path in (amendment, BASE / "gpu-runner.py", BASE / "compose-inspection.py"):
            assert frozen.get(str(path.resolve())) == context["file_sha"](path)
        decoded = json.loads(amendment.read_text())
        runner["validate"](
            decoded,
            Path(policy["swap_anchor_path"]),
            policy["swap_anchor_sha256"],
            context["file_sha"](context["gpu_runner"]["load_baseline"].__globals__["POLICY_PATH"]),
        )
        read_amendment = BASE.parent.parent / "diagnostic-swapin-amendment.json"
        assert frozen.get(str(read_amendment)) == context["file_sha"](read_amendment)
        assert decoded["previous_read_amendment_sha256"] == context["file_sha"](read_amendment)
        return {**policy, "maximum_new_swapout_bytes": 64 << 20}

    context["preflight"].__globals__["swap_policy"] = amended_policy
    original = context["preflight"]
    original.__globals__.update(ORDER=order, FLAGS=FLAGS)
    context.update(ORDER=order, FLAGS=FLAGS)

    def checked(entry, manifest_path=None, identity_path=None):
        result = original(entry, manifest_path, identity_path)
        sources = result["identity"]["sources"]
        assert sources.get(str(Path(__file__).resolve())) == context["file_sha"](Path(__file__))
        for document in (result["manifest"], result["identity"]):
            assert type(document["maximum_new_swapin_bytes"]) is int
            assert document["maximum_new_swapin_bytes"] == 1 << 30
        receipt = Path(entry["plan"]).with_name(Path(entry["plan"]).stem + "-result.json")
        if receipt.exists():
            process = json.loads(receipt.read_text())["commands"][0]
            assert 0 <= process["maximum_live_swapin_bytes"] <= 1 << 30
        return result

    def inspect(entry, manifest_path=None, identity_path=None):
        instance = runpy.run_path(str(context["F02"] / "benchmark-inspection.py"))
        function = instance["inspect"]
        function.__globals__.update(
            preflight=checked,
            check_swap=check_swap,
            ENV={**context["ENV"], "MLXL3_SSD_COLD_READY": entry.get("cold_ready_flag")},
        )
        return function(entry, manifest_path, identity_path)

    context["inspect"] = inspect
    context["preflight"] = checked
    return context
