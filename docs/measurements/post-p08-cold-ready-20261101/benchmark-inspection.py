"""GPU1 Cold A/B adapter; reuse the existing plan and external-oracle inspectors."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
F02 = BASE.parent / "post-p08-dense-epilogue-20261031"
COMPOSE = BASE.parent / "post-p08-compose-20261101/benchmark-inspection.py"
upstream = runpy.run_path(str(F02 / "benchmark-inspection.py"))
gpu_runner = runpy.run_path(str(BASE / "gpu-runner.py"))
EXPERIMENT = "POSTP08-COLD-READY-01-GPU1"
ORDER = {"bridge17": list("AB"), "pilot128": list("ABBA")}
FLAGS = {"A": "0", "B": "1"}
ENV = {
    **upstream["ENV"],
    "MLXL3_FLASH_PREPARED_CONV": "0",
    "MLXL3_FLASH_DENSE_EPILOGUE": "0",
    "MLXL3_FLASH_PREFILL_HEADS": "1",
}
DEPENDENCIES = (COMPOSE, BASE / "gpu-runner.py", F02 / "bounded-gpu-runner.py")
file_sha = upstream["file_sha"]
bridge = upstream["bridge"]
pool = upstream["pool"]


def swap_policy(manifest, identity, frozen):
    loader = gpu_runner["load_baseline"]
    anchor_path = loader.__globals__["ANCHOR_PATH"]
    policy_path = loader.__globals__["POLICY_PATH"]
    for document in (manifest, identity):
        assert type(document["maximum_new_swapout_bytes"]) is int
        assert document["maximum_new_swapout_bytes"] == 32 << 20, "GPU1 swap budget"
        assert document["swap_anchor_path"] == str(anchor_path), "GPU1 anchor agreement"
        assert document["swap_anchor_sha256"] == file_sha(anchor_path), "GPU1 anchor hash"
    assert manifest["policy_amendment"] == str(policy_path), "GPU1 policy path"
    for path in (anchor_path, policy_path, *DEPENDENCIES):
        assert frozen.get(str(path.resolve())) == file_sha(path), "unfrozen GPU1 dependency"
    gpu_runner["validate_identity_sources"](identity)
    return {
        "maximum_new_swapout_bytes": 32 << 20,
        "swap_anchor_path": str(anchor_path),
        "swap_anchor_sha256": manifest["swap_anchor_sha256"],
        "swap_anchor_vm_bytes": loader(anchor_path),
    }


def preflight(entry, manifest_path=None, identity_path=None):
    """Check a frozen A/B entry before launch, or inside inspect's failure boundary."""
    controls = {**ENV, "MLXL3_SSD_COLD_READY": entry["cold_ready_flag"]}
    configured = runpy.run_path(str(COMPOSE))
    core = configured["preflight"]
    core.__globals__.update(
        __file__=__file__,
        ROOT=ROOT,
        BASE=BASE,
        EXPERIMENT=EXPERIMENT,
        ORDER=ORDER,
        FLAGS={arm: ("0", "0") for arm in FLAGS},
        ENV=controls,
    )
    configured["upstream"]["swap_policy"] = swap_policy
    result = core(entry, manifest_path, identity_path)
    for row in result["manifest"]["plans"]:
        assert row["cold_ready_flag"] == FLAGS[row["arm"]], "Cold arm flag"
        assert row["gdn_flag"] == row["f02_flag"] == "0", "other candidates disabled"
    assert {"final_cli", "python_runner", "libtest"} <= result["identity"]["binaries"].keys(), (
        "required binary identities"
    )
    report_path = Path(entry["expected_report"])
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["environment"]["MLXL3_SSD_COLD_READY"] == entry["cold_ready_flag"], (
            "report Cold flag"
        )
    return result


def inspect(entry, manifest_path=None, identity_path=None):
    # Each call gets independent globals; the inherited try/finally persists every failure.
    instance = runpy.run_path(str(F02 / "benchmark-inspection.py"))
    original = instance["inspect"]
    original.__globals__.update(
        preflight=preflight,
        ENV={**ENV, "MLXL3_SSD_COLD_READY": entry.get("cold_ready_flag")},
    )
    return original(entry, manifest_path, identity_path)
