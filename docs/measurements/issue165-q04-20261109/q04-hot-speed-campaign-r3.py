import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time

out = Path(__file__).resolve().parent
folder = out / "q04-hot-speed-r3"
folder.mkdir(exist_ok=False)
exe = out / "q04-hot-test"
identity = json.loads((out / "q04-hot-identity.json").read_text())
assert hashlib.sha256(exe.read_bytes()).hexdigest() == identity["binary_sha256"]
for name, digest in identity["source"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
quiet = runpy.run_path("docs/measurements/slotstream-20261026/supervisor.py")["check_quiet"]
env = {k: v for k, v in os.environ.items() if not k.startswith("MLXL3_")}
env.update(MLXL3_FLASH_MODEL="/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531",
           MLXL3_SSD_CACHE_GIB="3", MLXL3_NATIVE_CACHE_SLRU="0",
           MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE="1", MLXL3_NATIVE_HC_WORKSPACE="0", MLXL3_SSD_READERS="4",
           DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
state = {"status": "running", "supervisor_pid": os.getpid(), "production_cli": False,
         "mtp": False, "steps": [], "plan": ["A1", "B1", "C1", "C2", "B2", "A2"]}
status = folder / "status.json"


def save():
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+"\n")
    temporary.replace(status)


def conditions():
    result = {}
    for command in (["pmset", "-g", "batt"], ["pmset", "-g", "therm"], ["sysctl", "vm.swapusage"], ["vm_stat"]):
        r = subprocess.run(command, capture_output=True, text=True, timeout=5)
        result[" ".join(command)] = {"exit": r.returncode, "output": r.stdout+r.stderr}
    return result


def interrupted(*_):
    raise KeyboardInterrupt("user interruption")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def run(name, tau, preflight=False):
    arm = folder / name
    arm.mkdir()
    record = {"name": name, "tau": tau, "status": "running", "preflight": preflight}
    state["steps"].append(record)
    quiet(record)
    record["before"] = conditions()
    save()
    arm_env = {**env, "MLXL3_NATIVE_Q04_SPEED_OUTPUT": str(arm / "report.json")}
    if tau is not None:
        arm_env["MLXL3_NATIVE_Q04_QUALITY_TAU"] = str(tau)
    with (arm / "test.log").open("w") as log:
        child = subprocess.Popen([str(exe), "gpu::fast_profile_tests::native_q04_ar_generation_pilot",
                                  "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                                 env=arm_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["model_pid"] = child.pid
        save()
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic()-started > 180:
                    raise TimeoutError(name)
                time.sleep(.5)
                quiet(record, child.pid)
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            record.update(status="interrupted_or_failed", exit=child.returncode, reaped=True)
            save()
            raise
    record.update(exit=code, reaped=True, after=conditions())
    assert code == 0 and "1 passed;" in (arm / "test.log").read_text(), name
    report = json.loads((arm / "report.json").read_text())
    assert report["status"] == "complete_native_ar_prototype_pilot" and (report["tau"] is None if tau is None else math.isclose(report["tau"], tau, rel_tol=1e-6))
    assert len(report["runs"]) == 2 and [r["warmup"] for r in report["runs"]] == [True, False]
    for r in report["runs"]:
        assert len(r["tokens"]) == 64 and r["decode_tokens"] == 63 and r["text"].strip()
        assert math.isfinite(r["decode_tokens_per_second"]) and r["decode_tokens_per_second"] > 0
        if tau is not None:
            assert r["policy"]["rows"] == 3024 and r["policy"]["swaps"] > 0
    assert all(r["device_hot_submissions"] > 0 for r in report["runs"]), "resident Hot inactive"
    old = json.loads((out / "q04-sushi-speed" / name / "report.json").read_text())
    assert report["prompt_tokens"] == old["prompt_tokens"]
    assert [r["tokens"] for r in report["runs"]] == [r["tokens"] for r in old["runs"]], "Hot changed this fixed-policy generation trajectory"
    record["prior_tokens_exactly_equal"] = True
    record["hot_submissions"] = [r["device_hot_submissions"] for r in report["runs"]]
    measured = report["runs"][1]
    record.update(status="validated_native_prototype_pilot", decode_tokens_per_second=measured["decode_tokens_per_second"],
                  ttft_seconds=measured["ttft_seconds"], total_seconds=measured["total_seconds"],
                  text_sha256=hashlib.sha256(measured["text"].encode()).hexdigest())
    save()
    with Path("opti.md").open("a") as journal:
        journal.write(f"\n- I165-Q04 speed {name} complete/reaped, preflight={preflight}: {measured['decode_tokens_per_second']:.6f} native-AR-prototype decode tok/s,64outputs/63timedsteps. Raw {arm}/report.json; no aggregate before controls, no CLI/MTP claim.\n")
    print(name, record["status"], record["decode_tokens_per_second"], flush=True)
    return report


try:
    save()
    state["quality_campaign_repeated"] = False
    save()
    reports = {}
    for name, tau in [("A1", None), ("B1", .1), ("C1", .2), ("C2", .2), ("B2", .1), ("A2", None)]:
        reports[name] = run(name, tau)
    first = reports["A1"]
    assert all(r["prompt_tokens"] == first["prompt_tokens"] and r["checkpoint"] == first["checkpoint"] for r in reports.values())
    for x, y in [("A1", "A2"), ("B1", "B2"), ("C1", "C2")]:
        assert reports[x]["runs"][1]["tokens"] == reports[y]["runs"][1]["tokens"], (x, y)
    means = {key: sum(reports[key+str(i)]["runs"][1]["decode_tokens_per_second"] for i in (1,2))/2 for key in "ABC"}
    summary = {"status": "complete_native_ar_prototype_pilot", "means_decode_tokens_per_second": means,
               "change_percent": {k:100*(means[k]/means["A"]-1) for k in "BC"},
               "control_drift_percent":100*(reports["A2"]["runs"][1]["decode_tokens_per_second"]/reports["A1"]["runs"][1]["decode_tokens_per_second"]-1),
               "production_cli":False, "mtp":False, "repetitions_per_setting":2,
               "limitation":"Short native AR pilot with early resident Hot on all arms and HCworkspace0, same-command GPU picker, host agreement. Different configuration from previous pilots; not a CLI/MTP speed qualification."}
    (folder / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    state["status"] = "complete_native_ar_prototype_pilot"
except BaseException as error:
    state.update(status="failed_or_interrupted", error=repr(error))
    with Path("opti.md").open("a") as journal:
        journal.write(f"\n- I165-Q04 speed campaign failed/interrupted: {error!r}; raw {status}. No complete speed qualification.\n")
    raise
finally:
    save()
