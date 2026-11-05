"""Regression checks for the campaign controller; no model or GPU is used."""
import copy
import atexit
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
folder = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("campaign", folder / "run-campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
checks = []
status = "failed"
output = folder / "controller-final-checks.json"
output.write_text('{"status":"running"}\n')
atexit.register(lambda: output.write_text(json.dumps({"status": status, "checks": checks}, indent=2) + "\n"))


def force_cleanup(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()

for mode in ["ui", "engine"]:
    original = json.loads((folder / (mode + "-0-a.json")).read_text())
    reference = campaign.validate(original, mode, None)
    cases = {
        "failed": lambda report: report.update(status="failed"),
        "incomplete": lambda report: report.update(complete=False),
        "missing sample": lambda report: report["samples"].pop(),
        "duplicate": lambda report: report["samples"].__setitem__(1, copy.deepcopy(report["samples"][0])),
        "NaN CPU": lambda report: report["samples"][0].update(cpu_seconds=float("nan")),
        "zero footprint": lambda report: report["samples"][0].update(footprint_bytes=0),
        "different text": lambda report: report["samples"][0].update(text_sha256="0" * 64),
        "invisible": lambda report: report["samples"][0].update(window_visible=False),
        "missing hash": lambda report: report["samples"][0].pop("text_sha256"),
    }
    if mode == "ui":
        cases["partial replay"] = lambda report: report["samples"][0].update(appends=23)
    else:
        cases.update({
            "missing MTP": lambda report: report["samples"][0]["stats"].pop("mtp_accepted_tokens"),
            "zero proposals": lambda report: report["samples"][0]["stats"].update(mtp_proposed_tokens=0),
            "invalid acceptance": lambda report: report["samples"][0]["stats"].update(mtp_accepted_tokens=129),
            "fractional count": lambda report: report["samples"][0]["stats"].update(mtp_proposed_tokens=68.5),
            "invalid rate": lambda report: report["samples"][0]["stats"].update(decode_tps=float("nan")),
            "invalid budget": lambda report: report["samples"][0]["stats"].update(generated_tokens=129),
        })
    for name, edit in cases.items():
        report = copy.deepcopy(original)
        edit(report)
        try:
            campaign.validate(report, mode, reference)
        except (ValueError, KeyError, TypeError):
            checks.append(mode + ": rejected " + name)
        else:
            raise AssertionError("Accepted invalid report: " + mode + "/" + name)

with tempfile.TemporaryDirectory(prefix="mlxl3-controller-check-") as directory:
    child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
    parent = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); print(p.stdout.readline().decode().strip(),flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, "-c", parent, child], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True, cwd=directory)
    try:
        assert select.select([process.stdout], [], [], 3)[0], "Child never became ready"
        child_pid = int(process.stdout.readline())
        campaign.stop_group(process)
        assert process.returncode is not None
        deadline = time.monotonic() + 3
        while True:
            state = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="],
                                   capture_output=True, text=True, timeout=2).stdout.strip()
            if not state or state.startswith("Z"):
                break
            assert time.monotonic() < deadline, "SIGTERM-ignoring child survived parent exit"
            time.sleep(0.02)
        checks.append("cleanup: exited parent does not leave a running SIGTERM-ignoring child")
    finally:
        force_cleanup(process)

    process = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True, cwd=directory)
    try:
        assert select.select([process.stdout], [], [], 3)[0]
        process.stdout.readline()
        try:
            process.communicate(timeout=0.05)
        except subprocess.TimeoutExpired:
            campaign.stop_group(process)
        else:
            raise AssertionError("Silent process unexpectedly exited")
        assert process.returncode is not None
        checks.append("cleanup: silent timeout kills and joins the process")
    finally:
        force_cleanup(process)

    disposable = Path(directory)
    binary = disposable / "build/app-fluidite/performance-a"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexec /bin/sleep 60\n")
    binary.chmod(0o700)
    original_popen, original_root, original_folder = subprocess.Popen, campaign.ROOT, campaign.FOLDER
    interrupted = []

    class InterruptedProcess(original_popen):
        def communicate(self, *args, **kwargs):
            if kwargs.get("timeout") == 180:
                interrupted.append(self)
                raise KeyboardInterrupt("simulated interruption")
            return super().communicate(*args, **kwargs)

    try:
        subprocess.Popen = InterruptedProcess
        campaign.ROOT = campaign.FOLDER = disposable
        try:
            campaign.main("ui")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Interruption was swallowed")
        report = json.loads((disposable / "ui-campaign.json").read_text())
        assert report["status"] == "failed" and not report["complete"] and not report["passes"]
        assert len(interrupted) == 1 and interrupted[0].returncode is not None
        checks.append("interruption: real controller saves failed report and joins its process")
    finally:
        subprocess.Popen, campaign.ROOT, campaign.FOLDER = original_popen, original_root, original_folder
        for process in interrupted:
            force_cleanup(process)

status = "passed"
print(f"Controller checks passed: {len(checks)} negative report/lifecycle checks")
