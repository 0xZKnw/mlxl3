import hashlib
import json
import os
import re
import resource
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_processes(text: str) -> list[tuple[int, int, str]]:
    """Parse ps rows; reject partial or ambiguous snapshots.

    raises: ValueError
    post: all(pid > 0 and parent >= 0 and name for pid, parent, name in __return__)
    post: len({pid for pid, _, _ in __return__}) == len(__return__)
    """
    lines = text.splitlines()
    if not lines or lines[0].split()[:2] != ["PID", "PPID"]:
        raise ValueError("invalid process header")
    rows = []
    seen = set()
    for line in lines[1:]:
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError("incomplete process row")
        pid, parent = int(parts[0]), int(parts[1])
        if pid <= 0 or parent < 0 or pid in seen:
            raise ValueError("invalid or duplicate process ID")
        seen.add(pid)
        rows.append((pid, parent, parts[2]))
    if not rows:
        raise ValueError("empty process snapshot")
    return rows


def competing_processes(rows, own_pid=None):
    parents = {pid: parent for pid, parent, _ in rows}
    compilers = {
        "swift-frontend",
        "swiftc",
        "swift-build",
        "xcodebuild",
        "rustc",
        "cargo",
        "cargo-kani",
        "kani",
        "kani-compiler",
        "cbmc",
        "clang",
        "clang++",
        "cc1",
        "ld",
        "ld64",
    }
    engines = {
        "mlxl3-rs",
        "mlxl3-native",
        "mlxl3-engine",
        "prototype-engine",
        "s1-engine",
        "s4e-engine",
    }

    def owned(pid):
        visited = set()
        while pid in parents and pid not in visited and pid != own_pid:
            visited.add(pid)
            pid = parents[pid]
        return pid == own_pid

    return [
        row
        for row in rows
        if (
            Path(row[2]).name in compilers
            or "mlxl3-check" in row[2]
            or (Path(row[2]).name in engines and not owned(row[0]))
        )
    ]


def check_quiet(record, own_pid=None):
    rows = parse_processes(
        subprocess.check_output(["ps", "-axo", "pid,ppid,comm"], text=True, timeout=5)
    )
    conflicts = competing_processes(rows, own_pid)
    record["quiet_checks"] = record.get("quiet_checks", 0) + 1
    record["last_process_snapshot"] = rows
    if conflicts:
        record["competing_processes"] = conflicts
        raise RuntimeError(f"concurrent build, verifier or external engine: {conflicts}")


def stop(child):
    """Stop and join an owned process, accepting a concurrent normal exit.

    raises: PermissionError, ProcessLookupError, subprocess.TimeoutExpired
    post: child.poll() is not None
    """
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            if child.poll() is None:
                raise
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            if child.poll() is None:
                raise
        child.wait(timeout=5)


def main(path):
    destination = path.parent
    output = destination / (path.stem + "-result.json")
    assert not output.exists(), "refusing to overwrite a run"
    matrix = json.loads(path.read_text())
    assert matrix and len({item["name"] for item in matrix}) == len(matrix)
    previous = runpy.run_path(
        "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/speed-heads/bench-supervisor.txt"
    )
    parse_vm = runpy.run_path("benchmarks/benchmark_flash_memory.py")["parse_vm"]
    records = []
    result = {
        "status": "incomplete",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "commands": records,
    }

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for item in matrix:
            record = {**item, "status": "running", "before": previous["snapshot"](parse_vm)}
            records.append(record)
            child = None
            started = time.monotonic()
            try:
                assert type(item.get("require_quiet", False)) is bool
                quiet = item.get("require_quiet", False)
                assert (
                    record["before"]["pmset -g batt"].splitlines()[0]
                    == "Now drawing from 'AC Power'"
                )
                if "before_launch" in previous:
                    previous["before_launch"](record)
                if quiet:
                    check_quiet(record)
                binary = Path(item["command"][0])
                record["executable_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                save()
                logfile = destination / (item["name"] + ".log")
                with logfile.open("x") as stream:
                    child = subprocess.Popen(
                        item["command"],
                        env={**os.environ, **item["env"]},
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    record["pid"] = child.pid
                    save()

                    def monitored_vm(text, quiet=quiet, record=record, child=child):
                        value = parse_vm(text)
                        if quiet:
                            check_quiet(record, child.pid)
                        return value

                    record["exit_code"] = previous["monitor"](child, record, monitored_vm)
                if quiet:
                    check_quiet(record, child.pid)
                record["after"] = previous["snapshot"](parse_vm)
                assert record["exit_code"] == item["expected_exit"]
                record["children_maximum_rss_bytes"] = int(
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                ) * (1 if sys.platform == "darwin" else 1024)
                assert 0 < record["children_maximum_rss_bytes"] <= 10 << 30
                for key in ("Swapins", "Swapouts"):
                    delta = record["after"]["vm_bytes"][key] - record["before"]["vm_bytes"][key]
                    assert 0 <= delta <= 256 << 20
                    record[key.lower() + "_bytes"] = delta
                assert (
                    record["after"]["pmset -g batt"].splitlines()[0]
                    == record["before"]["pmset -g batt"].splitlines()[0]
                )
                log = logfile.read_text()
                if item.get("native_test"):
                    expected = "ok" if item["expected_exit"] == 0 else "FAILED"
                    assert re.search(r"running 1 test\s", log)
                    assert re.search(
                        r"(?m)^test " + re.escape(item["native_test"]) + r" \.\.\. ", log
                    )
                    assert (
                        "1 passed; 0 failed" if expected == "ok" else "0 passed; 1 failed"
                    ) in log
                for text in item.get("required_text", []):
                    assert text in log, text
                if item.get("native_head_reference"):
                    reference = json.loads(Path(item["native_head_reference"]).read_text())
                    chain = reference["native_chain"]
                    assert (
                        reference["status"] == "complete"
                        and chain["status"] == "passed_native_chain"
                    )
                    assert chain["logit_count"] in [n * 248320 for n in range(1, 8)]
                    assert reference["draft_head"]["logits"]["count"] == chain["logit_count"]
                record["status"] = "completed_expected_outcome"
            except BaseException as error:
                record.update(status="failed_or_interrupted", error=repr(error))
                raise
            finally:
                if child is not None:
                    stop(child)
                    record["exit_code"] = child.returncode
                record["reaped"] = child is None or child.poll() is not None
                record["seconds"] = time.monotonic() - started
                save()
                with Path("opti.md").open("a") as stream:
                    stream.write(
                        f"\n- 45-Q/R processus {item['name']} : **{record['status']}**, joint={record['reaped']} ; {output}.\n"
                    )
                print(
                    {
                        k: record.get(k)
                        for k in ("name", "status", "reaped", "seconds", "swapouts_bytes")
                    },
                    flush=True,
                )
        result["status"] = "completed_expected_outcomes_not_speed"
    except BaseException as error:
        result.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
