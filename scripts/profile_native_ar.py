"""Bounded production-CLI AR controls/Metal trace; capture is not coverage validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import signal
import subprocess
import time
from pathlib import Path

from native_json_process import MAX_JSON_LINE, positive_seconds

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path(__file__).with_name("benchmark-flash-bridge.py")
BENCH = runpy.run_path(str(BENCHMARK))
ACCEPTED_NAMES = """
HC_WRITE_NORM ROUTED_NT2_GU ROUTED_NT2_DOWN EXPERT_UP_FINISH EXPERT_GU_PREPARE
EXPERT_OUTPUT_SLAB EXPERT_AXIS_QMV GDN_WHOLE GDN_JOINT_AB QSA_PREPARE QSA_GATE
RECORD_REUSE READ_SPLIT
"""
EXPERIMENT_NAMES = """
GDN_HEAD_GATES QSA_CHAINS QSA_PREFETCH GDN_RESIDENT_BUNDLE MUL1_SWAR TRELLIS_READER8
QMV_K_PRELOAD H128_SIMD MOE_FINISH_MIX GPU_SPIN_WAIT EXPERT_READAHEAD_OFF CACHE_SLRU
QMV_COOPERATIVE ROUTER_PAIR DEVICE_GREEDY RESIDENT_SCRATCH HC_WORKSPACE HC_UP_MIX
HC_DOWN_INJECT SSD_EXTRA_WAVE GDN_HALF_ELISION GDN_NORM_GATE FIXED_RESIDENCY
"""
ACCEPTED = tuple("MLXL3_NATIVE_" + name for name in ACCEPTED_NAMES.split())
EXPERIMENTS = tuple("MLXL3_NATIVE_" + name for name in EXPERIMENT_NAMES.split())
IDS = ("native-ar-warmup", "native-ar-measured")
INSTRUMENTS = ("Thread Activity", "Filesystem Activity", "System Call Trace")
MAX_LOG = 64 * 1024 * 1024


def strict(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite JSON number")
        return parsed

    return json.loads(text, object_pairs_hook=unique, parse_float=number, parse_constant=number)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_files(files):
    if not files:
        raise ValueError("empty frozen file inventory")
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen file differs: " + name)


def environment():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MLXL3_", "MLX_", "DYLD_", "MTL_", "METAL_"))
    }
    env.update({key: "1" for key in ACCEPTED})
    env.update({key: "0" for key in EXPERIMENTS})
    env.update(
        MLXL3_SSD_CACHE_GIB="3",
        MLXL3_SSD_READERS="4",
        MLXL3_SSD_DYNAMIC_READERS="0",
        MLXL3_SSD_LOOKAHEAD_DEPTH="0",
        MLXL3_PROMPT_CACHE_MIB="0",
        MLXL3_SSD_OVERLAP="1",
    )
    return env


def token_budget(value: int) -> int:
    """Only the two independently qualified generation budgets are supported.

    post: __return__ in (32, 128)
    raises: ValueError
    """
    if type(value) is not int or value not in (32, 128):
        raise ValueError("tokens must be exactly 32 or 128")
    return value


def requests(tokens=32, *, warmup=True):
    tokens = token_budget(tokens)
    if type(warmup) is not bool:
        raise ValueError("warmup must be a boolean")
    return [
        {
            "type": "generate",
            "request_id": identifier,
            "messages": [{"role": "user", "content": BENCH["PROMPTS"]["MAIN"]}],
            "max_tokens": tokens,
            "temperature": 0,
            "top_k": 1,
            "repetition_penalty": 1,
            "mtp": False,
            "dflash2": False,
            "mcp_enabled": False,
            "reuse_prompt_cache": False,
        }
        for identifier in (IDS if warmup else IDS[1:])
    ] + [{"type": "shutdown"}]


def validate_bridge(path, engine, model, golden, tokens=32, *, warmup=True):
    tokens = token_budget(tokens)
    if type(warmup) is not bool:
        raise ValueError("warmup must be a boolean")
    identifiers = IDS if warmup else IDS[1:]
    if path.stat().st_size > MAX_LOG:
        raise ValueError("bridge log exceeds bound")
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or any(len(line) > MAX_JSON_LINE or not line.endswith(b"\n") for line in lines):
        raise ValueError("empty, oversized or truncated bridge log")
    events = [strict(line.decode()) for line in lines]
    if any(
        type(e) is not dict
        or not isinstance(e.get("type"), str)
        or e["type"] in ("error", "cancelled")
        for e in events
    ):
        raise ValueError("invalid/error bridge event")
    ready = [e for e in events if e["type"] == "ready"]
    loading = [e for e in events if e["type"] == "loading"]
    if (
        len(loading) != 1
        or loading[0] != {"type": "loading", "model": str(model)}
        or len(ready) != 1
        or events[:2] != [loading[0], ready[0]]
    ):
        raise ValueError("missing/duplicate/out-of-order loading/ready preamble")
    row = ready[0]
    if (
        row.get("backend") != "direct-metal"
        or row.get("runtime_profile") != "release"
        or row.get("context_limit") != 4096
        or Path(row.get("runtime_executable", "")) != engine
        or Path(row.get("model_path", "")) != model
        or type(row.get("runtime_commit")) is not str
        or not row["runtime_commit"]
    ):
        raise ValueError("unexpected backend, executable, model or runtime")
    selected = []
    for identifier in identifiers:
        modes = [
            e
            for e in events
            if e["type"] == "generation_mode" and e.get("request_id") == identifier
        ]
        complete = [
            e for e in events if e["type"] == "complete" and e.get("request_id") == identifier
        ]
        if (
            len(modes) != 1
            or len(complete) != 1
            or events.index(modes[0]) > events.index(complete[0])
        ):
            raise ValueError("missing/duplicate/out-of-order generation evidence")
        event = complete[0]
        BENCH["validate"](event, modes[0], False, tokens)
        expected = {
            "generated_tokens": tokens,
            "decode_tokens": tokens - 1,
            "cached_prompt_tokens": 0,
            "context_limit": 4096,
        }
        if any(
            type(event["stats"].get(k)) is not int or event["stats"][k] != v
            for k, v in expected.items()
        ):
            raise ValueError("wrong decode/context/cache counts")
        if (event["cache_context"], event["token_hash"]) != (golden["text"], golden["token_hash"]):
            raise ValueError("full golden output differs")
        selected.append(event)
    if [e.get("request_id") for e in events if e["type"] == "complete"] != list(identifiers):
        raise ValueError("unexpected or reordered completion")
    return {"ready": row, "complete": selected}


def process_rows(timeout=1.0):
    output = subprocess.check_output(
        ["/bin/ps", "-ww", "-axo", "pid=,ppid=,pgid=,args="], text=True, timeout=timeout
    )
    return {
        int(p): (int(parent), int(group), args)
        for line in output.splitlines()
        if len(parts := line.split(None, 3)) == 4
        for p, parent, group, args in [parts]
    }


def cleanup(child, target, deadline):
    if child is None:
        return True
    known = {}
    registry = target.split(" --registry ", 1)[1].split(" bridge ", 1)[0]
    discovery_error = None
    signal_error = None

    def snapshot():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("process cleanup confirmation deadline exceeded")
        return process_rows(timeout=min(0.25, remaining))

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            alive = child.poll() is None
            rows = snapshot()
            owned = {
                pid
                for pid, (parent, group, args) in rows.items()
                if (alive and group == child.pid) or args == target or known.get(pid) == args
            }
            while extra := {pid for pid, (parent, _, _) in rows.items() if parent in owned} - owned:
                owned |= extra
            known.update({pid: rows[pid][2] for pid in owned})
            for pid in owned:
                current = snapshot().get(pid)
                if current is not None and current[2] == known[pid]:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                    except PermissionError as error:
                        signal_error = error
        except (OSError, subprocess.SubprocessError, TimeoutError) as error:
            discovery_error = error
        finally:
            # Discovery failure must not skip signalling/waiting our direct child.
            # After reap, never signal its potentially recycled numeric PGID.
            if child.poll() is None:
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    # Darwin can deny signalling a process/group already exiting.
                    # Still reap the direct child and verify the whole owned tree;
                    # the error is harmless only if that confirmation succeeds.
                    signal_error = error
            try:
                child.wait(timeout=max(0.001, min(0.1, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
    while time.monotonic() < deadline:
        rows = snapshot()
        if child.poll() is not None and not any(
            group == child.pid or args == target or known.get(pid) == args or registry in args
            for pid, (_, group, args) in rows.items()
        ):
            # An earlier failed snapshot could have missed detached descendants.
            if discovery_error is not None:
                raise RuntimeError(
                    "process discovery failed; descendant cleanup unconfirmed"
                ) from discovery_error
            return True
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    if discovery_error is not None:
        raise RuntimeError(
            "process discovery failed; descendant cleanup unconfirmed"
        ) from discovery_error
    if signal_error is not None:
        raise RuntimeError("process signalling failed; cleanup unconfirmed") from signal_error
    return False


def run(plan_path, out, mode, timeout=120.0):
    out.mkdir(parents=True)
    report = {
        "status": "preflight",
        "parity": False,
        "numeric_parity": False,
        "coverage_validated": False,
        "all_children_joined": False,
        "mode": mode,
    }

    def save():
        temporary = out / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "report.json")

    child, target, deadline = None, "", time.monotonic() + timeout
    save()
    try:
        if mode not in ("control", "trace") or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("control/trace and timeout in (0,300] required")
        plan = strict(plan_path.read_text())
        tokens = token_budget(plan.get("tokens", 32))
        warmup = plan.get("warmup", True)
        engine, model = (Path(plan[k]["path"]).resolve() for k in ("engine", "model"))
        files = plan["source_files"]
        required = {
            Path(__file__).resolve(),
            BENCHMARK.resolve(),
            Path(__file__).with_name("native_json_process.py").resolve(),
        }
        if not required <= {(ROOT / p).resolve() for p in files}:
            raise ValueError("controller/helper inventory incomplete")
        checked_files(files)
        checked_files(plan["model"]["metadata"])
        required_metadata = {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        if not {(model / p).resolve() for p in required_metadata} <= {
            (ROOT / p).resolve() for p in plan["model"]["metadata"]
        }:
            raise ValueError("model metadata inventory incomplete")
        if (
            sha(engine) != plan["engine"]["sha256"]
            or sha(plan["golden"]["path"]) != plan["golden"]["sha256"]
        ):
            raise ValueError("engine/golden fingerprint differs")
        golden = strict(Path(plan["golden"]["path"]).read_text())["runs"][0]
        if not golden["text"] or not golden["token_hash"]:
            raise ValueError("empty golden output")
        stdin, stdout, stderr = (
            out / name for name in ("requests.ndjson", "bridge.ndjson", "process.log")
        )
        stdin.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in requests(tokens, warmup=warmup)
            )
        )
        command = [
            str(engine),
            *plan["engine"].get("arguments", []),
            "--registry",
            str(out / "registry.json"),
            "bridge",
            str(model),
            "--context-length",
            "4096",
        ]
        target, env = " ".join(command), environment()
        report.update(
            plan_sha256=sha(plan_path),
            engine=plan["engine"],
            command=command,
            environment={k: v for k, v in env.items() if k.startswith("MLXL3_")},
            requests_sha256=sha(stdin),
            capture_status="not_requested",
            generated_tokens=tokens,
            warmup=warmup,
        )
        if mode == "trace":
            trace = out / "native-ar.trace"
            template = plan.get("template", "Metal System Trace")
            if template not in ("Metal System Trace", "Time Profiler"):
                raise ValueError("unsupported trace template")
            instruments = plan.get("instruments", list(INSTRUMENTS))
            if (
                type(instruments) is not list
                or any(type(name) is not str for name in instruments)
                or len(set(instruments)) != len(instruments)
                or not set(instruments) <= set(INSTRUMENTS)
            ):
                raise ValueError("instruments must be a unique subset of the supported three")
            window = plan.get("recording_window_seconds")
            if "recording_window_seconds" in plan and (
                type(window) is not int or not 1 <= window <= 30
            ):
                raise ValueError("recording_window_seconds must be an integer in [1,30]")
            tool = plan["xctrace"]
            if sha(tool["path"]) != tool["sha256"]:
                raise ValueError("xctrace fingerprint differs")
            options = plan["recording_options"]
            if sha(options["path"]) != options["sha256"]:
                raise ValueError("recording options fingerprint differs")
            env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
            command = [
                tool["path"],
                "record",
                "--template",
                template,
                "--recording-options",
                options["path"],
                "--time-limit",
                "180s",
                "--no-prompt",
                "--output",
                str(trace),
                "--target-stdin",
                str(stdin),
                "--target-stdout",
                str(stdout),
                "--launch",
                "--",
                *command,
            ]
            if window is not None:
                command[2:2] = ["--window", f"{window}s"]
            for instrument in instruments:
                command[2:2] = ["--instrument", instrument]
            report.update(
                trace_path=str(trace),
                capture_status="incomplete",
                trace_command=command,
                recording_options=options,
                instruments=instruments,
                recording_window_seconds=window,
                template=template,
            )
        report["status"] = "running"
        save()
        with stdin.open("rb") as source, stdout.open("wb") as output, stderr.open("wb") as log:
            child = subprocess.Popen(
                command,
                stdin=source,
                stdout=output if mode == "control" else log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
            report["pid"] = child.pid
            save()
            while child.poll() is None:
                if time.monotonic() >= deadline - min(5.0, timeout / 3):
                    raise TimeoutError("profile child exceeded deadline")
                if any(p.stat().st_size > MAX_LOG for p in (stdout, stderr)):
                    raise ValueError("process log exceeds bound")
                try:
                    child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode != 0:
            raise RuntimeError(f"process exited {child.returncode}")
        if any(
            group == child.pid or str(out / "registry.json") in args
            for _, group, args in process_rows().values()
        ):
            raise RuntimeError("launcher exited while a target/descendant is still running")
        if mode == "trace" and sha(options["path"]) != options["sha256"]:
            raise ValueError("recording options changed during execution")
        report["evidence"] = validate_bridge(stdout, engine, model, golden, tokens, warmup=warmup)
        if sha(engine) != plan["engine"]["sha256"]:
            raise ValueError("engine changed during execution")
        report["numeric_parity"] = True
        if mode == "trace":
            if not trace.is_dir() or not any(p.is_file() for p in trace.rglob("*")):
                raise ValueError("trace bundle missing/empty")
            report.update(status="captured_pending_export", capture_status="created_not_exported")
        else:
            report["status"] = "validated_control"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        log = out / "process.log"
        if log.exists():
            with log.open("rb") as stream:
                message = stream.read(65536).decode(errors="replace").lower()
            if any(
                s in message
                for s in ("not authorized", "permission denied", "authorization failed")
            ):
                report.update(status="permission_denied", capture_status="denied")
        raise
    finally:
        try:
            # Cleanup has its own one-second ceiling after generation/preflight.
            # A consumed generation deadline is not evidence that cleanup failed.
            report["all_children_joined"] = cleanup(child, target, time.monotonic() + 1.0)
        except BaseException as error:  # noqa: BLE001 -- Preserve cleanup failure beside original.
            report["cleanup_error"] = repr(error)
        if not report["all_children_joined"]:
            report.update(status="failed_cleanup", parity=False)
        report["parity"] = report["status"] == "validated_control" and report["all_children_joined"]
        save()
    if not report["all_children_joined"]:
        raise RuntimeError("profile child cleanup failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "trace"), required=True)
    parser.add_argument("--timeout", type=positive_seconds, default=120.0)
    parser.add_argument("--run-after-root-grant", action="store_true")
    args = parser.parse_args()
    if not args.run_after_root_grant:
        parser.error("explicit exclusive root grant required")

    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("profile interrupted by SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    run(args.plan, args.out.resolve(), args.mode, args.timeout)
