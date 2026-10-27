"""Bounded functional Qwen3.6 file creation/cancellation qualification, no speed claim."""

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from native_json_process import JsonProcess

out = root / "docs/measurements/desktop-updater-html-agent"
engine = root / "target/release/mlxl3-rs"
helper = root / "apps/MLXL3Studio/.build/out/Products/Debug/MLXL3Studio"
env = dict(
    os.environ,
    DYLD_LIBRARY_PATH="/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx/lib",
)
report = {
    "status": "running",
    "purpose": "functional only, no performance comparison",
    "source_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_diff_sha256": hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary"])
    ).hexdigest(),
    "system": platform.platform(),
    "conditions": subprocess.check_output(["pmset", "-g", "batt"], text=True),
    "hardware": subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip(),
    "hashes": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [engine, helper]
    },
    "protocol": {
        "context": 4096,
        "temperature": 0,
        "top_k": 1,
        "repetition_penalty": 1,
        "max_tokens": 2048,
        "mtp": False,
        "dflash": False,
        "deadline_seconds": 180,
        "repetitions": 1,
    },
}
report["source_files"] = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(
        [
            *(root / "apps/MLXL3Studio/Sources/MLXL3Studio").glob("*.swift"),
            *(root / "native/src").glob("*.rs"),
            *(root / "native/ffi").glob("*.cpp"),
        ]
    )
}
report["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
report["overall_deadline_seconds"] = 300
raw = (out / "qwen-events.jsonl").open("w")


def interrupted(*_):
    raise InterruptedError("Qualification interrupted")


signal.signal(signal.SIGTERM, interrupted)


def save():
    (out / "qwen-journey.json").write_text(json.dumps(report, indent=2) + "\n")


def receive(bridge, kind, seconds=180):
    deadline = time.monotonic() + seconds
    events = []
    while True:
        event = bridge.receive(deadline)
        raw.write(json.dumps(event) + "\n")
        raw.flush()
        events.append(event)
        if event.get("type") == "error":
            raise RuntimeError(event)
        if event.get("type") == kind:
            return event, events


try:
    report["runtime"] = json.loads(
        subprocess.check_output([str(engine), "runtime-info"], env=env, text=True, timeout=10)
    )
    models = json.loads(
        subprocess.check_output([str(engine), "list", "--json"], env=env, text=True, timeout=10)
    )
    model = next(item for item in models if item["name"] == "qwen3.6-35b-a3b")
    report["model"] = model
    report["model_config_sha256"] = hashlib.sha256(
        (Path(model["path"]) / "config.json").read_bytes()
    ).hexdigest()
    save()
    with tempfile.TemporaryDirectory(prefix="mlxl3-qwen-agent-") as directory:
        directory = Path(directory)
        workspace = directory / "files"
        workspace.mkdir()
        metadata = directory / "runtime"
        metadata.mkdir()
        registry = metadata / "models.json"
        registry.write_text(json.dumps({"version": 1, "models": {model["name"]: model}}))
        (metadata / "mcp.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "mcpServers": {
                        "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
                        "sandbox": {
                            "enabled": True,
                            "command": str(helper),
                            "args": ["--agent-tools", str(workspace)],
                        },
                    },
                }
            )
        )
        with (out / "qwen-stderr.log").open("wb") as stderr:
            with JsonProcess(
                [
                    str(engine),
                    "--registry",
                    str(registry),
                    "bridge",
                    model["name"],
                    "--context-length",
                    "4096",
                ],
                env=env,
                stderr=stderr,
            ) as bridge:
                receive(bridge, "ready")
                bridge.send({"type": "set_mcp", "enabled": True}, time.monotonic() + 10)
                catalog, _ = receive(bridge, "mcp_status", 20)
                assert catalog["mcp_tools"] == 1 and not catalog["mcp_errors"]

                def request(identifier, content, limit=2048):
                    return {
                        "type": "generate",
                        "request_id": identifier,
                        "conversation_id": identifier,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a coding agent. Use sandbox.run for filesystem work, then verify the file. Keep reasoning brief. Commands run offline in a private folder for at most 5 seconds.",
                            },
                            {"role": "user", "content": content},
                        ],
                        "max_tokens": limit,
                        "temperature": 0,
                        "top_k": 1,
                        "repetition_penalty": 1,
                        "mcp_enabled": True,
                        "dflash2": False,
                        "mtp": False,
                    }

                bridge.send(
                    request(
                        "create-html",
                        'Use sandbox.run now to create index.html containing <!doctype html><html><body><button onclick="this.textContent=42">MLXL3-QWEN-OK</button></body></html>, then cat index.html. Use one shell command. After the tool succeeds, reply Done.',
                    ),
                    time.monotonic() + 10,
                )
                complete, events = receive(bridge, "complete")
                assert any(e.get("type") == "tool_start" for e in events), "No real tool call"
                assert any(
                    e.get("type") == "tool_result"
                    and not e.get("is_error")
                    and "MLXL3-QWEN-OK" in e.get("text", "")
                    for e in events
                ), "No successful file readback"
                assert any(e.get("phase") == "tool_preparation" for e in events), (
                    "Command preparation stayed invisible"
                )
                html = (workspace / "index.html").read_text()
                assert (
                    "<!doctype html>" in html.lower()
                    and "MLXL3-QWEN-OK" in html
                    and "onclick" in html
                )
                (out / "qwen-generated.html").write_text(html)
                stats = complete["stats"]
                assert 0 < stats["generated_tokens"] <= 2048
                assert all(
                    math.isfinite(stats[k]) and stats[k] >= 0
                    for k in ["ttft_seconds", "decode_tps", "prefill_tps"]
                )
                report["file"] = {
                    "status": "passed",
                    "file_oracle": True,
                    "tool_calls": sum(e.get("type") == "tool_start" for e in events),
                    "stats": stats,
                    "answer": complete.get("assistant_context"),
                }
                save()
                bridge.send(
                    request(
                        "cancel",
                        "Explain in detail how to build a renderer, do not call a tool yet.",
                    ),
                    time.monotonic() + 10,
                )
                _, first = receive(bridge, "delta", 60)
                sent = time.monotonic()
                os.kill(bridge.process.pid, signal.SIGUSR1)
                cancelled, _ = receive(bridge, "cancelled", 10)
                assert cancelled["request_id"] == "cancel"
                report["cancel"] = {
                    "status": "passed",
                    "acknowledged": True,
                    "seconds_to_ack": time.monotonic() - sent,
                }
                bridge.send(
                    request("resume", "Reply Hello in one sentence. Do not use tools.", 256),
                    time.monotonic() + 10,
                )
                resumed, _ = receive(bridge, "complete", 60)
                assert (
                    resumed["request_id"] == "resume" and resumed["stats"]["generated_tokens"] > 0
                )
                assert "hello" in resumed.get("assistant_context", "").lower(), resumed
                report["resume"] = {
                    "status": "passed",
                    "generated_tokens": resumed["stats"]["generated_tokens"],
                }
        report["engine_joined"] = True
    report["status"] = "passed"
except BaseException as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
finally:
    raw.close()
    save()
print("Qwen3.6 actual sandbox file, visible preparation, cancellation and resumed turn passed")
