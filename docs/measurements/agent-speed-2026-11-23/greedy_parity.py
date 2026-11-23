#!/usr/bin/env python3
"""AGENT-2026-11-23-VITESSE (a): one greedy Agent turn (bugfix task, real sandbox tools),
recorded as its full event stream, for comparing decode paths selected by environment flags.

usage: greedy_parity.py ENGINE REGISTRY MODEL_DIR MODEL_NAME HELPER OUT.json MODE
MODE: plain (MLXL3_INKLING_GPU_DRAW=0, MLXL3_INKLING_LOOKUP=0) | gpu (MLXL3_INKLING_LOOKUP=0) | lookup (defaults)
"""
import importlib.util, json, os, selectors, shutil, subprocess, sys, tempfile, time
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("effort", here.parent / "agent-effort-2026-11-23/effort_trial.py")
effort = importlib.util.module_from_spec(spec); spec.loader.exec_module(effort)
engine, registry_source, model_dir, model_name, helper, out, mode = sys.argv[1:8]
env = dict(os.environ)
env.pop("MLXL3_INKLING_GPU_DRAW", None); env.pop("MLXL3_INKLING_LOOKUP", None)
if mode == "plain":
    env.update(MLXL3_INKLING_GPU_DRAW="0", MLXL3_INKLING_LOOKUP="0")
elif mode == "gpu":
    env.update(MLXL3_INKLING_LOOKUP="0")
elif mode != "lookup":
    raise SystemExit(f"unknown mode {mode}")
work = Path(tempfile.mkdtemp(prefix="mlxl3-parity-")); root = work / "files"; studio = work / "studio"; studio.mkdir()
project = studio / "project.json"; registry = work / "registry" / "models.json"; registry.parent.mkdir()
entry = dict(next(iter(json.loads(Path(registry_source).read_text())["models"].values())))
entry.update(path=model_dir, name=model_name)
registry.write_text(json.dumps({"version": 1, "models": {model_name: entry}}))
(registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
    "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
    "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                "args": ["--agent-tools", str(root), str(project), "long-calls"], "callTimeoutSeconds": 43}}}))
for path, text in effort.TASKS["bugfix"]["files"].items():
    (root / path).parent.mkdir(parents=True, exist_ok=True); (root / path).write_text(text)
project.write_text(json.dumps({"permissions": {"read": True, "write": True, "run": True, "commandSeconds": 5,
                                               "harness": "javascript", "internet": False}}))
report = {"id": "AGENT-2026-11-23-VITESSE", "mode": mode, "engine": engine, "status": "running", "events": []}
process = subprocess.Popen([engine, "--registry", str(registry), "bridge", model_name, "--context-length", "32768"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(Path(out).with_suffix(".stderr.log"), "w"),
                           text=True, bufsize=1, env=env)
sel = selectors.DefaultSelector(); sel.register(process.stdout, selectors.EVENT_READ)
def read(seconds=600):
    if not sel.select(timeout=seconds):
        raise RuntimeError(f"no event within {seconds} s")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("bridge exited")
    return json.loads(line)
def send(message):
    process.stdin.write(json.dumps(message) + "\n"); process.stdin.flush()
try:
    while (event := read(900)).get("type") != "ready":
        if event.get("type") == "error":
            raise RuntimeError(event.get("message"))
    send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
    while not ((event := read()).get("type") == "mcp_status" and event.get("request_id") == "mcp"):
        pass
    send({"type": "generate", "request_id": "parity", "conversation_id": f"parity-{time.time_ns()}",
          "messages": [{"role": "system", "content": effort.instructions()},
                       {"role": "user", "content": effort.TASKS["bugfix"]["prompt"]}],
          "max_tokens": -1, "tool_rounds": 256, "mcp_enabled": True,
          "temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0})
    started = time.monotonic(); answer = ""
    while True:
        if time.monotonic() - started > 900:
            raise RuntimeError("turn exceeded 900 s")
        event = read()
        if event.get("request_id") not in (None, "parity"):
            continue
        kind = event.get("type")
        if kind == "delta":
            report["events"].append(["delta", event.get("phase"), event.get("text")])
            if event.get("phase") == "answer":
                answer += event.get("text", "")
        elif kind == "tool_start":
            report["events"].append(["tool_start", event.get("tool_name"), event.get("arguments")])
        elif kind == "tool_result":
            report["events"].append(["tool_result", event.get("is_error"), event.get("text")])
        elif kind == "error":
            raise RuntimeError(event.get("message"))
        elif kind == "complete":
            stats = event.get("stats", {})
            report.update(token_hash=event.get("token_hash"), stop_reason=event.get("stop_reason"),
                          tool_rounds=event.get("tool_rounds"), generated_tokens=stats.get("generated_tokens"),
                          decode_tps=stats.get("decode_tps"), seconds=round(time.monotonic() - started, 1),
                          passed=effort.oracle("bugfix", root, answer))
            break
    report["status"] = "complete"
except Exception as error:
    report["status"] = "failed"; report["error"] = str(error)
finally:
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if process.poll() is None:
        try:
            send({"type": "shutdown"}); process.wait(timeout=30)
        except Exception:
            process.kill(); process.wait()
    shutil.rmtree(work, ignore_errors=True)
print(json.dumps({k: report.get(k) for k in ("mode", "status", "error", "passed", "tool_rounds", "generated_tokens", "decode_tps", "seconds", "token_hash")}))
sys.exit(0 if report["status"] == "complete" else 1)
