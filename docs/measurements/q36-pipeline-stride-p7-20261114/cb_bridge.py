"""Q36-PIPELINE-STRIDE-P7 (tool from Q36-CB-GPU-TIME-P4): one bridge process, warmup + measured AR requests, mach-aligned host times."""
import json, os, selectors, subprocess, sys, time
from pathlib import Path

binary, model, prompt_file, out, flag, inject, dylib = sys.argv[1:8]
out = Path(out)
out.mkdir(parents=False, exist_ok=False)
PROMPT = Path(prompt_file).read_text()
now = lambda: time.clock_gettime(time.CLOCK_UPTIME_RAW)  # mach_absolute_time base, as CACurrentMediaTime
report = {"status": "started", "binary": binary, "flag": flag, "inject": inject == "1", "requests": {}}
env = {k: v for k, v in os.environ.items() if k not in ("MLXL3_QWEN_PIPELINE", "MLXL3_QWEN_PIPELINE_STRIDE")}
if flag == "off":
    env["MLXL3_QWEN_PIPELINE"] = "0"
elif flag != "default":
    env["MLXL3_QWEN_PIPELINE_STRIDE"] = flag
if inject == "1":
    env |= {"DYLD_INSERT_LIBRARIES": dylib, "MLXL3_CB_TIMING_PATH": str(out / "cb.bin")}

def save():
    (out / "run.json").write_text(json.dumps(report, indent=1) + "\n")

stderr = open(out / "bridge.stderr", "w")
proc = subprocess.Popen([binary, "bridge", model, "--context-length", "4096"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=stderr, env=env, text=True, bufsize=1)
sel = selectors.DefaultSelector(); sel.register(proc.stdout, selectors.EVENT_READ)

def event(deadline):
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("bridge silent past deadline")
        if not sel.select(timeout=left):
            continue
        line = proc.stdout.readline()
        t = now()
        if not line:
            raise RuntimeError(f"bridge closed stdout (exit {proc.poll()})")
        value = json.loads(line)
        if value.get("type") == "error":
            raise RuntimeError(f"bridge error: {value}")
        return t, value

def generate(rid, tokens=128, timeout=180):
    sent = now()
    proc.stdin.write(json.dumps({"type": "generate", "request_id": rid, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0.0, "top_k": 0, "repetition_penalty": 1.0, "reuse_prompt_cache": False,
        "mcp_enabled": False, "mtp": False, "mtp_depth": 1, "mtp_head_path": "", "dflash2": False, "dflash_mode": 1,
        "dflash_draft_path": ""}) + "\n")
    proc.stdin.flush()
    deadline = time.monotonic() + timeout; deltas = []; text = []
    while True:
        t, v = event(deadline)
        if v.get("request_id") != rid:
            continue
        if v["type"] == "delta":
            deltas.append(t); text.append(v.get("text", ""))
        elif v["type"] == "cancelled":
            raise RuntimeError("cancelled")
        elif v["type"] == "complete":
            stats = v.get("stats", {})
            r = {"sent": sent, "deltas": deltas, "complete": t, "text": "".join(text), "token_hash": v.get("token_hash"),
                 "stats": {k: stats.get(k) for k in ("generated_tokens", "decode_tokens", "decode_seconds", "decode_tps",
                                                     "prefill_seconds", "prefill_tps", "ttft_seconds", "prompt_tokens")}}
            if (not r["text"] or not r["token_hash"] or r["stats"]["generated_tokens"] != tokens or not deltas):
                raise RuntimeError(f"invalid completion {r['stats']}")
            return r

try:
    deadline = time.monotonic() + 120
    while True:
        _, v = event(deadline)
        if v.get("type") == "ready":
            break
    report["ready_at"] = now(); save()
    report["requests"]["warmup"] = generate("warmup"); save()
    for i in range(3):
        time.sleep(2)
        report["requests"][f"run{i}"] = generate(f"run{i}"); save()
    proc.stdin.write('{"type":"shutdown"}\n'); proc.stdin.flush()
    proc.wait(timeout=60)
    report["exit_code"] = proc.returncode
    report["status"] = "complete" if proc.returncode == 0 else "bridge-exit-nonzero"
except BaseException as e:
    report["status"] = "failed"; report["error"] = repr(e)
finally:
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    report["final_exit_code"] = proc.returncode
    save(); stderr.close()
print(json.dumps({"status": report["status"], "error": report.get("error"),
                  "tps": [r["stats"]["decode_tps"] for k, r in report["requests"].items() if k != "warmup"]}))
sys.exit(0 if report["status"] == "complete" else 1)
