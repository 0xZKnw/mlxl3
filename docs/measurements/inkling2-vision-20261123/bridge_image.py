"""Drive `mlxl3-rs bridge` with one image request (bounded waits, process always reaped).

usage: python3 -I bridge_image.py ENGINE MODEL_DIR IMAGE PROMPT OUT_JSON [--dspark] [--temperature T] [--max-tokens N]
"""
import argparse, base64, json, mimetypes, os, select, subprocess, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine"); ap.add_argument("model"); ap.add_argument("image"); ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("--dspark", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--history", nargs=2, default=None, help="first user text and assistant answer; the image goes in turn 1")
    a = ap.parse_args()
    mime = mimetypes.guess_type(a.image)[0] or "image/png"
    url = f"data:{mime};base64," + base64.b64encode(open(a.image, "rb").read()).decode()
    proc = subprocess.Popen([a.engine, "bridge", a.model, "--context-length", "16384"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    events, result = [], {"status": "incomplete"}

    def read(deadline):
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"engine exited {proc.returncode}: {proc.stderr.read()[-2000:]}")
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line.strip():
                    return json.loads(line)
        raise TimeoutError("engine silent")

    try:
        while True:
            ev = read(time.time() + 300)
            if ev["type"] == "error":
                raise RuntimeError(ev)
            if ev["type"] == "ready":
                result["ready"] = {k: ev.get(k) for k in ("model", "images_supported", "dspark_supported", "runtime_commit")}
                break
        request = {"type": "generate", "request_id": "img1", "conversation_id": "vision-check",
                   "messages": ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                              {"type": "text", "text": a.history[0]}]},
                                 {"role": "assistant", "content": a.history[1]}] if a.history else [])
                   + [{"role": "user", "content": a.prompt if a.history else
                       [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": a.prompt}]}],
                   "max_tokens": a.max_tokens, "temperature": a.temperature, "dspark": a.dspark}
        if a.effort:
            request["reasoning_effort"] = a.effort
        proc.stdin.write(json.dumps(request) + "\n"); proc.stdin.flush()
        text = {"thinking": "", "answer": ""}
        while True:
            ev = read(time.time() + 300)
            if ev.get("type") == "delta":
                text[ev.get("phase", "answer")] = text.get(ev.get("phase", "answer"), "") + ev.get("text", "")
            elif ev.get("type") == "error":
                raise RuntimeError(ev)
            elif ev.get("type") == "complete":
                result["complete"] = {k: v for k, v in ev.items() if k not in ("text",)}
                break
            else:
                events.append(ev.get("type"))
        result.update(status="ok", text=text, other_events=sorted(set(events)))
    except Exception as e:  # noqa: BLE001 - the failure is the result
        result.update(status="failed", error=str(e)[:3000])
    finally:
        try:
            proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n"); proc.stdin.flush()
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill(); proc.wait(timeout=10)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({"status": result["status"], "error": result.get("error"),
                      "ready": result.get("ready"), "answer": result.get("text", {}).get("answer", "")[:1500],
                      "thinking_chars": len(result.get("text", {}).get("thinking", "")),
                      "complete": result.get("complete")}, indent=1)[:6000])
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
