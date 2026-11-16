"""Release 2.0.13: a T=0 request with a presence penalty must bypass the
certified head (no lfm3_cert stats); a plain greedy request must use it."""
import json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from native_json_process import JsonProcess  # noqa: E402

engine, model, out = sys.argv[1:4]
directory = tempfile.TemporaryDirectory(prefix="mlxl3-presence-")
env = {k: v for k, v in os.environ.items() if k not in ("MLXL3_LFM3_CERT_HEAD", "MLXL3_LFM3_LOOKUP")}
process = JsonProcess([engine, "--registry", str(Path(directory.name) / "models.json"), "bridge", model,
                       "--context-length", "4096"], stderr=None, env=env)
report = {"status": "error", "runs": {}}
try:
    for index, (name, presence) in enumerate((("greedy", 0.0), ("presence", 0.5))):
        deadline = time.monotonic() + 300
        rid = str(index + 1)
        process.send({"type": "generate", "request_id": rid, "conversation_id": f"presence-{rid}",
                      "messages": [{"role": "user", "content": "Explique en quelques phrases pourquoi le ciel est bleu."}],
                      "max_tokens": 64, "temperature": 0, "top_k": 1, "repetition_penalty": 1,
                      "presence_penalty": presence, "reuse_prompt_cache": False}, deadline)
        while True:
            event = process.receive(deadline)
            if event.get("type") == "error":
                raise RuntimeError(event)
            if event.get("request_id") == rid and event.get("type") == "complete":
                stats = event.get("stats") or {}
                report["runs"][name] = {k: stats.get(k) for k in ("generated_tokens", "lfm3_cert_tokens", "lfm3_cert_fallbacks")}
                report["runs"][name]["token_hash"] = event.get("token_hash")
                break
    g, p = report["runs"]["greedy"], report["runs"]["presence"]
    ok = (isinstance(g["lfm3_cert_tokens"], int) and g["lfm3_cert_tokens"] > 0
          and p["lfm3_cert_tokens"] is None and p["lfm3_cert_fallbacks"] is None
          and p["generated_tokens"] and g["generated_tokens"])
    report["status"] = "passed" if ok else "failed"
finally:
    process.close()
    directory.cleanup()
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
sys.exit(0 if report["status"] == "passed" else 1)
