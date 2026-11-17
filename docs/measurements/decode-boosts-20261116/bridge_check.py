#!/usr/bin/env python3
"""BRIDGE-1 (decode boosts): run the real `mlxl3-rs bridge` once per flag set.

Each configuration is a fresh bridge process (the flags are read from the
environment). Correctness only: normal completion, token hashes, counters.
Writes one JSON report with an explicit status; a timeout, a bridge error or a
missing `complete` event is recorded as a failure, never as a pass.

usage: bridge_check.py {lfm3,lfm3-lookup,q36,q36-q3,lfm3-defaults,q36-defaults}
       BASE_BINARY NEW_BINARY MODEL OUT.json [MTP_HEAD]

The `*-defaults` campaigns also check which counters each configuration
reports (`expect`: counter name -> present or absent).
"""
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    ("prose", "Explique en cinq phrases comment fonctionne la photosynthèse."),
    (
        "copy",
        "Recopie exactement cette fonction Python en renommant `total` en `somme`, "
        "sans rien ajouter :\n\n```python\ndef moyenne(valeurs):\n    total = 0\n"
        "    for v in valeurs:\n        total += v\n    if not valeurs:\n        return 0\n"
        "    return total / len(valeurs)\n\n\ndef variance(valeurs):\n    m = moyenne(valeurs)\n"
        "    total = 0\n    for v in valeurs:\n        total += (v - m) ** 2\n"
        "    return total / len(valeurs)\n```",
    ),
]
GREEDY = {"temperature": 0.0, "top_k": 0, "repetition_penalty": 1.0}
DESKTOP = {"temperature": 0.2, "top_k": 80, "repetition_penalty": 1.05}
LFM3_PRESET = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1}
MAX_TOKENS = 160
READY_SECONDS = 900
# Options removed from the inherited environment before each configuration.
OPTION_NAMES = (
    "MLXL3_LFM3_CERT_HEAD", "MLXL3_LFM3_CERT_SAMPLE", "MLXL3_LFM3_CERT_SCREEN",
    "MLXL3_LFM3_CERT_CHECK", "MLXL3_LFM3_LOOKUP", "MLXL3_LFM3_EMBED_LAZY",
    "MLXL3_TOPK_READBACK", "MLXL3_MTP_SAMPLED_DRAFT",
)
EVENT_SECONDS = 300


class BridgeError(RuntimeError):
    pass


def run_config(binary, model, env_extra, sampler, extra_request, stderr_path):
    env = {k: v for k, v in os.environ.items() if k not in OPTION_NAMES}
    env.update(env_extra)
    command = [str(binary), "bridge", str(model), "--context-length", "4096"]
    with open(stderr_path, "w") as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, bufsize=1, env=env,
        )
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def read_event(seconds):
            deadline = time.monotonic() + seconds
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not selector.select(timeout=left):
                    raise BridgeError(f"no event within {seconds} s")
                line = process.stdout.readline()
                if not line:
                    raise BridgeError(f"bridge exited (code {process.poll()})")
                try:
                    return json.loads(line)
                except json.JSONDecodeError as error:
                    raise BridgeError(f"invalid event line: {line[:200]!r}") from error

        results = []
        try:
            while True:
                event = read_event(READY_SECONDS)
                if event.get("type") == "error":
                    raise BridgeError(f"bridge error before ready: {event.get('message')}")
                if event.get("type") == "ready":
                    break
            for index, (name, prompt) in enumerate(PROMPTS):
                request_id = f"b1-{index}"
                request = {
                    "type": "generate", "request_id": request_id,
                    "conversation_id": f"b1-{index}-{time.time_ns()}",
                    "reuse_prompt_cache": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS, **sampler, **extra_request,
                }
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                text, complete = "", None
                while complete is None:
                    event = read_event(EVENT_SECONDS)
                    if event.get("request_id") not in (None, request_id):
                        continue
                    kind = event.get("type")
                    if kind == "delta":
                        text += event.get("text", "")
                    elif kind == "error":
                        raise BridgeError(f"{name}: {event.get('message')}")
                    elif kind == "complete":
                        complete = event
                stats = complete.get("stats", {})
                results.append({
                    "prompt": name,
                    "token_hash": complete.get("token_hash"),
                    "stop_reason": complete.get("stop_reason"),
                    "generated_tokens": stats.get("generated_tokens"),
                    "decode_tps_indicative": stats.get("decode_tps"),
                    "memory": stats.get("memory"),
                    "counters": {
                        key: value for key, value in stats.items()
                        if key.startswith(("lfm3_", "mtp_", "topk_readback"))
                    },
                    "text": text,
                    "text_chars": len(text),
                })
            return {"status": "completed", "results": results}
        except BridgeError as error:
            return {"status": "failed", "error": str(error), "results": results}
        finally:
            if process.poll() is None:
                try:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=20)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()


def campaign(kind, mtp_head):
    mtp = {"mtp": True, "mtp_head_path": str(mtp_head), "mtp_depth": 2}
    cert = ("lfm3_cert_tokens", "lfm3_cert_fallbacks")
    on = lambda *names: {n: True for n in names}  # noqa: E731
    off = lambda *names: {n: False for n in names}  # noqa: E731
    if kind == "lfm3-defaults":
        check = {"MLXL3_LFM3_CERT_CHECK": "1"}
        return [
            ("base-greedy", "base", {}, GREEDY, {}, {}),
            ("new-greedy-default", "new", {}, GREEDY, {},
             {**on(*cert), **off("topk_readback_rows")}),
            ("new-greedy-default-check", "new", check, GREEDY, {},
             on(*cert, "lfm3_cert_check_mismatches")),
            ("new-greedy-screen-int8", "new", {"MLXL3_LFM3_CERT_SCREEN": "int8"}, GREEDY, {},
             on(*cert)),
            ("new-greedy-cert-off", "new", {"MLXL3_LFM3_CERT_HEAD": "0"}, GREEDY, {},
             off(*cert, "topk_readback_rows")),
            ("new-lfm3preset-default", "new", {}, LFM3_PRESET, {}, on(*cert)),
            ("new-lfm3preset-default-check", "new", check, LFM3_PRESET, {},
             on(*cert, "lfm3_cert_check_mismatches")),
            ("new-desktop-default", "new", {}, DESKTOP, {}, on(*cert)),
            ("new-desktop-default-check", "new", check, DESKTOP, {},
             on(*cert, "lfm3_cert_check_mismatches")),
            ("new-desktop-cert-sample-off", "new", {"MLXL3_LFM3_CERT_SAMPLE": "0"}, DESKTOP, {},
             {**off(*cert), **on("topk_readback_rows")}),
            ("new-desktop-all-off", "new",
             {"MLXL3_LFM3_CERT_SAMPLE": "0", "MLXL3_TOPK_READBACK": "0"}, DESKTOP, {},
             off(*cert, "topk_readback_rows")),
        ]
    if kind == "q36-defaults":
        return [
            ("base-greedy", "base", {}, GREEDY, {}, {}),
            ("new-greedy-default", "new", {}, GREEDY, {}, off(*cert, "topk_readback_rows")),
            ("new-desktop-default", "new", {}, DESKTOP, {},
             {**on("topk_readback_rows"), **off(*cert)}),
            ("new-desktop-readback-off", "new", {"MLXL3_TOPK_READBACK": "0"}, DESKTOP, {},
             off("topk_readback_rows")),
            ("new-lfm3preset-default", "new", {}, LFM3_PRESET, {}, on("topk_readback_rows")),
            # MTP rows stay opt-in: rows counted here come from `select_token` only.
            ("new-desktop-mtp2-default", "new", {}, DESKTOP, mtp, on("mtp_blocks")),
            ("new-desktop-mtp2-readback-on", "new", {"MLXL3_TOPK_READBACK": "1"}, DESKTOP, mtp,
             on("mtp_blocks", "topk_readback_rows")),
        ]
    if kind == "lfm3":
        return [
            ("base-greedy", "base", {}, GREEDY, {}),
            ("new-greedy-default", "new", {}, GREEDY, {}),
            ("new-greedy-int4-check", "new",
             {"MLXL3_LFM3_CERT_SCREEN": "int4", "MLXL3_LFM3_CERT_CHECK": "1"}, GREEDY, {}),
            ("new-greedy-lookup-v2", "new", {"MLXL3_LFM3_LOOKUP": "v2"}, GREEDY, {}),
            ("new-greedy-embed-lazy", "new", {"MLXL3_LFM3_EMBED_LAZY": "1"}, GREEDY, {}),
            ("new-lfm3preset-cert-sample-check", "new",
             {"MLXL3_LFM3_CERT_SAMPLE": "1", "MLXL3_LFM3_CERT_CHECK": "1"}, LFM3_PRESET, {}),
            ("new-desktop-cert-sample-check", "new",
             {"MLXL3_LFM3_CERT_SAMPLE": "1", "MLXL3_LFM3_CERT_CHECK": "1"}, DESKTOP, {}),
            ("new-desktop-topk-readback", "new", {"MLXL3_TOPK_READBACK": "1"}, DESKTOP, {}),
        ]
    if kind == "lfm3-lookup":
        return [
            ("base-greedy", "base", {}, GREEDY, {}),
            ("base-greedy-lookup-v1", "base", {"MLXL3_LFM3_LOOKUP": "1"}, GREEDY, {}),
            ("new-greedy-lookup-v1", "new", {"MLXL3_LFM3_LOOKUP": "1"}, GREEDY, {}),
            ("new-greedy-lookup-v2", "new", {"MLXL3_LFM3_LOOKUP": "v2"}, GREEDY, {}),
            ("new-greedy-lookup-v2-nocert", "new",
             {"MLXL3_LFM3_LOOKUP": "v2", "MLXL3_LFM3_CERT_HEAD": "0"}, GREEDY, {}),
        ]
    if kind == "q36-q3":
        return [
            ("base-greedy", "base", {}, GREEDY, {}),
            ("new-greedy-default", "new", {}, GREEDY, {}),
            ("new-desktop-mtp2-sampled-draft", "new", {"MLXL3_MTP_SAMPLED_DRAFT": "1"}, DESKTOP, mtp),
            ("new-desktop-mtp2-sampled-draft-readback", "new",
             {"MLXL3_MTP_SAMPLED_DRAFT": "1", "MLXL3_TOPK_READBACK": "1"}, DESKTOP, mtp),
            ("new-lfm3preset-mtp2-sampled-draft", "new", {"MLXL3_MTP_SAMPLED_DRAFT": "1"},
             LFM3_PRESET, mtp),
        ]
    return [
        ("base-greedy", "base", {}, GREEDY, {}),
        ("new-greedy-default", "new", {}, GREEDY, {}),
        ("new-desktop-topk-readback", "new", {"MLXL3_TOPK_READBACK": "1"}, DESKTOP, {}),
        ("new-desktop-mtp2", "new", {}, DESKTOP, mtp),
        ("new-desktop-mtp2-sampled-draft", "new", {"MLXL3_MTP_SAMPLED_DRAFT": "1"}, DESKTOP, mtp),
        ("new-desktop-mtp2-sampled-draft-readback", "new",
         {"MLXL3_MTP_SAMPLED_DRAFT": "1", "MLXL3_TOPK_READBACK": "1"}, DESKTOP, mtp),
    ]


def main():
    kind, base, new, model, out = sys.argv[1:6]
    mtp_head = sys.argv[6] if len(sys.argv) > 6 else None
    kinds = ("lfm3", "lfm3-lookup", "q36", "q36-q3", "lfm3-defaults", "q36-defaults")
    if kind not in kinds or (kind.startswith("q36") and not mtp_head):
        sys.exit(__doc__)
    binaries = {"base": Path(base), "new": Path(new)}
    out = Path(out)
    report = {"campaign": kind, "model": model, "max_tokens": MAX_TOKENS,
              "status": "in_progress", "configs": {}}
    for label, which, env, sampler, extra, *expect in campaign(kind, mtp_head):
        expect = expect[0] if expect else {}
        started = time.monotonic()
        entry = run_config(binaries[which], model, env, sampler, extra,
                           out.with_name(f"{out.stem}-{label}.stderr"))
        entry.update(binary=which, env=env, sampler=sampler, expect=expect,
                     mtp=bool(extra), wall_seconds=round(time.monotonic() - started, 1))
        report["configs"][label] = entry
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(label, entry["status"], entry.get("error", ""), flush=True)
    configs = report["configs"]
    failures = [label for label, entry in configs.items() if entry["status"] != "completed"]
    reference = configs["base-greedy"]
    hashes = [r["token_hash"] for r in reference["results"]] if not failures else None
    checks = {}
    for label, entry in configs.items():
        if entry["sampler"] is GREEDY and label != "base-greedy" and hashes:
            got = [r["token_hash"] for r in entry["results"]]
            checks[label] = {"greedy_hashes_equal_base": got == hashes
                             and all(isinstance(h, str) and h for h in got)}
        mismatches = [r["counters"].get("lfm3_cert_check_mismatches")
                      for r in entry["results"]]
        if any(m is not None for m in mismatches):
            checks.setdefault(label, {})["check_mismatches"] = mismatches
        ended = [r["stop_reason"] for r in entry["results"]]
        checks.setdefault(label, {})["stop_reasons"] = ended
        checks[label]["tokens"] = [r["generated_tokens"] for r in entry["results"]]
        # Every prompt must report (or omit / leave null) each expected counter.
        checks[label]["expect_ok"] = bool(entry["results"]) and all(
            (r["counters"].get(name) is not None) == present
            for r in entry["results"] for name, present in entry["expect"].items()
        )
    report["checks"] = checks
    ok = not failures and all(
        c.get("greedy_hashes_equal_base", True)
        and all(m in (None, 0) for m in c.get("check_mismatches", []))
        and all(isinstance(t, int) and t > 0 for t in c["tokens"])
        and c["expect_ok"]
        for c in checks.values()
    )
    report["status"] = "passed" if ok else "failed"
    report["failures"] = failures
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("STATUS", report["status"], failures)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
