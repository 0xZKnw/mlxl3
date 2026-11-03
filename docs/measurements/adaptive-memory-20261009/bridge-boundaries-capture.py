import json
import os
import sys
import tempfile
import time
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").is_file())
sys.path[:0] = [str(root / "scripts"), str(root / "benchmarks")]
from benchmark_idle_memory import exchange, quality
from native_json_process import JsonProcess

D = root / "docs/measurements/adaptive-memory-20261009"
BIN = root / "target/release/mlxl3-rs"
MODEL = "/Users/henko/Library/Application Support/io.mlxl3.desktop/Models/Qwen3.8-27B-exl3-6a9ca9d0"
HEAD = "/Users/henko/Library/Application Support/io.mlxl3.desktop/Drafts/Qwen3.8-27B-MTP-4bit"
report = {"status": "running", "parity": False, "cases": [], "reaped": []}


def save():
    (D / "bridge-boundaries-capture.json").write_text(json.dumps(report, indent=2) + "\n")


def request(name, budget=8, reuse=True, depth=2, long=True):
    return {
        "type": "generate",
        "request_id": name,
        "conversation_id": "isolated-memory-check",
        "messages": [
            {
                "role": "user",
                "content": ("Explain this table.\n" + "1 2 3 4 5 6 7 8 9 10\n" * 30)
                if long
                else "Write a Python matrix multiplication function.",
            }
        ],
        "max_tokens": budget,
        "temperature": 0,
        "top_k": 1,
        "repetition_penalty": 1,
        "mtp": True,
        "mtp_depth": depth,
        "mtp_head_path": HEAD,
        "reuse_prompt_cache": reuse,
    }


def await_ready(child):
    end = time.monotonic() + 120
    while True:
        event = child.receive(end)
        if event.get("type") == "error":
            raise ValueError(event)
        if event.get("type") == "ready":
            assert event["context_memory"]["draft_bytes_per_token"] == 4096
            return event


base = os.environ.copy()
base.update(MLXL3_EMBEDDINGS_PACKED="1", MLXL3_ALLOCATOR_CACHE_MIB="0")
base.pop("MLXL3_MTP_ADAPTIVE", None)
for key in (
    "MLXL3_METAL_CAPTURE_PATH",
    "MLXL3_METAL_CAPTURE_REQUEST",
    "MLXL3_PROMPT_CACHE_MIB",
    "MLXL3_CACHE_COMPACTION",
):
    base.pop(key, None)
save()
try:
    with tempfile.TemporaryDirectory(prefix="memory-boundaries-") as temporary:
        command = [
            str(BIN),
            "--registry",
            str(Path(temporary) / "models.json"),
            "bridge",
            MODEL,
            "--context-length",
            "4096",
        ]
        for value in ["0", "bad", "-1", "4097", ""]:
            env = {**base, "MLXL3_PROMPT_CACHE_MIB": value}
            with (
                (D / f"bridge-cache-budget-{value or 'empty'}.stderr").open("w+") as log,
                JsonProcess(command, env=env, stderr=log) as child,
            ):
                await_ready(child)
                outputs = [
                    exchange(child, request(f"{value}-cold")),
                    exchange(child, request(f"{value}-warm")),
                ]
                if value == "0":
                    qs = [quality(x) for x in outputs]
                    assert all(q["stats"]["cached_prompt_tokens"] == 0 for q in qs)
                    assert (
                        qs[0]["token_hash"] == qs[1]["token_hash"]
                        and qs[0]["text_sha256"] == qs[1]["text_sha256"]
                    )
                else:
                    assert all(
                        x["type"] == "error" and "MLXL3_PROMPT_CACHE_MIB" in x["message"]
                        for x in outputs
                    )
                recovery = exchange(child, request(f"{value}-recovery", reuse=False))
                q = quality(recovery)
                assert (
                    q["stats"]["generated_tokens"] == 8 and q["stats"]["cached_prompt_tokens"] == 0
                )
                child.send({"type": "shutdown", "request_id": "shutdown"}, time.monotonic() + 5)
                assert child.process.wait(timeout=10) == 0
                report["cases"].append({"budget": value, "outputs": outputs, "recovery": recovery})
                report["reaped"].append(True)
                save()
        for value in ["bad", "-1", "4097", ""]:
            env = {**base, "MLXL3_ALLOCATOR_CACHE_MIB": value}
            with (D / f"bridge-allocator-budget-{value or 'empty'}.stderr").open("w+") as log:
                with JsonProcess(command, env=env, stderr=log) as child:
                    try:
                        await_ready(child)
                    except (RuntimeError, ValueError):
                        pass
                    else:
                        raise AssertionError("invalid allocator budget accepted")
                    assert child.process.wait(timeout=10) != 0
                log.flush()
                log.seek(0)
                message = log.read()
                assert "MLXL3_ALLOCATOR_CACHE_MIB" in message
                report["cases"].append(
                    {"allocator_budget": value, "rejected": True, "diagnostic": message}
                )
                report["reaped"].append(True)
                save()
        trace = root / "build/adaptive-memory/bridge-profile.gputrace"
        assert not trace.exists()
        env = {
            **base,
            "MTL_CAPTURE_ENABLED": "1",
            "MLXL3_METAL_CAPTURE_REQUEST": "profile",
            "MLXL3_METAL_CAPTURE_PATH": str(trace),
        }
        with (
            (D / "bridge-profile.stderr").open("w+") as log,
            JsonProcess(command, env=env, stderr=log) as child,
        ):
            ready = await_ready(child)
            warm = exchange(child, request("warmup", budget=8, reuse=False, depth=1, long=False))
            quality(warm)
            assert not trace.exists()
            plain = exchange(child, request("plain", budget=2, reuse=False, depth=1, long=False))
            a = quality(plain)
            assert not trace.exists()
            profiled = exchange(
                child,
                request("profile", budget=2, reuse=False, depth=1, long=False),
                timeout=300,
            )
            b = quality(profiled)
            assert trace.exists()
            files = [p for p in trace.rglob("*") if p.is_file()] if trace.is_dir() else [trace]
            size = sum(p.stat().st_size for p in files)
            assert len(files) > 0 and size > 0
            repeat = exchange(child, request("profile", budget=2, reuse=False, depth=1, long=False))
            c = quality(repeat)
            for x in (b, c):
                assert (
                    x["token_hash"] == a["token_hash"]
                    and x["text_sha256"] == a["text_sha256"]
                    and x["stats"]["generated_tokens"] == 2
                )
            assert size == sum(p.stat().st_size for p in files)
            child.send({"type": "shutdown", "request_id": "shutdown"}, time.monotonic() + 5)
            assert child.process.wait(timeout=10) == 0
            report["capture"] = {
                "ready": ready,
                "warmup": warm,
                "plain": plain,
                "profiled": profiled,
                "repeat": repeat,
                "path": str(trace),
                "bytes": size,
                "files": len(files),
                "analysis": "blocked_missing_Xcode",
            }
            report["reaped"].append(True)
            save()
    report.update(status="complete", parity=True)
except BaseException as error:
    report.update(status="failed", parity=False, error=repr(error))
    raise
finally:
    save()
print(
    json.dumps(
        {
            "status": report["status"],
            "cases": len(report["cases"]),
            "capture_bytes": report.get("capture", {}).get("bytes"),
        }
    )
)
