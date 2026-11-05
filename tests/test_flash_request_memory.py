"""Opt-in JSONL integration test on one physical Flash Next engine."""

import asyncio
import json
import os
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(k) for k in ("MLXL3_FLASH_ENGINE", "MLXL3_FLASH_MODEL")),
    reason="requires a guarded physical Flash engine and checkpoint",
)


@pytest.mark.parametrize(
    "limit,message",
    [
        ("0", "positive integer MiB"),
        ("-1", "positive integer MiB"),
        ("18446744073709551615", "positive integer MiB"),
        ("1", "too small for expert banks"),
    ],
)
def test_flash_load_rejects_invalid_and_insufficient_budgets(tmp_path, limit, message):
    env = {**os.environ, "MLXL3_FLASH_REQUEST_MIB": limit, "MLXL3_SSD_CACHE_GIB": "3"}
    result = subprocess.run(
        [
            env["MLXL3_FLASH_ENGINE"],
            "--registry",
            str(tmp_path / "models.json"),
            "bridge",
            env["MLXL3_FLASH_MODEL"],
            "--context-length",
            "4096",
        ],
        input="",
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=30,
    )
    assert result.returncode != 0 and message in result.stderr
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == ["loading"]
    assert "loading layer" not in result.stderr


@pytest.mark.parametrize("tokens", [0, 2])
def test_flash_run_request_budget(tmp_path, tokens):
    env = {**os.environ, "MLXL3_FLASH_REQUEST_MIB": "10000", "MLXL3_SSD_CACHE_GIB": "3"}
    result = subprocess.run(
        [
            env["MLXL3_FLASH_ENGINE"],
            "--registry",
            str(tmp_path / "models.json"),
            "run",
            env["MLXL3_FLASH_MODEL"],
            "--prompt",
            "Bonjour",
            "--max-tokens",
            str(tokens),
        ],
        input="",
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=75,
    )
    if tokens == 0:
        assert result.returncode != 0 and "minimum cache" in result.stderr
    else:
        assert result.returncode == 0 and "2 tokens]" in result.stdout


def test_flash_bridge_admission_cancel_and_reuse(tmp_path):
    async def check():
        env = {**os.environ, "MLXL3_FLASH_REQUEST_MIB": "10000", "MLXL3_SSD_CACHE_GIB": "3"}
        process = await asyncio.create_subprocess_exec(
            env["MLXL3_FLASH_ENGINE"],
            "--registry",
            str(tmp_path / "models.json"),
            "bridge",
            env["MLXL3_FLASH_MODEL"],
            "--context-length",
            "0",
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        async def send(kind, identity, **fields):
            process.stdin.write(
                (json.dumps({"type": kind, "request_id": identity, **fields}) + "\n").encode()
            )
            await process.stdin.drain()

        async def until(kind, identity):
            while line := await process.stdout.readline():
                event = json.loads(line)
                assert isinstance(event["type"], str)
                if event.get("request_id", "") != identity:
                    continue
                if event["type"] == kind:
                    return event
                assert event["type"] not in {"error", "cancelled", "complete"}, event
            raise AssertionError(f"bridge exited waiting for {kind}")

        async def run():
            ready = await until("ready", "")
            assert not ready["mtp_supported"] and not ready["dflash_supported"]
            initial = ready["flash"]["memory"]
            slots = initial["expert_slots_by_layer"]
            assert len(slots) == 48 and min(slots) >= 10
            assert initial["expert_bank_payload_bytes"] == sum(slots) * 1248012
            messages = [
                {
                    "role": "user",
                    "content": "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE.",
                }
            ]
            await send("generate", "large", messages=messages, max_tokens=-1, top_k=1)
            assert "memory budget" in (await until("error", "large"))["message"]
            await send("generate", "stop", messages=messages, max_tokens=128, top_k=1)
            await until("flash_memory_plan", "stop")
            process.send_signal(signal.SIGUSR1)
            await until("cancelled", "stop")
            await send(
                "generate", "short", messages=messages, max_tokens=17, top_k=1, memory_saver=True
            )
            plan = await until("flash_memory_plan", "short")
            assert plan["ar"]["kv_bytes"] == (87 + 17) * 24576
            end = await until("flash_memory_end", "short")
            assert end["positions"] == 87 + 16
            program = end["ledger"]["program_cache"]
            if ready["flash"]["target_policy"]["native_prepare"]:
                assert program["entries"] == 1 and program["creations"] >= 1 and program["hits"] > 0
            state = end["ledger"]["state_backing_bytes"]
            assert state["qsa_kv"] >= end["positions"] * 24576
            assert state["gdn"] >= 36 * 48 * 128 * 128 * 4
            done = await until("complete", "short")
            assert (
                done["token_hash"] == "4b46ba2824c9f47c" and done["stats"]["generated_tokens"] == 17
            )
            readings = {"ping": [], "flash_memory": []}
            for idx, mode in enumerate("ABBA" + "BAAB"):
                probe, expected = (
                    ("ping", "pong") if mode == "A" else ("flash_memory", "flash_memory")
                )
                identity = f"probe-{idx}"
                start = time.perf_counter_ns()
                await send(probe, identity)
                response = await until(expected, identity)
                readings[probe].append((time.perf_counter_ns() - start) / 1e6)
                if mode == "B":
                    live = response["ledger"]
                    assert live["expert_slots_by_layer"] == slots
                    assert live["program_cache"]["entries"] == 0
                    assert (
                        live["program_cache"]["evictions"]
                        == program["evictions"] + program["entries"]
                    )
                    assert live["fixed_weight_backing_bytes"] > 0
                    assert live["mlx_active_other_bytes"] is not None
                    assert live["memory"]["mlx_active_bytes"] == (
                        live["expert_bank_backing_bytes"]
                        + live["fixed_weight_backing_bytes"]
                        + sum(live["state_backing_bytes"].values())
                        + live["mlx_active_other_bytes"]
                    )
            print("FLASH_OBSERVER_MILLISECONDS", json.dumps(readings))
            await send("shutdown", "end")
            await process.wait()
            assert process.returncode == 0

        try:
            await asyncio.wait_for(run(), timeout=180)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()

    asyncio.run(check())
