"""Compare shipped Flash modes and existing opt-ins, in forward/reverse order."""

import argparse
import hashlib
import json
import math
import os
import platform
import runpy
import statistics
import subprocess
from pathlib import Path

bridge = runpy.run_path(str(Path(__file__).with_name("benchmark-flash-bridge.py")))
PROFILES = (
    "shipped-off",
    "shipped-mtp1",
    "shipped-auto",
    "shipped-precap",
    "tuned-auto",
    "tuned-b2",
    "tuned-b3",
    "tuned-b4",
    "tuned-precap",
)


def configuration(name):
    if name not in PROFILES:
        raise ValueError("unknown Flash profile")
    # Inherited experiments must not silently change the shipped baseline.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLXL3_")}
    tuned = name.startswith("tuned-")
    if tuned:
        env.update(
            MLXL3_ALLOCATOR_CACHE_MIB="64",
            MLXL3_SSD_GPU_HOT="1",
            MLXL3_SSD_GPU_HOT_WIDE="1",
            MLXL3_FLASH_MULTI_TOKEN="1",
            MLXL3_SSD_UNIQUE_COHORT="1",
        )
    width = {"tuned-b2": 1, "tuned-b3": 2, "tuned-b4": 3}.get(name)
    if width is not None:
        env.update(MLXL3_FLASH_DRAFT_ADAPTIVE="0", MLXL3_FLASH_DRAFT_WIDTH=str(width))
    if name.endswith("-precap"):
        env["MLXL3_FLASH_DRAFT_PRECAP"] = "1"
    return env, {
        "mtp": name == "shipped-mtp1",
        "auto_cache": not tuned,
        "cache_gib": 3.0,
    }


def number(value: float, *, positive: bool = False) -> float:
    """Validate times and targets.

    raises: ValueError
    post: math.isfinite(__return__) and __return__ == value
    post: __return__ > 0 if positive else __return__ >= 0
    """
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError("invalid finite nonnegative measurement")
    return value


def counter(value: int) -> int:
    """Validate exact counters without accepting bools.

    raises: ValueError
    post: type(__return__) is int and __return__ >= 0 and __return__ == value
    """
    if type(value) is not int or value < 0:
        raise ValueError("invalid nonnegative integer counter")
    return value


def entry_metrics(entry, name, tokens):
    _, options = configuration(name)
    dflash = name not in ("shipped-off", "shipped-mtp1")
    bridge["validate"](
        {
            "stats": entry["stats"],
            "token_hash": entry["token_hash"],
            "cache_context": entry["text"],
        },
        entry["generation_mode"],
        options["mtp"],
        tokens,
        dflash,
    )
    stats = entry["stats"]
    if (
        stats.get("context_limit") != 4096
        or not 0 < counter(stats.get("prompt_tokens")) < 4096 - tokens
    ):
        raise ValueError("context clipping invalidates the width comparison")
    result = {
        "decode_tps": stats["decode_tps"],
        "ttft_seconds": stats["ttft_seconds"],
        "end_to_end_ttft_seconds": number(stats["end_to_end_ttft_seconds"], positive=True),
        "total_seconds": number(stats["elapsed_seconds"], positive=True),
        "memory": stats.get("memory"),
    }
    memory = result["memory"]
    if memory is not None:
        if not isinstance(memory, dict):
            raise ValueError("invalid memory snapshot")
        for value in memory.values():
            if value is not None:
                counter(value)
    if not dflash:
        return result
    profile = stats["dflash_profile"]
    blocks = counter(stats["dflash_blocks"])
    proposed = counter(stats["dflash_proposed_tokens"])
    accepted = counter(stats["dflash_accepted_tokens"])
    if blocks == 0 or accepted > proposed:
        raise ValueError("vacuous or invalid speculative generation")
    width = {
        "tuned-b2": 1,
        "tuned-b3": 2,
        "tuned-b4": 3,
        "tuned-precap": 3,
        "shipped-precap": 3,
    }.get(name, 7)
    if counter(entry["generation_mode"].get("dflash_proposals")) != width:
        raise ValueError("requested draft width was not used")
    for key, total in (("drafted_widths", "drafted_rows"), ("verified_widths", "verified_rows")):
        hist = profile[key]
        if not isinstance(hist, list) or len(hist) != 8:
            raise ValueError("missing width distribution")
        counts = [counter(n) for n in hist]
        if (
            sum(counts) != blocks
            or sum((i + 1) * n for i, n in enumerate(counts)) != counter(profile[total])
            or any(counts[width + 1 :])
            or (key == "drafted_widths" and counts[0])
        ):
            raise ValueError("inconsistent computed/verified draft work")
    if profile["verified_rows"] != proposed + blocks:
        raise ValueError("verification counters disagree")
    if profile["drafted_rows"] < profile["verified_rows"]:
        raise ValueError("more verified rows than computed rows")
    if name in ("tuned-b2", "tuned-b3", "tuned-b4") and (
        profile["drafted_widths"] != profile["verified_widths"]
        or not profile["drafted_widths"][width]
        or not width * blocks - width * (width - 1) // 2 <= proposed <= width * blocks
    ):
        raise ValueError("fixed-width profile actually used adaptation")
    expert_bytes = counter(profile["expert_demand_bytes"])
    if expert_bytes % 1_248_012:
        raise ValueError("expert demand is not a whole number of logical records")
    times = [
        number(profile[k])
        for k in ("draft_seconds", "verification_seconds", "commit_cache_seconds")
    ]
    if times[1] <= 0 or sum(times) > number(stats["dflash_block_seconds"], positive=True) + 1e-6:
        raise ValueError("inconsistent inclusive block durations")
    result.update(
        drafted_rows_per_decode_token=profile["drafted_rows"] / (tokens - 1),
        verified_rows_per_decode_token=profile["verified_rows"] / (tokens - 1),
        expert_demand_bytes_per_decode_token=expert_bytes / (tokens - 1),
        acceptance_percent=100 * accepted / proposed if proposed else 0,
        verification_ms_per_decode_token=1000 * times[1] / (tokens - 1),
        draft_ms_per_decode_token=1000 * times[0] / (tokens - 1),
        drafted_widths=profile["drafted_widths"],
        verified_widths=profile["verified_widths"],
    )
    return result


def summarize(arms, tokens, repeats, target):
    number(target, positive=True)
    expected = list(PROFILES) + list(reversed(PROFILES))
    if [arm["profile"] for arm in arms] != expected or [a["index"] for a in arms] != list(
        range(len(expected))
    ):
        raise ValueError("incomplete or reordered campaign")
    oracles, values = {}, {}
    for arm in arms:
        report, name = arm["report"], arm["profile"]
        if (
            report.get("status") != "validated"
            or not report.get("reaped")
            or not report.get("reader_joined")
        ):
            raise ValueError("unfinished bridge process")
        for key, count in (("warmup", 1), ("runs", repeats)):
            entries = report[key]
            for prompt in bridge["PROMPTS"]:
                if sum(e["prompt"] == prompt for e in entries) != count:
                    raise ValueError("missing or duplicate prompt measurements")
            if len(entries) != count * len(bridge["PROMPTS"]):
                raise ValueError("foreign prompt measurements")
            for entry in entries:
                metrics = entry_metrics(entry, name, tokens)
                identity = entry["token_hash"], entry["text"]
                reference = oracles.setdefault(entry["prompt"], identity)
                if identity != reference:
                    raise ValueError("token/text divergence between modes")
                if key == "runs":
                    values.setdefault(name, {}).setdefault(entry["prompt"], []).append(metrics)
    summary = {}
    for name, prompts in values.items():
        summary[name] = {}
        for prompt, rows in prompts.items():
            speeds = [r["decode_tps"] for r in rows]
            drift = 100 * (max(speeds) / min(speeds) - 1)
            summary[name][prompt] = {
                "measurements": rows,
                "median_decode_tps": statistics.median(speeds),
                "spread_percent": drift,
                "stable_within_3_percent": drift <= 3,
                "target_reached": min(speeds) >= target and drift <= 3,
            }
    return summary


def conditions():
    result = {"platform": platform.platform()}
    for command in (("pmset", "-g", "batt"), ("pmset", "-g", "therm"), ("vm_stat",)):
        try:
            result[command[0] + " " + command[-1]] = subprocess.check_output(
                command, text=True, timeout=5, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[command[0] + " " + command[-1]] = f"unavailable: {error}"
    return result


def run(engine, model, draft, output, *, tokens=128, repeats=1, target=15.0, runner=None):
    if type(tokens) is not int or tokens < 8 or type(repeats) is not int or repeats < 1:
        raise ValueError("tokens >= 8 and repeats >= 1 required")
    number(target, positive=True)
    if not engine.is_file() or not model.is_dir() or not draft.is_dir():
        raise ValueError("engine, target and draft must exist")
    output.mkdir(parents=True, exist_ok=False)
    identity = hashlib.sha256(engine.read_bytes()).hexdigest()
    report = {
        "status": "incomplete",
        "parity": False,
        "binary_sha256": identity,
        "tokens": tokens,
        "repeats": repeats,
        "target_tps": target,
        "model": str(model.resolve()),
        "draft": str(draft.resolve()),
        "arms": [],
    }

    def save():
        temporary = output / "campaign.tmp"
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temporary.replace(output / "campaign.json")

    save()
    try:
        for index, name in enumerate((*PROFILES, *reversed(PROFILES))):
            env, options = configuration(name)
            before = conditions()
            arm = {
                "index": index,
                "profile": name,
                "before": before,
                "status": "incomplete",
                "output": str((output / f"{index:02d}-{name}").resolve()),
            }
            report["arms"].append(arm)
            save()
            if "AC Power" not in before.get("pmset batt", ""):
                raise RuntimeError("campaign requires observed AC power before each arm")
            if hashlib.sha256(engine.read_bytes()).hexdigest() != identity:
                raise ValueError("engine changed during the campaign")
            child = (runner or bridge["run"])(
                engine,
                model,
                output / f"{index:02d}-{name}",
                tokens=tokens,
                repeats=repeats,
                timeout=300,
                environment=env,
                dflash_draft=draft if name not in ("shipped-off", "shipped-mtp1") else None,
                **options,
            )
            arm["after"] = conditions()
            if (
                child.get("status") != "validated"
                or not child.get("reaped")
                or not child.get("reader_joined")
            ):
                raise ValueError("unfinished bridge process")
            for key, count in (("warmup", 1), ("runs", repeats)):
                if len(child[key]) != 2 * count or any(
                    sum(e["prompt"] == prompt for e in child[key]) != count
                    for prompt in bridge["PROMPTS"]
                ):
                    raise ValueError("missing or duplicate prompt measurements")
            if (
                child["binary_sha256"] != identity
                or hashlib.sha256(engine.read_bytes()).hexdigest() != identity
            ):
                raise ValueError("engine identity differs")
            if "AC Power" not in arm["after"].get("pmset batt", ""):
                raise RuntimeError("AC power changed during arm")
            # Check every completed arm immediately, including warmup, before the next model.
            reference = {}
            completed_reports = [(a["profile"], a["report"]) for a in report["arms"][:-1]]
            for profile_name, completed in [*completed_reports, (name, child)]:
                for entry in completed["warmup"] + completed["runs"]:
                    entry_metrics(entry, profile_name, tokens)
                    actual = entry["token_hash"], entry["text"]
                    if reference.setdefault(entry["prompt"], actual) != actual:
                        raise ValueError("token/text divergence between modes")
            arm.update(report=child, status="complete")
            save()
        report["summary"] = summarize(report["arms"], tokens, repeats, target)
        report.update(status="completed", parity=True)
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            parity=False,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("engine", "model", "draft", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-tps", type=float, default=15)
    args = parser.parse_args()
    run(
        args.engine,
        args.model,
        args.draft,
        args.output,
        tokens=args.tokens,
        repeats=args.repeats,
        target=args.target_tps,
    )
