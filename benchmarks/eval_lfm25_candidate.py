"""Replay frozen BF16/V2 quality protocols against one new native checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from eval_lfm25_quality import log_probabilities, native_window


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ppl_gate(candidate: float, v2: float, teacher: float) -> bool:
    if any(not math.isfinite(value) or value <= 0 for value in (candidate, v2, teacher)):
        raise ValueError("Expected three finite positive perplexities")
    return math.log(candidate) <= max(math.log(v2), math.log(teacher)) + math.log(1.02)


def run(
    baseline_path: Path, ids_dir: Path, model: Path, binary: Path, output: Path, timeout: float
):
    if output.exists():
        raise FileExistsError("Preserve the previous quality attempt; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / (output.stem + "-windows")
    artifacts.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "quality_passed": False,
        "publication": False,
        "cases": {},
        "model": str(model),
        "baseline_sha256": file_sha(baseline_path),
        "native_sha256": file_sha(binary),
    }
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        baseline = json.loads(baseline_path.read_text())
        if (
            baseline.get("status") != "complete"
            or baseline["native_sha256"] != report["native_sha256"]
        ):
            raise ValueError("Incomplete baseline or different native binary")
        cases = baseline["cases"]
        if not {"main-8192", "bos-paired"}.issubset(cases):
            raise ValueError("Both primary protocols are required")
        order = ["bos-paired", "main-8192"] + [
            name for name in cases if name not in ("bos-paired", "main-8192")
        ]
        total_windows = sum(len(c["windows"]) for c in cases.values())
        if total_windows <= 0:
            raise ValueError("Empty frozen quality plan")
        done = 0
        for name in order:
            old = cases[name]
            ids = json.loads((ids_dir / f"{name}.json").read_text())
            if (
                not isinstance(ids, list)
                or len(ids) != old["input_tokens"] + 1
                or any(type(t) is not int or not 0 <= t < 128000 for t in ids)
                or hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
                != old["token_sha256"]
            ):
                raise ValueError(f"Frozen token IDs changed: {name}")
            width = old["window"]
            if (
                width <= 0
                or old["input_tokens"] <= 0
                or old["input_tokens"] % width
                or len(old["windows"]) * width != old["input_tokens"]
            ):
                raise ValueError("Incomplete frozen window plan")
            result = report["cases"][name] = {
                "status": "running",
                "windows": [],
                "input_tokens": old["input_tokens"],
                "token_sha256": old["token_sha256"],
                "score_first": old["score_first"],
                "score_last": old["score_last"],
            }
            for number, reference in enumerate(old["windows"]):
                filename = f"teacher-{number:02d}.npy"
                teacher_path = Path(old["teacher_dir"]) / filename
                if file_sha(teacher_path) != old["teacher_files"][filename]:
                    raise ValueError(f"BF16 teacher changed: {name}/{filename}")
                teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
                if (
                    teacher.ndim != 2
                    or teacher.shape[0] != width
                    or not teacher.shape[1]
                    or not np.isfinite(teacher).all()
                ):
                    raise ValueError("Invalid BF16 teacher")
                start = number * width
                tokens, labels = ids[start : start + width], ids[start + 1 : start + width + 1]
                first, last = old["score_first"], old["score_last"]
                if not 0 <= first <= last < width:
                    raise ValueError("Invalid frozen scoring bounds")
                if reference["tokens"] != last - first + 1:
                    raise ValueError("Frozen label count disagrees with its scoring bounds")
                check = math.fsum(
                    -log_probabilities(teacher[i])[labels[i]] for i in range(first, last + 1)
                ) / (last - first + 1)
                if not math.isclose(check, reference["bf16_nll"], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Teacher/label protocol no longer reproduces V2")
                report["stage"] = f"{name}/{number + 1}"
                save()
                measured = native_window(
                    binary,
                    model,
                    tokens,
                    labels,
                    teacher,
                    artifacts / f"{name}-{number:02d}.stderr",
                    timeout,
                    score_first=first,
                    score_last=last,
                )
                if measured["tokens"] != reference["tokens"] or not math.isclose(
                    measured["bf16_nll"], check, rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError("Incomplete or mismatched scored native window")
                result["windows"].append(measured)
                done += 1
                save()
                print(
                    json.dumps(
                        {
                            "case": name,
                            "window": number + 1,
                            "done": done,
                            "total": total_windows,
                            "percent": 100 * done / total_windows,
                            "eta_seconds": report["elapsed_seconds"]
                            * (total_windows - done)
                            / done,
                            "kl": float(measured["kl"]),
                        }
                    ),
                    flush=True,
                )
            count = sum(w["tokens"] for w in result["windows"])
            if count != old["scored_tokens"]:
                raise ValueError("Incomplete scored protocol")
            for label, field in (("native", "nll"), ("bf16", "bf16_nll")):
                nll = math.fsum(float(w[field]) * w["tokens"] for w in result["windows"]) / count
                result[label] = {"mean_nll": nll, "perplexity": math.exp(nll)}
            kl = math.fsum(float(w["kl"]) * w["tokens"] for w in result["windows"]) / count
            if not math.isfinite(kl) or kl < -1e-9:
                raise ValueError("Invalid aggregate KL")
            result.update(
                status="complete",
                scored_tokens=count,
                mean_kl_bf16_to_native=kl,
                greedy_match_fraction=sum(w["greedy_matches"] for w in result["windows"]) / count,
            )
            result["kl_change_from_v2_percent"] = 100 * (kl / old["mean_kl_bf16_to_native"] - 1)
            result["ppl_change_from_v2_percent"] = 100 * (
                result["native"]["perplexity"] / old["native"]["perplexity"] - 1
            )
            result["kl_gate"] = kl <= old["mean_kl_bf16_to_native"] * (
                0.99 if name in ("main-8192", "bos-paired") else 1.02
            )
            result["ppl_gate"] = ppl_gate(
                result["native"]["perplexity"],
                old["native"]["perplexity"],
                result["bf16"]["perplexity"],
            )
            result["regressed_windows_over_20percent"] = [
                i + 1
                for i, (current, previous) in enumerate(
                    zip(result["windows"], old["windows"], strict=True)
                )
                if current["kl"] > previous["kl"] * 1.2
            ]
            save()
        report["quality_passed"] = all(
            c["kl_gate"] and c["ppl_gate"] for c in report["cases"].values()
        )
        report["status"] = "complete"
        report["stage"] = "numerical evaluation complete; functional tests separate"
        save()
        return report
    except BaseException as error:
        report.update(
            status="failed",
            quality_passed=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "ids-dir", "model", "binary", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    run(args.baseline, args.ids_dir, args.model, args.binary, args.output, args.timeout)
