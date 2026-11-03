"""Score frozen router38 logits in Float64 and apply its preregistered quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from eval_lfm25_quality import losses
from lfm25_router_plan import validate_plan

ALPHAS = ("0.25", "0.5", "1")
VALIDATION_NAMES = tuple(f"validation-{i:02d}" for i in range(12))


def artifact(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def score_case(actual_path, teacher_path, labels: list[int], *, vocabulary=128000, chunk_size=32):
    if (
        type(vocabulary) is not int
        or not 0 < vocabulary <= 2**32
        or type(chunk_size) is not int
        or not 1 <= chunk_size <= 32
        or not isinstance(labels, list)
        or not labels
        or any(type(token) is not int or not 0 <= token < vocabulary for token in labels)
    ):
        raise ValueError("Expected nonempty valid labels and chunks of at most32 rows")
    actual = np.load(actual_path, mmap_mode="r", allow_pickle=False)
    teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    expected = (len(labels), vocabulary)
    if any(
        row.shape != expected or row.dtype not in (np.dtype("float16"), np.dtype("float32"))
        for row in (actual, teacher)
    ):
        raise ValueError("Expected complete F16/F32 logits with the exact token/vocabulary shape")
    total = np.zeros(3, dtype=np.float64)
    matches = 0
    for start in range(0, len(labels), chunk_size):
        stop = min(start + chunk_size, len(labels))
        x, y = actual[start:stop], teacher[start:stop]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite logits in a scored chunk")
        for offset, label in enumerate(labels[start:stop]):
            total += losses(x[offset], y[offset], label)
            matches += int(x[offset].argmax() == y[offset].argmax())
    nll, teacher_nll, kl = (float(value / len(labels)) for value in total)
    try:
        ppl, teacher_ppl = math.exp(nll), math.exp(teacher_nll)
    except OverflowError as error:
        raise ValueError("Nonfinite perplexity") from error
    if not all(math.isfinite(v) for v in (nll, teacher_nll, kl, ppl, teacher_ppl)):
        raise ValueError("Nonfinite quality metrics")
    return {
        "status": "complete",
        "tokens": len(labels),
        "kl": kl,
        "nll": nll,
        "ppl": ppl,
        "teacher_nll": teacher_nll,
        "teacher_ppl": teacher_ppl,
        "greedy_matches": matches,
        "labels_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(labels)}I", *labels)).hexdigest(),
        "actual": {
            **artifact(Path(actual_path)),
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
        },
        "teacher": {
            **artifact(Path(teacher_path)),
            "shape": list(teacher.shape),
            "dtype": str(teacher.dtype),
        },
    }


def improves_kl(actual: float, baseline: float) -> bool:
    """Require the registered one-percent mean KL improvement, including a strict decrease.

    raises: ValueError
    post: not __return__ or (baseline > 0 and 0 <= actual < baseline)
    """
    if not math.isfinite(actual) or not math.isfinite(baseline) or actual < 0 or baseline < 0:
        raise ValueError("Invalid mean KL")
    return baseline > 0 and actual < baseline and actual <= 0.99 * baseline


def domain_gate(
    kl: float, baseline_kl: float, ppl: float, baseline_ppl: float, teacher_ppl: float
) -> bool:
    """Apply both registered domain guards to finite nonnegative metrics.

    raises: ValueError
    post: not __return__ or (kl <= 1.02 * baseline_kl and ppl <= 1.02 * max(baseline_ppl, teacher_ppl))
    """
    if (
        any(not math.isfinite(v) for v in (kl, baseline_kl, ppl, baseline_ppl, teacher_ppl))
        or kl < 0
        or baseline_kl < 0
        or min(ppl, baseline_ppl, teacher_ppl) < 1
    ):
        raise ValueError("Invalid domain metrics")
    return kl <= 1.02 * baseline_kl and ppl <= 1.02 * max(baseline_ppl, teacher_ppl)


def _cases(result) -> dict:
    if isinstance(result, dict) and "status" in result:
        if result["status"] != "complete":
            raise ValueError("Incomplete score report")
        result = result.get("cases")
    if isinstance(result, list):
        if len(result) != 12 or any(not isinstance(row, dict) for row in result):
            raise ValueError("Expected12 validation cases")
        names = [row.get("name", name) for row, name in zip(result, VALIDATION_NAMES, strict=True)]
        if len(set(names)) != 12:
            raise ValueError("Duplicate validation case")
        result = dict(zip(names, result, strict=True))
    if not isinstance(result, dict) or set(result) != set(VALIDATION_NAMES):
        raise ValueError("Missing, extra or misnamed validation cases")
    for row in result.values():
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise ValueError("Incomplete validation case")
        count, matches = row.get("tokens"), row.get("greedy_matches")
        if (
            type(count) is not int
            or count < 1
            or type(matches) is not int
            or not 0 <= matches <= count
        ):
            raise ValueError("Invalid or empty scored positions")
        for field in ("kl", "nll", "ppl", "teacher_nll", "teacher_ppl"):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid quality metric: " + field)
        for nll, ppl in (("nll", "ppl"), ("teacher_nll", "teacher_ppl")):
            if row[ppl] < 1 or not math.isclose(
                math.log(row[ppl]), row[nll], abs_tol=1e-12, rel_tol=1e-12
            ):
                raise ValueError("Inconsistent perplexity and NLL")
        if any(not isinstance(row.get(key), dict) for key in ("actual", "teacher")):
            raise ValueError("Missing logits artifacts")
        if any(
            type(row[key].get("bytes")) is not int or row[key]["bytes"] < 1
            for key in ("actual", "teacher")
        ):
            raise ValueError("Empty logits artifact")
        hashes = [
            row.get("labels_u32le_sha256"),
            row["teacher"].get("sha256"),
            row["actual"].get("sha256"),
        ]
        if any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in hashes
        ):
            raise ValueError("Missing label/teacher SHA evidence")
    return result


def _aggregate(rows) -> dict:
    rows = list(rows)
    tokens = sum(row["tokens"] for row in rows)
    if tokens < 1:
        raise ValueError("Empty quality aggregate")
    means = {
        field: math.fsum(row[field] * row["tokens"] for row in rows) / tokens
        for field in ("kl", "nll", "teacher_nll")
    }
    return {
        "tokens": tokens,
        **means,
        "ppl": math.exp(means["nll"]),
        "teacher_ppl": math.exp(means["teacher_nll"]),
        "greedy_matches": sum(row["greedy_matches"] for row in rows),
    }


def _domains(cases, domains):
    return {
        domain: _aggregate(
            cases[name]
            for name, label in zip(VALIDATION_NAMES, domains, strict=True)
            if label == domain
        )
        for domain in dict.fromkeys(domains)
    }


def select_candidate(baseline, candidates: dict, domains: list[str]) -> dict:
    """Accept complete reports, validation case maps, or ordered lists of12 score_case results.

    All three registered candidate keys are required; exact KL ties prefer 0.25,0.5,1.
    Labels, teacher SHA/NLL and scored counts must match the baseline case by case.
    """
    if (
        not isinstance(domains, list)
        or len(domains) != 12
        or any(not isinstance(v, str) or not v for v in domains)
        or len(set(domains)) != 6
        or set(Counter(domains).values()) != {2}
        or not isinstance(candidates, dict)
        or set(candidates) != set(ALPHAS)
    ):
        raise ValueError("Expected six paired domains and registered candidates")
    baseline = _cases(baseline)
    base_domains, overall = _domains(baseline, domains), _aggregate(baseline.values())
    result = {
        "status": "complete",
        "baseline": overall,
        "baseline_domains": base_domains,
        "candidates": {},
        "selected": None,
    }
    for alpha in ALPHAS:
        if alpha not in candidates:
            continue
        cases = _cases(candidates[alpha])
        for name in VALIDATION_NAMES:
            row, before = cases[name], baseline[name]
            if (
                row["tokens"] != before["tokens"]
                or row["labels_u32le_sha256"] != before["labels_u32le_sha256"]
                or row["teacher"]["sha256"] != before["teacher"]["sha256"]
                or row["teacher_nll"] != before["teacher_nll"]
            ):
                raise ValueError("Candidate scoring protocol differs from baseline: " + name)
        summary, per_domain = _aggregate(cases.values()), _domains(cases, domains)
        failed = [
            domain
            for domain, row in per_domain.items()
            if not domain_gate(
                row["kl"],
                base_domains[domain]["kl"],
                row["ppl"],
                base_domains[domain]["ppl"],
                base_domains[domain]["teacher_ppl"],
            )
        ]
        eligible = improves_kl(summary["kl"], overall["kl"]) and not failed
        result["candidates"][alpha] = {
            "overall": summary,
            "domains": per_domain,
            "eligible": eligible,
            "failed_domains": failed,
        }
        previous = result["selected"]
        if eligible and (
            previous is None or summary["kl"] < result["candidates"][previous]["overall"]["kl"]
        ):
            result["selected"] = alpha
    return result


def run_score(plan, teacher_dir, candidate_dir, phase="validation", *, vocabulary=128000) -> dict:
    if not isinstance(plan, dict) or plan.get("status") != "frozen":
        raise ValueError("Expected a frozen router38 plan")
    validate_plan(plan)
    if phase == "validation":
        section = plan["validation"]
        inputs, labels, domains = section["input_ids"], section["labels"], section["domains"]
        if (
            len(inputs) != 12
            or len(labels) != 12
            or len(domains) != 12
            or section.get("shape") != [12, 1023]
            or any(len(x) != 1023 or len(y) != 1023 for x, y in zip(inputs, labels, strict=True))
        ):
            raise ValueError("Incomplete frozen validation windows")
        cases = dict(zip(VALIDATION_NAMES, labels, strict=True))
    elif phase == "final":
        final = plan["final"]
        if not isinstance(final, dict) or set(final) != {"chat-holdout", "cjk-holdout"}:
            raise ValueError("Incomplete final holdouts")
        if any(
            not isinstance(row, dict)
            or len(row.get("input_ids", [])) != 1024
            or len(row.get("labels", [])) != 1024
            for row in final.values()
        ):
            raise ValueError("Expected1024 inputs and labels per final holdout")
        cases = {name: row["labels"] for name, row in final.items()}
        domains = None
    else:
        raise ValueError("Unknown router38 scoring phase")
    plan_sha = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    captures = {}
    for role, folder in (("actual", candidate_dir), ("teacher", teacher_dir)):
        capture = _read_json(Path(folder) / "report.json")
        if (
            not isinstance(capture, dict)
            or capture.get("status") != "complete"
            or capture.get("plan_sha256") != plan_sha
            or capture.get("kind") != ("teacher" if role == "teacher" else "quantized")
            or not isinstance(capture.get("files"), dict)
        ):
            raise ValueError("Incomplete or mismatched actual capture manifest: " + role)
        for name, labels in cases.items():
            record = capture["files"].get(name + ".npy")
            if (
                not isinstance(record, dict)
                or record.get("shape") != [len(labels), vocabulary]
                or record.get("storage_dtype") not in ("float16", "float32")
            ):
                raise ValueError("Missing or partial captured logits: " + name)
        captures[role] = capture
    report = {"status": "running", "phase": phase, "cases": {}, "plan_sha256": plan_sha}
    for name, labels in cases.items():
        report["cases"][name] = score_case(
            Path(candidate_dir) / f"{name}.npy",
            Path(teacher_dir) / f"{name}.npy",
            labels,
            vocabulary=vocabulary,
        )
        for role in ("actual", "teacher"):
            expected = captures[role]["files"][name + ".npy"]
            if expected["storage_dtype"] != report["cases"][name][role]["dtype"] or any(
                report["cases"][name][role][key] != expected.get(key) for key in ("sha256", "bytes")
            ):
                raise ValueError("Captured logits artifact changed: " + role + "/" + name)
    report["overall"] = _aggregate(report["cases"].values())
    report["domains"] = _domains(report["cases"], domains) if domains is not None else {}
    report["status"] = "complete"
    return report


def _read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _atomic_report(output: Path, report: dict):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            handle.flush()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--teacher-dir", type=Path, required=True)
    score.add_argument("--candidate-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--phase", choices=("validation", "final"), default="validation")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x"):
        pass
    report = {"status": "running", "cases": {}, "publication": False}
    _atomic_report(args.output, report)
    try:
        report = run_score(_read_json(args.plan), args.teacher_dir, args.candidate_dir, args.phase)
        report["publication"] = False
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        _atomic_report(args.output, report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
