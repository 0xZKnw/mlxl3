"""77B: qualify observed process pressure, never attribute model swap-outs."""

import argparse
import hashlib
import json
import signal
from pathlib import Path

U64_MAX = (1 << 64) - 1
INT32_MAX = (1 << 31) - 1
IDENTITY = ("process_id", "process_start_abstime")
GAUGES = ("process_footprint_bytes", "process_compressed_logical_bytes")
CUMULATIVE = (
    "process_lifetime_peak_bytes",
    "process_compressed_peak_logical_bytes",
    "process_compressed_lifetime_logical_bytes",
    "process_decompressions",
    "process_ledger_swapins_logical_bytes",
)
FIELDS = IDENTITY + GAUGES + CUMULATIVE
LIMITS = {
    "process_lifetime_peak_bytes": 12_000_000_000,
    "process_compressed_logical_bytes": 512 << 20,
    "process_compressed_peak_logical_bytes": 512 << 20,
    "compressed_lifetime_delta_bytes": 1 << 30,
    "ledger_swapins_delta_logical_bytes": 0,
}


class AnalysisError(ValueError):
    """Carry the partial observations without marking unknown counters as zero."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def u64(value: object) -> int:
    """Read an actual unsigned counter; bool and null are not counters.

    raises: ValueError
    post: type(__return__) is int and 0 <= __return__ <= 18446744073709551615
    """
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise ValueError(f"expected u64 counter, got {value!r}")
    return value


def _samples(document):
    if "events" in document:
        ready = document.get("ready")
        yield "ready", ready.get("memory") if isinstance(ready, dict) else None
        events = document["events"]
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise TypeError(f"invalid event at {index}")
            if event.get("type") == "complete":
                stats = event.get("stats")
                yield f"events[{index}]", stats.get("memory") if isinstance(stats, dict) else None
    else:
        runtime = document.get("runtime")
        yield "ready", runtime.get("memory") if isinstance(runtime, dict) else None
        for group in ("warmup", "runs"):
            entries = document.get(group)
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{group} must be a nonempty list")
            for index, entry in enumerate(entries):
                stats = entry.get("stats") if isinstance(entry, dict) else None
                yield f"{group}[{index}]", stats.get("memory") if isinstance(stats, dict) else None


def analyze(document):
    """Inspect existing checker/benchmark JSON; this does not validate model outputs."""
    report = {
        "status": "failed",
        "qualification": None,
        "limits": LIMITS.copy(),
        "samples": [],
        "maxima": None,
        "final_deltas_since_ready": None,
        "startup_ledger_swapins_logical_bytes": None,
        "startup_already_affected": None,
        "budget_violations": [],
        "scope": "ready through complete; cleanup, process swap-outs and physical SSD I/O unmeasured",
    }
    try:
        if not isinstance(document, dict):
            raise TypeError("source report must be an object")
        issues = []
        report["source_errors"] = issues
        if document.get("status") not in ("complete", "validated"):
            issues.append("source report must be complete or validated")
        if "parity" in document and document["parity"] is not True:
            issues.append("source parity is not validated")
        for flag in ("reaped", "reader_joined", "sampler_joined"):
            if flag in document and document[flag] is not True:
                issues.append(f"source {flag} is not true")
        if document.get("active_request") is not None:
            issues.append("source has an unfinished request")
        baseline = previous = None
        for label, memory in _samples(document):
            sample = {
                "label": label,
                "memory": dict.fromkeys(FIELDS),
                "deltas_since_ready": None,
            }
            report["samples"].append(sample)
            report["final_deltas_since_ready"] = None
            if not isinstance(memory, dict):
                raise TypeError(f"missing memory at {label}")
            values = sample["memory"]
            for key in FIELDS:
                try:
                    values[key] = u64(memory.get(key))
                except ValueError as error:
                    raise ValueError(f"{label}.{key}: {error}") from error
            if any(values[key] == 0 for key in IDENTITY):
                raise ValueError(f"nonpositive process identity at {label}")
            if values["process_decompressions"] >= INT32_MAX:
                raise ValueError(f"saturated/invalid decompressions at {label}")
            if values["process_lifetime_peak_bytes"] < values["process_footprint_bytes"]:
                raise ValueError(f"physical peak below footprint at {label}")
            if (
                values["process_compressed_peak_logical_bytes"]
                < values["process_compressed_logical_bytes"]
            ):
                raise ValueError(f"compressed peak below gauge at {label}")
            if baseline is None:
                baseline = values
                incoming = values["process_ledger_swapins_logical_bytes"]
                report["startup_ledger_swapins_logical_bytes"] = incoming
                report["startup_already_affected"] = incoming > 0
            else:
                if any(values[key] != baseline[key] for key in IDENTITY):
                    raise ValueError(f"process identity changed at {label}")
                if any(values[key] < previous[key] for key in CUMULATIVE):
                    raise ValueError(f"cumulative counter reversed at {label}")
            deltas = {key: values[key] - baseline[key] for key in GAUGES + CUMULATIVE}
            sample["deltas_since_ready"] = deltas
            report["final_deltas_since_ready"] = deltas
            if report["maxima"] is None:
                report["maxima"] = {key: values[key] for key in GAUGES + CUMULATIVE}
            else:
                for key in GAUGES + CUMULATIVE:
                    report["maxima"][key] = max(report["maxima"][key], values[key])
            for key in (
                "process_lifetime_peak_bytes",
                "process_compressed_logical_bytes",
                "process_compressed_peak_logical_bytes",
            ):
                if values[key] > LIMITS[key]:
                    report["budget_violations"].append(
                        {
                            "sample": label,
                            "metric": key,
                            "observed": values[key],
                            "limit": LIMITS[key],
                            "message": f"{key} exceeds budget at {label}",
                        }
                    )
            if (
                deltas["process_compressed_lifetime_logical_bytes"]
                > LIMITS["compressed_lifetime_delta_bytes"]
            ):
                report["budget_violations"].append(
                    {
                        "sample": label,
                        "metric": "compressed_lifetime_delta_bytes",
                        "observed": deltas["process_compressed_lifetime_logical_bytes"],
                        "limit": LIMITS["compressed_lifetime_delta_bytes"],
                        "message": f"compressed lifetime delta exceeds budget at {label}",
                    }
                )
            if (
                deltas["process_ledger_swapins_logical_bytes"]
                > LIMITS["ledger_swapins_delta_logical_bytes"]
            ):
                report["budget_violations"].append(
                    {
                        "sample": label,
                        "metric": "ledger_swapins_delta_logical_bytes",
                        "observed": deltas["process_ledger_swapins_logical_bytes"],
                        "limit": LIMITS["ledger_swapins_delta_logical_bytes"],
                        "message": f"logical swapin delta exceeds budget at {label}",
                    }
                )
            previous = values
        if len(report["samples"]) < 2:
            raise ValueError("source has no complete memory samples")
        if issues or report["budget_violations"]:
            raise ValueError(
                "; ".join(issues + [v["message"] for v in report["budget_violations"]])
            )
        report.update(status="complete", qualification="observed_process_pressure_within_budget")
        return report
    except (TypeError, ValueError) as error:
        report["error"] = str(error)
        raise AnalysisError(str(error), report) from error


def run(source: Path, destination: Path):
    # Exclusive creation preserves existing analyses, including failed attempts.
    with destination.open("x", encoding="utf-8") as stream:
        report = {"status": "failed", "qualification": None, "samples": []}
        identity = {"source": str(source.resolve()), "source_sha256": None}
        try:
            stream.write(
                json.dumps({**report, **identity, "error": "analysis did not finish"}) + "\n"
            )
            stream.flush()
            raw = source.read_bytes()
            identity["source_sha256"] = hashlib.sha256(raw).hexdigest()
            report = analyze(json.loads(raw))
            return report
        except BaseException as error:
            if isinstance(error, AnalysisError):
                report = error.report
            report.update(
                status="failed", qualification=None, error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            report.update(identity)
            stream.seek(0)
            stream.truncate()
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


def interrupted(signum, _frame):
    raise InterruptedError(f"analysis interrupted by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    run(args.source, args.destination)
