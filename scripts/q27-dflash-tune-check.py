#!/usr/bin/env python3
"""Run one real DFlash2 Tune through the bridge and save its rows.

PRODUCT-TUNE-NEAR-TIE / PRODUCT-DFLASH-SAMPLING: checks that a greedy Tune reports its
tolerated near ties and that a sampled Tune measures with the request's sampling
settings. One bridge process, one GPU model; the report is written even on failure.
"""

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_json_process import JsonProcess, positive_seconds  # noqa: E402

_spec = importlib.util.spec_from_file_location("q27_bench", HERE / "q27-decode-bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
SAMPLERS = _bench.SAMPLERS


def wait_for(process, deadline, accept):
    while True:
        event = process.receive(deadline)
        if event.get("type") == "error":
            raise RuntimeError(f"bridge error: {event}")
        if accept(event):
            return event


def validate(complete, sampled):
    rows = complete.get("rows")
    if not isinstance(rows, list) or [row.get("depth") for row in rows] != [0, 1, 2, 3]:
        raise RuntimeError(f"Tune rows are not the four modes: {rows}")
    for row in rows:
        rate = row.get("decode_tps")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise RuntimeError(f"invalid Tune rate: {row}")
        if bool(row.get("sampled")) != sampled:
            raise RuntimeError(f"row sampled flag {row.get('sampled')} != {sampled}")
        for tie in row.get("near_ties") or []:
            if abs(tie["gap"]) > 0.1:
                raise RuntimeError(f"tolerated divergence above the near-tie bound: {tie}")
    if complete.get("best_depth") not in (0, 1, 2, 3) or bool(complete.get("sampled")) != sampled:
        raise RuntimeError(f"invalid Tune result: {complete}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--dflash-draft", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampler", choices=tuple(SAMPLERS), default="greedy")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=positive_seconds, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("preserve the previous report")
    sampled = args.sampler != "greedy"
    report = {"status": "running", "engine": args.engine, "model": args.model,
              "sampler": args.sampler, "sampling": SAMPLERS[args.sampler] if sampled else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mlxl3-tune-") as directory, open(
        args.out.with_suffix(".stderr.log"), "wb"
    ) as log:
        process = JsonProcess(
            [args.engine, "--registry", str(Path(directory) / "models.json"), "bridge",
             args.model, "--context-length", str(args.context)],
            stderr=log,
        )
        try:
            deadline = time.monotonic() + args.timeout
            ready = wait_for(process, deadline, lambda e: e.get("type") == "ready")
            report["ready"] = {key: ready.get(key) for key in (
                "dflash_supported", "dflash_family", "dflash_tune_supported",
                "dflash_sampling_supported", "runtime_commit")}
            request = {"type": "tune_dflash", "request_id": "tune", "memory_saver": False,
                       "dflash_draft_path": args.dflash_draft}
            if sampled:
                request.update(SAMPLERS[args.sampler])
            process.send(request, deadline)
            complete = wait_for(process, deadline, lambda e: e.get("request_id") == "tune"
                                and e.get("type") == "dflash_tune_complete")
            report["complete"] = complete
            validate(complete, sampled)
            report["status"] = "passed"
        except Exception as error:  # the failure is saved with an explicit state
            report["status"] = "failed"
            report["error"] = repr(error)
            raise
        finally:
            report["seconds"] = time.monotonic() - started
            args.out.write_text(json.dumps(report, indent=2))
            process.close()
    for row in report["complete"]["rows"]:
        print(f"depth {row['depth']}: {row['decode_tps']:.2f} tok/s eligible={row['eligible']} "
              f"reason={row.get('reason')} near_ties={row.get('near_ties')} sampled={row.get('sampled')}")
    print(f"best_depth {report['complete']['best_depth']}")


if __name__ == "__main__":
    main()
