"""Short, sequential production-tuner comparison; requires local Apple GPU fixtures.

Each engine warms all four modes and measures the same two 96-token prompts.
No downloads, registry changes or chat history. Supply engines in ABBA order.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3.6-35B-A3B-EXL3-2.49bpw")
    parser.add_argument("--head", default="models/Qwen3.6-35B-A3B-MTP-4bit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = None
    for index, engine in enumerate(args.engines):
        label = f"{index}-{Path(engine).parent.name}"
        conditions = {
            key: subprocess.check_output(command, text=True).strip()
            for key, command in {
                "battery": ["pmset", "-g", "batt"],
                "thermal": ["pmset", "-g", "therm"],
                "swap": ["sysctl", "vm.swapusage"],
            }.items()
        }
        (args.output / f"{label}-conditions.json").write_text(json.dumps(conditions, indent=2))
        with tempfile.TemporaryDirectory(prefix="mtp-tune-benchmark-") as directory:
            request = {"type": "tune_mtp", "request_id": "benchmark",
                       "mtp_head_path": str(Path(args.head).resolve())}
            process = subprocess.run(
                [str(Path(engine).resolve()), "--registry", str(Path(directory) / "models.json"),
                 "bridge", str(Path(args.model).resolve()), "--context-length", "4096"],
                input=json.dumps(request) + "\n", text=True, capture_output=True, timeout=180,
            )
        (args.output / f"{label}.jsonl").write_text(process.stdout)
        (args.output / f"{label}.stderr").write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f"{label}: engine exited {process.returncode}: {process.stderr}")
        events = [json.loads(line) for line in process.stdout.splitlines()]
        errors = [event for event in events if event["type"] == "error"]
        completed = [event for event in events if event["type"] == "mtp_tune_complete"]
        if errors or len(completed) != 1:
            raise RuntimeError(f"{label}: failed tuner: {errors}")
        result = completed[0]
        rows = result["rows"]
        assert [row["depth"] for row in rows] == [0, 1, 2, 3]
        assert all(row["token_hashes"] == rows[0]["token_hashes"] for row in rows)
        assert all(row["decode_tokens"] == 190 for row in rows)
        if reference is None:
            reference = rows[0]["token_hashes"]
        assert rows[0]["token_hashes"] == reference
        (args.output / f"{label}-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"label": label, "engine": engine, "best": result["best_depth"],
                          "tps": [row["decode_tps"] for row in rows],
                          "acceptance": [[row["accepted_tokens"], row["proposed_tokens"]] for row in rows],
                          "hashes": reference}), flush=True)


if __name__ == "__main__":
    main()
