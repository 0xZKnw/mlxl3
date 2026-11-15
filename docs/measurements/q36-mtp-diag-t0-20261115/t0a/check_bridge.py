"""Q36-MTP-ACCEPT-POS-T0A: check bridge histograms against counters, REL totals and REL token hashes.

Fails closed: any missing pass, prompt, run or field is a failure, and the summary is written with
status "failed" and the reasons. Usage: python -I check_bridge.py <t0a directory>.
"""

import json
import sys
from pathlib import Path

REL = Path("/Users/justin/Documents/mix-stq1_0/docs/measurements/rel-engine-2.0.11-beta1-20261115")
PROMPTS = ("short", "document")


def histogram_totals(h):
    if len(h) != 4 or any(len(row) != 4 for row in h):
        raise ValueError(f"histogram shape {h}")
    for w in range(4):
        for a in range(4):
            if not isinstance(h[w][a], int) or h[w][a] < 0:
                raise ValueError(f"invalid cell {w},{a}")
            if a > w and h[w][a]:
                raise ValueError(f"impossible cell {w},{a}")
    cycles = sum(map(sum, h))
    spec = sum(sum(h[w]) for w in range(1, 4))
    proposed = sum(w * h[w][a] for w in range(4) for a in range(4))
    accepted = sum(a * h[w][a] for w in range(4) for a in range(4))
    per_width = [sum(h[w]) for w in range(4)]
    positions = []
    for p in range(3):
        hit = sum(h[w][a] for w in range(4) for a in range(4) if w > p and a > p)
        tested = sum(h[w][a] for w in range(4) for a in range(4) if w > p and a >= p)
        positions.append([hit, tested])
    return cycles, spec, proposed, accepted, per_width, positions


def rel_reference(depth):
    ref = {}
    for path in sorted((REL / f"speed-mtp{depth}-BAAB").glob("*-B.json")):
        data = json.loads(path.read_text())
        for prompt in PROMPTS:
            for run in data["prompts"][prompt]["runs"]:
                s = run["stats"]
                key = (
                    run["token_hash"],
                    s["mtp_accepted_tokens"],
                    s["mtp_proposed_tokens"],
                    s["mtp_blocks"],
                    tuple(s["mtp_depth_blocks"]),
                    s["decode_tokens"],
                )
                ref.setdefault(prompt, set()).add(key)
    for prompt in PROMPTS:
        if len(ref.get(prompt, ())) != 1:
            raise ValueError(
                f"REL depth {depth} {prompt}: expected one reference, got {ref.get(prompt)}"
            )
    return {prompt: next(iter(v)) for prompt, v in ref.items()}


def main():
    root = Path(sys.argv[1])
    errors, rows = [], []
    for depth in (1, 2, 3):
        try:
            reference = rel_reference(depth)
            out = root / f"bridge-mtp{depth}"
            results = json.loads((out / "results.json").read_text())
            if results.get("status") != "complete" or results.get("parity") is not True:
                errors.append(
                    f"mtp{depth}: status {results.get('status')} parity {results.get('parity')}"
                )
            passes = sorted(out.glob("*-[AB].json"))
            if len(passes) != 4:
                errors.append(f"mtp{depth}: expected 4 passes, found {len(passes)}")
            for path in passes:
                data = json.loads(path.read_text())
                label = data["label"]
                for prompt in PROMPTS:
                    runs = data["prompts"][prompt]["runs"]
                    if not runs:
                        errors.append(f"mtp{depth} {path.name} {prompt}: no runs")
                    ref = reference[prompt]
                    for index, run in enumerate(runs):
                        s = run["stats"]
                        tag = f"mtp{depth} {path.name} {prompt} run{index}"
                        if run["token_hash"] != ref[0]:
                            errors.append(f"{tag}: token hash {run['token_hash']} != REL {ref[0]}")
                        observed = (
                            s["mtp_accepted_tokens"],
                            s["mtp_proposed_tokens"],
                            s["mtp_blocks"],
                            tuple(s["mtp_depth_blocks"]),
                            s["decode_tokens"],
                        )
                        if observed != ref[1:]:
                            errors.append(f"{tag}: counters {observed} != REL {ref[1:]}")
                        if label == "A":
                            if "mtp_accept_histogram" in s:
                                errors.append(f"{tag}: release reports a histogram")
                            continue
                        h = s.get("mtp_accept_histogram")
                        lookup = s.get("mtp_lookup_accept_histogram")
                        if h is None or lookup is None:
                            errors.append(f"{tag}: candidate histogram missing")
                            continue
                        if any(map(any, lookup)):
                            errors.append(f"{tag}: lookup histogram not empty {lookup}")
                        cycles, spec, proposed, accepted, per_width, positions = histogram_totals(h)
                        checks = {
                            "cycles == sum(depth_blocks)": cycles == sum(s["mtp_depth_blocks"]),
                            # depth_blocks is indexed by the configured depth, not the cycle width.
                            "depth_blocks only at depth": all(
                                v == 0 for i, v in enumerate(s["mtp_depth_blocks"]) if i != depth
                            ),
                            "widths <= depth": all(per_width[w] == 0 for w in range(depth + 1, 4)),
                            "speculative == blocks": spec == s["mtp_blocks"],
                            "proposed": proposed == s["mtp_proposed_tokens"],
                            "accepted": accepted == s["mtp_accepted_tokens"],
                            "cycles + accepted == decode": cycles + accepted == s["decode_tokens"],
                            "lookup blocks == 0": s["mtp_lookup_blocks"] == 0,
                        }
                        for name, ok in checks.items():
                            if not ok:
                                errors.append(f"{tag}: {name} failed ({h}, {s})")
                        rows.append(
                            {
                                "depth": depth,
                                "pass": path.name,
                                "prompt": prompt,
                                "run": index,
                                "histogram": h,
                                "positions_hit_tested": positions,
                                "proposed": proposed,
                                "accepted": accepted,
                                "token_hash": run["token_hash"],
                            }
                        )
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"mtp{depth}: {type(error).__name__}: {error}")
    expected_rows = (
        3 * 2 * len(PROMPTS)
    )  # depths x candidate passes x prompts, one measured run each
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} candidate rows, got {len(rows)}")
    summary = {"status": "passed" if not errors else "failed", "errors": errors, "rows": rows}
    (root / "bridge-summary.json").write_text(json.dumps(summary, indent=2))
    for row in rows:
        print(
            row["depth"], row["pass"], row["prompt"], row["histogram"], row["positions_hit_tested"]
        )
    print("\n".join(errors) if errors else "all bridge checks passed")
    print("bridge_check", summary["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
