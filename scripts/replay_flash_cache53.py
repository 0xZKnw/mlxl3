"""Prepare exact53 route sequences; LFU replay starts cold and cannot reproduce53 hit state."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def routes(events, part):
    API["validate_events"](events)
    roots = [e for e in events if e["parent"] is None]
    assert len(roots) == 1 and roots[0]["name"] == part
    layers = [[] for _ in range(48)]
    parents = []
    for event in events:
        parent = event["parent"]
        layer = parents[parent] if parent is not None else None
        if event["name"].startswith("layer_"):
            layer = int(event["name"].removeprefix("layer_"))
        parents.append(layer)
        if event["name"] == "moe_forward_rows" and layer is not None:
            assert event["data"]["capacity"] == 53
            layers[layer].append(event["data"]["ids"].copy())
    assert layers[0] and all(len(layer) == len(layers[0]) for layer in layers)
    assert all([len(ids) for ids in layer] == [len(ids) for ids in layers[0]] for layer in layers)
    return layers


def read_routes(directory, part):
    return routes(API["load_events"](directory / f"trace_1-{part}.json"), part)


def main(source, destination):
    destination.mkdir()
    status = {"status": "incomplete", "confirmed_tps_gain": False}
    status_path = destination / "extraction.json"
    status_path.write_text(json.dumps(status))
    try:
        cases, hashes = [], {}
        for prompt in ["MAIN", "HOLD"]:
            for mode in ["OFF", "MTP"]:
                directory = source / f"natural-{mode}-c3-{prompt}" / "profile"
                summary_path = directory / "trace_1-summary.json"
                summary = json.loads(summary_path.read_text())
                golden = json.loads(
                    (
                        ROOT
                        / f"docs/measurements/flash-next-ssd-impl/opt-39a-les-lru/confirmation/reference128-{prompt}.json"
                    ).read_text()
                )
                API["validate_summary"](summary, "natural", golden)
                parts = {}
                for part in ["prefill", "decode"]:
                    path = directory / f"trace_1-{part}.json.gz"
                    parts[part] = read_routes(directory, part)
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[str(summary_path)] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
                cases.append(
                    {
                        "name": f"{mode}-{prompt}",
                        "layers": [{part: parts[part][i] for part in parts} for i in range(48)],
                    }
                )
        (destination / "routes.json").write_text(json.dumps(cases, separators=(",", ":")))
        status.update(
            status="complete_validated_routes_not_original_cache_state", cases=4, sources=hashes
        )
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
