"""Same native bridge, larger token budget; retain and report early EOS counts."""
import runpy
import sys
from pathlib import Path

ns = runpy.run_path("scripts/benchmark-flash-bridge.py")
base = ns["validate"]

def validate(event, mode, mtp, budget, dflash=False):
    actual = event["stats"]["generated_tokens"]
    if type(actual) is not int or not 128 <= actual <= budget:
        raise ValueError("invalid long-output count")
    base(event, mode, mtp, actual, dflash)

if __name__ == "__main__":
    run = ns["run"]
    run.__globals__["validate"] = validate
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
        tokens=512, repeats=2, cache_gib=3, prompt="HOLD",
        dflash_draft=Path(sys.argv[4]), timeout=240)
