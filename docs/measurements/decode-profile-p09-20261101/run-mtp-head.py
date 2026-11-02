"""Explicitly authorized MTPHEAD restart; preserve the original P09 anchor."""

import json
import runpy
import sys
from pathlib import Path

API = runpy.run_path(str(Path(__file__).with_name("run.py")))
ANCHOR = Path(__file__).with_name("anchor-mtp-head.json")


def run(plan, identity):
    sources = json.loads(identity.read_text())["sources"]
    assert {str(ANCHOR.resolve()), str(Path(__file__).resolve())} <= set(sources)
    API["run"].__globals__["ANCHOR"] = ANCHOR
    return API["run"](plan, identity)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
