"""Run platform-neutral proof inventory contracts even without Linux prlimit."""

import json
import re
from pathlib import Path

from test_kani_ci import (
    test_real_baseline_inventory_has_complete_disjoint_shards as _baseline,
)
from test_kani_ci import (
    test_required_obligations_preserve_baseline_and_complete_partitions as _required,
)


def test_baseline_inventory_partitions_are_portable():
    _baseline()


def test_required_proof_names_are_portable():
    _required()


def test_slru_manifest_matches_its_literal_source_harnesses():
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/src/flash_support/slru.rs").read_text()
    declared = re.findall(r"#\[kani::proof\](?:\s*#\[[^\n]*\])*\s*fn\s+(\w+)\s*\(", source)
    assert declared, "SLRU proof inventory must not be empty"
    prefix = "flash_support::slru::"
    required = json.loads((root / "scripts/kani_obligations.json").read_text())[
        "required_harnesses"
    ]
    assert {name for name in required if name.startswith(prefix)} == {
        prefix + name for name in declared
    }
