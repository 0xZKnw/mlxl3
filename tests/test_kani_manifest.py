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

from scripts.kani_ci import required_names


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


def test_qsa_chain_manifest_matches_actual_policy_declarations():
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/src/qsa_chain_policy.rs").read_text()
    declared = re.findall(r"#\[kani::proof\](?:\s*#\[[^\n]*\])*\s*fn\s+(\w+)\s*\(", source)
    assert declared, "QSA command-chain proofs must not disappear"
    prefix = "qsa_chain_policy::"
    assert {name for name in required_names() if name.startswith(prefix)} == {
        prefix + name for name in declared
    }


def test_ar10_default_policy_names_match_real_proof_declarations():
    root = Path(__file__).resolve().parents[1]
    required = set(required_names())
    for module, name in [
        ("ar10_defaults", "qualified_defaults_preserve_strict_overrides_and_legacy_io"),
        ("expert_ar_policy", "expert_ar_flags_preserve_qualified_defaults_and_overrides"),
    ]:
        source = (root / "native/src" / (module + ".rs")).read_text()
        declared = re.findall(r"#\[kani::proof\](?:\s*#\[[^\]]+\])*\s*fn (\w+)\(", source)
        assert name in declared and f"{module}::{name}" in required
    assert (
        "expert_ar_policy::expert_ar_flags_are_strict_off_and_require_qualified_hardware"
        not in required
    )
