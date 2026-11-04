"""Run platform-neutral proof inventory contracts even without Linux prlimit."""

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
