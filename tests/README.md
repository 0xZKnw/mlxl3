# Tests and workflow checks

- `test_*.py`: Python regression tests for the engine, conversion, scripts, and reports; some require optional dependencies.
- `*-check.swift`: Desktop and interface checks; `fake-*-engine.py` supplies simulated engines for some workflows.
- `physical_*.py`: harnesses for explicit checks on compatible hardware and models; their presence does not mean they run by default.
- `*.rs` and `quantize_packing_proof.cpp`: targeted native contracts and proofs.
- [fixtures/](fixtures/README.md): preserved test corpora and data.

Commands used in practice are in [check.yml](../.github/workflows/check.yml), [rust.yml](../.github/workflows/rust.yml), and [scripts/check-e2e.sh](../scripts/check-e2e.sh). The E2E suite runs Python tests followed by Desktop checks; review its dependencies and hardware conditions before running it. Skipped tests remain unexecuted.
