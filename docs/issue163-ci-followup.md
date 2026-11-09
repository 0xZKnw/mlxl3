# PR163: follow-up of the first expanded-head CI

Baseline: published b28d33b72fc2e495f8b69611217a6a9e2bf8d798, run 42604347336 and 42604347354. Changes below are local until the next published commit is verified.

## Apple compilation

Hosted macOS reported six errors missed by Linux feature gating:
- MTP still imports the independent strict `opt_in` parser removed during Qwen profile refactoring. Restore its previous semantics; MTP does not become a profile default.
- PrefetchPool::abandon visibility resolved to a nested owner rather than crate::gpu. Narrow explicit visibility fixes its existing sibling caller.
- Four temporary Cow<Matrix> operands could not coerce through `&...?`. Explicit `.as_ref()` retains temporary lifetime and avoids cloning.

Strict portable Clippy passed after the edits. Actual Apple compilation remains required; it is not implied by this Linux result.

## Native converter fixture pollution

All 21 worker reuse parameter combinations failed in the combined hosted suite because they asserted global `mlx.core` absence after unrelated MLX tests had imported it. Production output and child-lifecycle assertions were already passing.

The test now blocks MLX imports only across the actual converter operation, verifies the guard catches an explicit MLX import, and keeps all output, shape, batching and reaping assertions. It does not erase `sys.modules` or skip loaded-MLX environments.

- First local collection omitted PYTHONPATH=src and ran no tests: ModuleNotFoundError. Corrected environment on retry.
- Full native quantizer fixture: 52 passed, 1 physical skip, 2.99s.
- Separate process with mlx.core preloaded: all 21 reuse combinations passed, 1.36s.
- Ruff check and formatting passed.

Logs: docs/measurements/pr163-ci-followup/.

## Kani

Hosted inventory failed during compilation with a memory-allocation abort under the 6 GiB bound; zero proofs ran. This differs from the earlier two route-proof timeouts. Investigation and exact-domain discovery changes are tracked in issue163-routes-proof-followup.md. Neither failure is called a runtime counterexample or a successful proof.
