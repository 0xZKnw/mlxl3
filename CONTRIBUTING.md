# Contributing to MLXL3

Bug reports, focused fixes, documentation and reproducible measurements are
welcome. Issues and pull requests can be written in **English or French**.
Participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## Choose the right place

- [Report a bug](https://github.com/0xZKnw/mlxl3/issues/new/choose), including
  installation, Desktop, Agent, downloads or engine failures.
- Use the performance form for a speed or memory regression. Include the
  baseline and the workload; an unexplained tokens/s number is hard to compare.
- Describe the problem a feature would solve before proposing an implementation.
  Discuss substantial changes in an issue before starting a large pull request.
- Use [Discord](https://discord.gg/9hU46NVNnY) for usage questions. Report
  vulnerabilities **privately**, following [SECURITY.md](SECURITY.md).

Search existing issues and the [documentation index](docs/README.md) first.
Desktop and engine versions are independent: include both when relevant.

## Set up a development checkout

For macOS development, use Apple Silicon, macOS 26.2 or later, Rust 1.89+,
Swift 6.2+, Python 3.12 and compatible Xcode Command Line Tools. The pinned
release environment uses MLX 0.32.2.

```sh
git clone https://github.com/0xZKnw/mlxl3.git
cd mlxl3
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' -c packaging/constraints-macos.txt
swift build --package-path apps/MLXL3Studio
```

See [Build and verify](README.md#build-and-verify) for engine, converter and DMG
packaging. Do not download large checkpoints just to change documentation or
work on a CPU-only component. Use existing local models and small, redistributable
fixtures. Keep credentials, personal conversations and model weights out of commits.

## Run the checks relevant to your change

Portable Rust checks do not require an Apple GPU or an MLX installation:

```sh
cargo fmt --check
cargo clippy --locked --all-targets --no-default-features --features chat -- -D warnings
cargo test --locked --no-default-features --features chat
```

On a Mac with the Python dependencies installed:

```sh
export MLXL3_MLX_ROOT="$(.venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("mlx").submodule_search_locations)))')"
cargo test --locked --no-default-features --features mlx,chat --lib -- --test-threads=1
MLXL3_TEST_PYTHON="$PWD/.venv/bin/python" scripts/check-e2e.sh
```

The E2E script runs Python and the native Desktop/protocol checks. For a focused
Desktop preferences change, `scripts/check-desktop.sh --preferences-only` is also
available. `MLXL3_MACOS_SDK` can select an installed compatible SDK.

Add a regression that fails before a bug fix and exercises the production path.
Test boundaries and errors as well as the successful case. Rust logic changes
must attempt Kani on the actual implementation, following [AGENTS.md](AGENTS.md).
State which properties and bounds were checked; a timeout is not a proof.

CPU tests and hosted CI do not qualify physical GPU behavior. Run affected
physical tests explicitly, one at a time, with the required model and hardware.
List skipped or unavailable checks in the PR instead of treating them as passed.
Documentation-only changes need link and format checks, not an inference run.

## Engine, numerical and performance changes

Read [opti.md](opti.md) and its relevant historical reports before proposing or
benchmarking an optimization. Follow the journal's preregistration and recording
rules. Preserve failed and inconclusive attempts.

Check correctness before measuring speed. For numerical or cache changes,
exercise the real CLI/bridge and the relevant logits or state against an
independent reference. Record the source revision, checkpoint and quantization,
hardware, context, output budget, sampling and acceleration settings. Compare
the same workload; separate prefill, decode, TTFT, total latency and memory.
Report noise and regressions, and keep GPU measurements separate from builds,
proof runs and other inference. A kernel result alone is not an app-level gain.

## Submit a pull request

Keep the change focused and explain the user-visible problem, resulting behavior
and validation. Link the related issue when there is one. For UI changes, add
screenshots of the affected states and relevant window sizes. For engine changes,
include correctness evidence and any applicable measurement or proof report.

Use the PR template to record commands, outcomes, skipped checks and limitations.
Do not add unrelated dependency upgrades or generated release artifacts. Follow
the existing license and preserve upstream notices when adapting code. Maintainers
review and decide releases separately from contributions; no release or response
schedule is guaranteed.
