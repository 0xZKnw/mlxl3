# MLXL3 Desktop 1.4.2 · build 24

A faster Qwen MTP path, a responsive composer during background updates, and
automatic cleanup of an idle model. Includes **engine 1.4.2** and the changes
from [PR #26](https://github.com/0xZKnw/mlxl3/pull/26) and
[PR #28](https://github.com/0xZKnw/mlxl3/pull/28).

## Chat and model lifecycle

- **Send and Tune MTP stay available during update checks and downloads** when
  the model is ready. The composer now follows updater state changes immediately.
  Installing an update still blocks inference.
- **Automatic model unloading after inactivity** is configurable in Settings:
  Never, 1/5/10/15/30 minutes, 1 hour or 2 hours. The default is **15 minutes**,
  and the preference survives relaunch.
- Generation, tool calls, tuning and MTP/MCP configuration suspend the idle
  deadline. A full delay starts again after completion or cancellation. Unloading
  releases the engine, head and caches while preserving model files, selection,
  conversations and settings; select the model again to reload it.
- **MTP preparation recovers from malformed replies.** An OFF request without
  an acknowledgement also recovers after 30 seconds. Recovery does not interrupt
  an active generation or allow a stale idle deadline to eject a reloaded model.

## Qwen performance

- Engine 1.4.2 progressively submits target layers during MTP on the measured
  **M5 / Qwen3.6-35B-A3B** profile. With the stock head, code decode improved
  **7.16–7.80%** against the same binary with that submission disabled, confirmed
  in opposite pass orders. Other devices and geometries retain their previous
  defaults.
- An **optional, separately prepared compact Q4 draft projection** reduces MTP
  projection work. It uses 87.77 MiB of additional tensors. Select the prepared
  folder and run Tune MTP; the automatically downloaded stock head remains the
  default. [Preparation instructions](../README.md#experimental-compact-draft-preparation).
- Validated code medians reached **75–77 tok/s with the compact head** and
  **69–70 tok/s with the stock head** on M5 Air/24 GiB, Qwen3.6-35B-A3B EXL3
  2.49 bpw, greedy MTP2, context 4096/cache OFF, 128 warmup and 128 generated
  tokens, two repeats, ABBA/BAAB on AC power. These are prompt-specific results;
  **80–85 tok/s is not established** and gains from separate comparisons are
  not added together.
- Upgrading from Desktop 1.4.0 also includes engine 1.4.1's measured M5 dense
  decode tile selection. [Engine 1.4.1 scope](release-engine-v1.4.1.md).

## Reliability and validation

- The intermittent macOS EOF regression test now distinguishes ordinary process
  teardown from intentional silence. Its diagnostic and child cleanup checks
  remain enforced; no retry or test removal masks the failure.
- PR26's final source passed all **eight PR/push CI jobs**: Rust formatting,
  strict Clippy, Linux/macOS tests, MLX release compilation, Desktop E2E and
  Kani. Native jobs passed 56 Rust tests and 293 Python tests; Desktop jobs
  passed 296 Python tests, 49 composer cases and 114 idle/MTP assertions.
- Both Kani jobs passed 39/39 CPU harnesses, with 5,463 successful obligations,
  98 satisfied coverage checks and 72 inspected internal unreachable obligations.
  Bounds include MTP ≤3 proposals, prefixes ≤8 and lookup history ≤20. This bounded
  CPU verification does not prove Metal, MLX, FFI, Swift or concurrency.
- Historical skips remain visible: native 2 ignored tests on Linux/3 on macOS,
  Python 1 skip in native jobs and 4 in Desktop jobs; physical Metal checks are
  explicitly excluded from hosted Desktop CI. See the
  [source validation and limits](engine-v1.4.2-upstream.md) and
  [numerical/performance protocols](engine-v1.4.2-pipeline.md).
- [Release packaging checks](release-v1.4.2-validation.md) also cover the signed
  updater, archive hashes, the mounted DMG, the release app's built-in checks
  and 43 production bridge requests with exact MTP parity and recovery.

## Install and compatibility

- Download **MLXL3-Desktop-v1.4.2-b24-Apple-Silicon.dmg**, or use the app's
  update action. The DMG includes the native engine, MLX 0.32.2 and Metal assets;
  Python and Homebrew are not required. Model weights are downloaded separately.
- Requires **Apple Silicon and macOS 26.2 or later**. The app remains ad-hoc
  signed and not notarized; on first launch, use macOS **Open Anyway** if needed.
- The independent **engine-v1.4.2** update supports Desktop 1.4.0+, but the
  composer, idle-unload and MTP-recovery UI changes require Desktop 1.4.2.
  Restart Desktop after an independent engine update.
- Source commit, artifact size, SHA-256 and final release checks accompany the
  [GitHub release](https://github.com/0xZKnw/mlxl3/releases/tag/v1.4.2).

## Experimental scope

The stock MTP head remains the default. Prompt lookup is opt-in and OFF by
default because its gain was not reproduced. RapidMLX-inspired GDN fusion and
other unsuccessful prototypes were withdrawn; they are not included as boosts.
No universal speedup or long-context gain is claimed. Original negative results
are retained in the [optimization journal](../opti.md).
