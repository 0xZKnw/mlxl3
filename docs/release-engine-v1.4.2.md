# MLXL3 engine 1.4.2

This release reduces Qwen MTP scheduling and draft-projection overhead while
retaining full-target verification. It includes
[PR #26](https://github.com/0xZKnw/mlxl3/pull/26) and retains the exact M5 dense
decode geometry introduced in [engine 1.4.1](release-engine-v1.4.1.md).

## Performance changes

- **Progressive target-layer submission** allows the GPU to begin while the
  CPU constructs later layers. The default is restricted to MTP on Apple M5,
  hidden 2048, 40 layers, 256 experts/top8—the measured Qwen3.6-35B-A3B profile.
  Arithmetic, precision and target verification remain unchanged.
- **Optional compact affine4 draft projection**, prepared separately from the
  target checkpoint: 79,591 logical IDs, 79,616 padded rows and 87.77 MiB for the
  projection/map. Draft candidates map back to the full vocabulary; every
  delivered token is verified by the full EXL3 target. The stock head remains
  the default and no weights or vocabulary lists are bundled.
- Array multi-output evaluation retains its input owners and validates empty
  or null inputs. Tune keys include the pipeline/lookup policy so measurements
  from another policy are not silently reused.

### Measured results

M5 Air/24 GiB, MLX 0.32.2, Qwen3.6-35B-A3B EXL3 2.49 bpw, greedy MTP2,
context 4096/cache OFF, 128 warmup/128 generated tokens, two repeats, AC power:

| Comparison | Code decode | Complete code time |
| --- | ---: | ---: |
| Pipeline, stock head, same binary OFF→scoped default | +7.16–7.80% | −6.27–6.80% |
| Pipeline, compact head, same binary OFF→scoped default | +7.53–9.80% | −6.32–8.22% |
| Compact draft versus engine 1.4.1, separate campaign | +8.20% | −7.15% |

Pipeline gains were confirmed in ABBA and BAAB with exact token/text parity
and decode control drift within 3%. Code medians were **69–70 tok/s stock** and
**75–77 tok/s compact**; French medians were 61–62 and 68–69 respectively.
These comparisons are separate and are not added. No stable 80–85 tok/s,
universal-model gain or confirmed long-context gain is claimed.
[Protocols, numerical oracles and raw evidence](engine-v1.4.2-pipeline.md).

## Options and defaults

- `MLXL3_QWEN_PIPELINE=0` disables progressive submission; `1` experimentally
  extends it to all Qwen trunk paths. Other hardware/geometries keep the prior
  default. The default policy also participates in Tune's cache key.
- Prepare a compact head with the existing
  [conversion instructions](../README.md#experimental-compact-draft-preparation),
  select its folder in Desktop and run Tune MTP. Compact projection is optional.
- `MLXL3_MTP_LOOKUP=1` enables experimental repeated-prompt drafting. It is
  **OFF by default**; a speed benefit was not reproduced. Full-target
  verification and exact cache repair remain mandatory.

## Verification and limits

- Final PR26 source passed **8/8 PR/push CI jobs**: formatting, strict Clippy,
  native Linux/macOS tests, shipped `mlx,chat` release compilation and Desktop
  protocol/E2E checks. Native jobs passed 56 Rust tests (2 ignored Linux/3 macOS)
  and 293 Python tests/1 skip; Desktop passed 296 Python tests/4 skips.
- Both Kani 0.68.0 jobs verified **39/39 CPU harnesses**: 5,463 SUCCESS,
  98 SATISFIED, 72 internal UNREACHABLE and zero FAILURE/UNDETERMINED.
  Symbolic domains and unwinding assertions remain enabled; MTP ≤3/unwind 6,
  prefixes ≤8/unwind 34 and lookup history ≤20/unwind 33 are bounded checks.
  CPU verification does not prove MLX, Metal, FFI or concurrency.
- Independent physical checks exercise logits, hidden states, recurrent/KV
  rollback, stock/compact recursive drafts, cancellation and the real bridge.
  [Numerical checks and scope](engine-v1.4.2-pipeline.md#vérification-et-limites).
- RapidMLX-inspired GDN fusion was numerically checked, then withdrawn after
  control drift exceeded the allowed limit. Other rejected prototypes remain
  documented and are not delivered or advertised as additional boosts.
  [Research and CI repair](engine-v1.4.2-upstream.md).

## Install and compatibility

Download **MLXL3-Engine-v1.4.2-arm64.tar.gz** through Desktop's independent
engine updater. Requires **Desktop 1.4.0+, Apple Silicon, macOS 26.2+**,
bridge protocol 1 and MLX 0.32.2. Restart Desktop after updating. No model weights
are included. The archive carries a manifest, hashes and third-party notices.

Desktop 1.4.2 also bundles this engine. Its composer fix, automatic idle unload
and MTP-preparation recovery require the
[Desktop update](release-v1.4.2.md), rather than an engine-only installation.
Source commit, artifact size, SHA-256 and final release checks accompany the
[GitHub release](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.2).
