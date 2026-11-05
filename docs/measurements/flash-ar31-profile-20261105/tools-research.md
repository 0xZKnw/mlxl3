# AR31 profiling tools and clock capabilities — 2026-11-05

Read the AR31 preregistration in `opti.md` before these queries. Capability and
source research only: **no recording, capture, model launch, benchmark, build,
prover, GPU workload or OS setting change** was performed. Only this report was
written. The requested baseline remains accepted13 defaults ON, AR20 and AR30
experiments OFF; its current short MAIN32 controls are10.409/10.253 tok/s.

**Recommendation:** first use the installed Metal System Trace on the actual
qualified release CLI. Xcode exists; the active developer directory merely
points to Command Line Tools. A per-command `DEVELOPER_DIR` override makes
`xctrace` available without changing the machine's developer-directory setting.
Inspect the resulting trace before adding engine instrumentation. It can expose
CPU scheduling, Metal commands and GPU work on a common timeline, avoiding the
old test-only profiler's JSON/allocation/clock problems.

## Observed local capabilities

| Read-only query | Observed result |
| --- | --- |
| `sw_vers` | macOS27.2, build26B123 |
| `xcode-select -p` | `/Library/Developer/CommandLineTools` |
| Plain `xcodebuild -version` | Refused because the active directory is CLT |
| Plain `xcrun xctrace version`, `list templates`, `help record/export` | Exit72: utility not found under that directory |
| Spotlight/app directory inspection | `/Applications/Xcode.app`, version27.2 |
| `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace version` | `xctrace version27.2 (27B52)` |
| Same override, `xctrace list templates` | Includes Metal System Trace, Time Profiler, CPU Profiler, System Trace, File Activity, CPU Counters, Allocations, Activity Monitor |
| Same override, `xctrace list instruments` | Includes GPU, Metal Application, Metal GPU Counters, Metal Resource Events, Time Profiler, Filesystem Activity, Thread Activity, System Call Trace, Points of Interest |
| Same override, `xctrace record --template 'Metal System Trace' --show-recording-options` | Succeeded; **this documented mode does not start recording** |
| Same query with `--instrument 'Metal GPU Counters'` | Succeeded, same configurable-options output; no explicit counter-set option appeared there |
| `DevToolsSecurity -status` | Developer mode currently disabled |
| Current user's groups | Includes `_developer` and `admin`; group membership alone does not establish recording permission |

The options JSON exposed Hangs settings and Time Profiler settings with normal
frequency, context-switch sampling OFF, kernel stacks OFF and waiting-thread
recording OFF. It did not prove which M5 GPU counter sets a recording will
actually collect. Do not infer that counters are enabled merely from the
instrument's name. The full local man page is
`/Applications/Xcode.app/Contents/Developer/usr/share/man/man1/xctrace.1`.
Instruments itself resides at
`/Applications/Xcode.app/Contents/Applications/Instruments.app` in this Xcode.

No capture permission was tested. `--no-prompt` is available; use it in a bounded
controller so a privacy/authorization denial becomes recorded evidence instead
of an unattended dialog. Do not enable Developer mode, change TCC, use sudo,
change GPU performance state or run `xcode-select -s` silently. A denied capture
is a limitation to report, not permission to change those settings. Apple lists
`xctrace` as an Xcode-supplied trace recording/export tool. [Apple CLI reference](https://developer.apple.com/documentation/xcode/xcode-command-line-tool-reference)

## Other installed Apple tools

Only executable discovery and `--help` were used:

- `/usr/bin/metalperftrace`: `collect` retrieves always-on metrics for a bounded
  date/duration window; `overview --json` reports per-process/per-layer metrics;
  `listen --json --pid PID` streams at a minimum1-second interval. `setup`
  changes collection/HUD behavior in a running process and was **not used**.
  Its frame/layer aggregation may be absent or too coarse for this headless
  compute CLI. It is not established as a replacement for per-dispatch timing.
- `/usr/bin/gpucapture`: supports attaching to a PID/boundary and producing a
  `.gputrace`; `--count`, `--until-exit`, resource/backtrace recording options
  exist. No capturable-process query, configuration or capture was performed.
- `/usr/bin/gpudebug`: supports opening/replaying `.gputrace`, JSON output,
  one-shot sessions and an idle timeout. Only its help was queried. Replaying
  a captured workload is a separate diagnostic, not the original AR timing.

These tools are documented on Apple's current [Metal developer tools page](https://developer.apple.com/metal/tools/).
Start with a system trace rather than full resource capture: a model trace can
retain many weight buffers and perturb the24GiB working set. Resource/shader
capture and hardware-counter replay should be separate, narrow diagnostics if
the initial timeline identifies a kernel needing them.

## Commands for the first capture, after preregistration and grant

The following are a **plan, not executed commands**. Resolve `CLI`, its SHA and
the model/checkpoint identity from the final frozen AR30 artifact plan. Use a
fresh output directory, disposable registry, and a sanitized baseline environment
with accepted defaults unchanged and all experimental flags OFF. Preserve the
existing benchmark's cache3GiB/readers4/dynamic0/lookahead0/prefix0/context4096,
T0/top-k1/penalty1/no-MTP/no-DFlash configuration.

Launch the actual executable, not a Python parent whose child capture coverage
would be uncertain. The production entry point from
`scripts/benchmark-flash-bridge.py:136–149` is:

```text
CLI --registry DISPOSABLE_REGISTRY bridge MODEL --context-length 4096
```

The direct bridge processes stdin sequentially (`native/src/direct_bridge.rs`),
so an input file may contain the same warmup generate request, measured generate
request and shutdown message, using distinct request IDs. Both generates retain
the exact MAIN prompt and32-token request schema from the existing benchmark.
The controller must independently validate both complete events,31 decode
tokens, full golden text/hash and orderly shutdown; a populated `.trace` alone
does not establish correct inference. Keep load/prefill/warmup in the trace so
the measured decode window can be selected from actual request events.

Installed syntax verified from local help/man:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 90s --no-prompt \
  --output CAPTURE_ROOT/native-ar.trace \
  --target-stdin CAPTURE_ROOT/requests.ndjson \
  --target-stdout CAPTURE_ROOT/bridge.ndjson \
  --env MLXL3_SSD_CACHE_GIB=3 \
  --env MLXL3_SSD_READERS=4 \
  --env MLXL3_SSD_DYNAMIC_READERS=0 \
  --env MLXL3_SSD_LOOKAHEAD_DEPTH=0 \
  --env MLXL3_PROMPT_CACHE_MIB=0 \
  --launch -- CLI --registry DISPOSABLE_REGISTRY bridge MODEL --context-length 4096

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace export \
  --input CAPTURE_ROOT/native-ar.trace --toc --output CAPTURE_ROOT/toc.xml

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace export \
  --input CAPTURE_ROOT/native-ar.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="SCHEMA_FROM_TOC"]' \
  --output CAPTURE_ROOT/table.xml
```

`CAPTURE_ROOT`, `CLI`, `MODEL`, registry and schema are explicit placeholders,
not literal paths or an assertion that that schema exists. Determine table
schemas from the actual TOC; do not hardcode an old Xcode schema. The installed
man page additionally documents export time-range restrictions and the newer
`toc`, `list tracks` and `show` inspection commands. Avoid `--window` for the
first bounded capture because a ring could silently discard model-load/start
events that are needed to assess completeness.

The outer supervisor needs its own bounded deadline, complete process-tree
cleanup and preserved error logs. A recording's time limit is not a promise
that the launched CLI exited or that descendants were reaped. An attach after
the bridge's ready event is a fallback if startup dominates the recording; do
not send measured requests until tracing-start is observed. The existing
`--notify-tracing-started` option supports that handshake, without injecting
extra GPU work.

Use one normal same-protocol control before and after the traced run to quantify
observer overhead; the traced run is a diagnostic regardless of throughput.
First export CPU threads/stacks, Metal command/encoder/pipeline identities,
GPU intervals, scheduling and filesystem tracks that actually exist. If a
required track is missing, declare it missing. Do not manufacture kernel
durations by splitting composite command-buffer time evenly.

Apple's [M5/A19 performance walkthrough](https://developer.apple.com/videos/play/tech-talks/111432/)
shows Metal System Trace with CPU/GPU timelines, compute tracks and optional
performance-limiter counters on M5. This supports tool applicability, not a
guarantee that this machine's permissions and default template expose every
counter. The [Metal tools documentation](https://developer.apple.com/metal/tools/)
also describes system tracing and hardware-counter views.

## Metal0.33 API inventory and one unsafe wrapper to avoid

The repository locks `metal=0.33.0`, crate checksum
`c7047791b5bc903b8cd963014b355f71dc9864a9a0b727057676c1dcae5cbc15`.
Inspected installed source root:
`/Users/justin/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/metal-0.33.0/src`.

| Capability | Actual binding/source |
| --- | --- |
| Query sampling boundaries | `device.rs:1421–1428` enum; `supports_counter_sampling` at1655 |
| Enumerate counter sets | `DeviceRef::counter_sets`, `device.rs:2132`; set name accessor in `counters.rs` |
| Correlated CPU/GPU clocks | `DeviceRef::sample_timestamps`, `device.rs:2128` |
| Create counter buffer | `new_counter_sample_buffer_with_descriptor`, `device.rs:1969`; count/set/storage descriptor setters in `counters.rs` |
| Compute encoder begin/end samples | `computepass.rs`: attachment/sample-buffer/start/end index accessors; `commandbuffer.rs:140` encoder-with-descriptor constructor |
| Explicit per-command sampling | `ComputeCommandEncoderRef::sample_counters_in_buffer`, `encoder.rs:1857` |
| GPU-side resolve into a buffer | `BlitCommandEncoderRef::resolve_counters`, `encoder.rs:1542` |
| Whole command-buffer GPU host times | No matching typed accessors found in `commandbuffer.rs`; native already uses raw Objective-C `GPUStartTime`/`GPUEndTime` at `flash_profile.rs:338–340` |

**Do not use the installed `CounterSampleBufferRef::resolve_counter_range` as
written.** At `counters.rs:71–79`, it creates a Vec with capacity but length0,
computes the byte length from that empty slice (0), requests a zero-byte NSData
copy, then sets the Vec length to the requested sample count. This exposes
uninitialized integer elements. That conclusion follows directly from the
source; no UB was executed or test/proof claim made. File SHA256:
`d23188bde20656d2e7ecdb80c474f4ae645a88d3f1ab688c11d4918609aee26d`.

If custom counters become necessary, use a reviewed raw NSData resolve/copy
that checks non-nil and the exact expected byte count into initialized storage,
or a correctly bounded GPU blit resolve followed by the existing completion
point. Timestamp results are8-byte `MTLCounterResultTimestamp`; other counter
sets have different structs. Reject zero/error-sentinel values, wrong lengths,
bad indexes and incomplete buffers. CPU range/lifetime guards can be checked
with Rust/Kani later; that would not prove GPU samples themselves.

## M5 sampling boundaries and per-dispatch feasibility

Apple identifies M5 as Apple GPU family10 in the [feature tables](https://developer.apple.com/metal/capabilities/).
The docs state that Apple silicon normally samples at stage boundaries; command
boundary sampling is a distinct capability. Query both `AtStageBoundary` and
`AtDispatchBoundary` on the **actual** device, and enumerate timestamp counters,
before allocating/encoding samples. No such device query was executed here,
so M5's exact returned bitset and available counter names remain unverified.
[Sampling API overview](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)

For stage-boundary support, use `MTLComputePassDescriptor` sample attachments
and begin/end-of-encoder indexes. The installed SDK's `MTLComputePass.h:33–48`
says these indexes are invalid on unsupported devices. Current native
`Metal::encode_bound` creates and ends **one compute encoder per dispatch**
(`gpu.rs:1170–1181`). Therefore stage-boundary samples can bracket the existing
dispatch without adding encoders, command buffers or CPU waits. That inference
only applies to these one-dispatch encoders; multi-dispatch custom encoders
would produce a composite interval, which must be labeled as such.

If dispatch-boundary sampling is supported, explicit encoder sample commands
are another option. Apple warns that a barrier serializes previous encoder
work and can slow the workload; without a barrier, the sample can overlap
commands. Its effect also does not isolate work in other passes. A barrier
mode must be a separate diagnostic, not silently imposed on the release
critical-path measurement. [Compute sampling semantics](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/samplecounters%28samplebuffer%3Asampleindex%3Abarrier%3A%29)

Buffers must be preallocated/bounded and retained through their command's
completion, then resolved at existing completion points. Do not allocate or
resolve a buffer per dispatch or add a wait to obtain each sample. Apple's
family10 table lists a32KiB maximum counter-buffer length and32 sample buffers;
the vendor's opaque representation means that dividing that length by the
standard8-byte output struct does **not** prove a legal sample count. Check
creation errors and recorded capacity/dropped samples explicitly. Count every
captured/uncaptured dispatch so a full buffer cannot masquerade as completeness.

## Clock alignment: three different domains to keep distinct

1. **Command-buffer fields.** Apple defines `GPUStartTime`/`GPUEndTime` as
   host seconds relative to system mach time, available after completion.
   `kernelStartTime`/`kernelEndTime` describe CPU driver scheduling, **not a GPU
   shader kernel's runtime**. [GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime),
   [kernelStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime)
2. **Counter timestamps.** `sampleTimestamps` returns correlated CPU/GPU
   timestamp pairs. Apple's current conversion guide treats CPU timestamps as
   nanoseconds and maps GPU time linearly using two bracketing pairs. Its older
   Apple-silicon explanation also states GPU samples align with mach absolute
   time. Use the explicit correlation and record the units; do not rely on an
   undocumented assumption about raw tick frequency. [Conversion guide](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time),
   [Apple silicon explanation](https://developer.apple.com/videos/play/tech-talks/10001/)
3. **Existing profile CPU events.** These are elapsed `std::time::Instant`
   values relative to a private process epoch. The current event explicitly
   says no CPU-Instant/GPU-host-time mapping was recorded. Neither a subtraction
   from another process's monotonic origin nor a wall-clock adjustment repairs
   this. Use Instruments' shared timeline, or explicitly record a bracketed
   process-epoch/mach-time correspondence before joining custom timelines.

For custom counters, with correlated samples `(C0,G0)`, `(C1,G1)`, map
`C(G)=C0+(G-G0)*(C1-C0)/(G1-G0)` and retain raw values, finite/positive spans,
bracketing coverage and mapping uncertainty. Convert raw `mach_absolute_time`
ticks with `mach_timebase_info`; do not apply that conversion twice to the
already-nanosecond CPU values returned by Metal. Sample correlation sparingly:
Apple notes it can trap into the kernel and perturb performance. [Clock correlation](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)

Whole-CB intervals can overlap and contain gaps; they are not shader busy time.
Compute CPU exclusive intervals and GPU interval unions on one calibrated
timeline, report unresolved gaps and concurrent work, and retain unique
request/thread/command/encoder/dispatch identities. Do not add inclusive CPU
spans, overlapping CB durations or performance-limiter percentages into a
96ms/token budget. Hardware counters, stage timestamps and total release AR
throughput answer different questions.

## Minimum next decision

Register one bounded native-CLI Metal System Trace attempt plus its unchanged
observer controls. Confirm permission, correct32/31 golden outputs, nonempty
actual target tracks and exportability. If those data identify dispatches,
CPU waits and SSD activity adequately, use them before touching engine code.
If the trace lacks a needed identity/clock/counter, add only that missing
instrumentation under a new preregistered diagnostic, with bounded storage and
numeric/observer-overhead checks. No privileged changes are needed merely to
use the discovered Xcode path; successful capture permission remains untested.
