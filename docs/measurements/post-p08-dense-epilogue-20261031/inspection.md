# F02 bridge inspection — CPU preparation

Prepared for `POSTP08-F02-01`; no model, GPU, build or native executable was
launched by this work. The root coordinator owns launch plans, the prospective
new campaign anchor, admission and final binary identities.

`benchmark-inspection.py` derives from T18a inspector
`f223e1cf7f940504413ffcc77a8e6c6df00006851af64514d28fbcf332e0b3db`.
It reuses the actual `scripts/benchmark-flash-bridge.py::validate` and
`issue58-20261029/pool-speed-driver.py::check_golden`. It does not reuse their
old manifests, binary identities or measured results. The unchanged bridge
runner's silence/error/EOF/shutdown/reader tests remain applicable.

## Root API and metadata

Load the module with `runpy.run_path`, then call
`preflight(entry, manifest_path, identity_path)` **before creating a PID**.
After the existing guarded runner completes, call
`inspect(entry, manifest_path, identity_path)`. The default metadata paths are
this folder's `prepared/manifest.json` and `final-identity.json`; neither is
created by the inspector. No new launcher or parser is introduced.

Both metadata documents require `experiment: POSTP08-F02-01`. Identity retains
the existing `sources: {path: sha256}` and `binaries.final_cli/python_runner`
objects with `path`/`sha256`. Include the inspector, shared bridge/golden
validators, manifest, each plan and both golden files in `sources`. The actual
source and binary bytes are checked before launch and again during inspection.
The historical T18 CLI `1fea100a…` is explicitly refused as an F02 executable.

Manifest retains `model`, `dflash_draft`,
`external_golden_files_for_root_validation: {MAIN, HOLD}`, and `plans`.
An entry contains `name`, `phase: bridge17|pilot128`, `mode: AR|MTP|DF`,
`arm: A|C`, `f02_flag: "0"|"1"`, `position_within_mode`, `plan`, `output`,
`expected_report: <output>/report.json`, and `tokens_per_generation: 17|128`.
The registered per-mode arm order is AC for bridge17 and ACCA for pilot128.
The root must execute that order serially and inspect each receipt before the
next process; this read-only preflight does not run or schedule processes.

Each plan contains one command, using the new identity's Python and CLI:
`python scripts/benchmark-flash-bridge.py CLI MODEL OUTPUT --cache-gib 3
--tokens 17|128 --repeats 1 --timeout 180`, then no mode argument for AR,
`--mtp` for MTP, or `--dflash-draft DRAFT --dflash-mode 1` for DFlash.
Paths in the actual command are absolute. Existing fixed controls are checked,
including T18/T04/T05/T17OFF, four readers, 3GiB, Hot/Wide/Unique/overlap1,
lookahead0, allocator64MiB and MLX40ops/40MB. F02A=0/C=1.

## Checks and limits

The inspector requires validated, joined bridge/reader and supervisor receipts;
MAIN/HOLD warmups then measurements, exact modes/counters, 17/128 complete
outputs, external prefix/full FNV and full128 text, finite consistent decode
and prefill timings, native footprint/peak strictly below12GB, zero Swapout,
Swapin≤256MiB and supervisor RSS≤10GiB. Initial output is durably `incomplete`;
missing/malformed/partial/divergent reports become failed and do not assert
golden success. Existing inspections cannot be overwritten. Numerical logits
and state parity remain the separate native oracles, not a bridge text claim.

CPU verification:

- `inspection_selftest.py`: **52 cases, 12 positive and 40 negative**. The
  original17 T18a scenarios are retained/adapted; positives cover all three
  modes, both budgets and both arms. All data/binaries/goldens are disposable
  synthetic files; no private artifacts, checkpoint or compiled CLI is needed.
- Omitting the actual report F02 flag check is detected by `f02_flag_mismatch`;
  the mutation exits1, then the source is restored byte-exact and all52 pass.
- Initial bridge/POOL tool suites: **78 passed in13.61s**, including fake-child
  silence/crash/invalid/late-error and join handling.
- The final argv check discovered that the existing CLI did not expose its
  `run(timeout=...)` API. Under `POSTP08-F02-CLI-TIMEOUT`, two lines expose
  `--timeout` as float, default300 preserved, and forward it to the same API.
  No new driver was added. A real Python CLI invocation with a silent fake
  bridge reproduces the old parser rejection; after correction it reaches the
  0.2s deadline, writes failed, joins child/reader and leaves no process group.
  CLI0/nan/inf are rejected before spawn. The initial test assumed ready had
  arrived inside0.2s; its retained failure was corrected to accept expiry while
  waiting for ready or complete, with the same deadline and cleanup assertions.
  **Final bridge/POOL suite:82 passed in8.03s**; all52 inspector cases repass.
- Ruff strict, format, Python byte compilation and diff check passed. Prior
  CrossHair absence is reused explicitly; no formal verification success or
  remote CI run is claimed. Root can add both new files and the standalone
  selftest to the existing Python CI step without private data or GPU.

Receipts: `inspection-selftest-restored.json`, `inspection-mutation.json/.log`,
`bridge-tools-tests.log`, `bridge-tools-timeout-final-r2.log`,
`inspection-selftest-after-timeout.json`, and `bridge-cli-help.log`. The early
r1 assertion-message mismatch was corrected by consolidating duplicated checks;
its failed log is retained. No launch plan,
new campaign anchor, real F02 binary or model gain is validated here.
