# Local verification evidence

The [parent report](../../apple-silicon-stability-20261030.md) distinguishes
passing tests, dependency failures, unavailable verifiers and physical checks.
`verification.json` hashes the changed executable sources and saved logs. Later
documentation changes do not requalify a different executable version.

Commands run from the repository root (Rust tools isolated under `/tmp`, Python
in a disposable venv; neither is a production manifest change):

```sh
cargo fmt --check
cargo clippy --locked --no-default-features --all-targets --features chat -- -D warnings
cargo test --locked --no-default-features --features chat
cargo build --locked --no-default-features --features chat
cargo kani --lib --no-default-features --harness optional_metal_apis_never_bypass_their_runtime_requirement --harness qwen_attention_widths_match_wide_arithmetic_or_reject --harness qwen_gated_delta_widths_match_wide_arithmetic_or_reject
```

The last command failed because Kani was unavailable. The new symbolic harnesses
are still wired into CI. CBMC was also attempted and unavailable. The temporary
Apple-target manifest has `build=false`, points at the real sources and uses
tokenizers `default-features=false, features=["fancy-regex"]` only to avoid native
cross-build dependencies. `cargo check --all-targets --features mlx,chat` and
strict Clippy for `aarch64-apple-darwin` passed with it. Those logs establish no
FFI linkage, SDK/Metal compilation or native execution.

Python commands used `PYTHONPATH=src:apps/MLXL3Studio/Vendor/PonyExl3`:

```sh
python -m pytest -q tests/test_native_json_process.py tests/test_mtp_bridge.py tests/test_mtp_model_switch.py tests/test_flash_modes.py tests/test_flash_memory_admission.py tests/test_flash_memory.py tests/test_flash_monitor_timeout.py tests/test_package_engine.py
python -m pytest -q --ignore=tests/test_lfm25_modal.py --ignore=tests/test_lfm25_v3_calibration.py --ignore=tests/test_ling_conversion_adapter.py --ignore=tests/test_ling_integration.py -k 'not test_export_preserves_runtime_codebook_markers and not test_source_loader_forces_remote_code_off'
python -m pytest -q tests/test_lfm25_modal.py tests/test_lfm25_v3_calibration.py
ruff check --select F,E9 tests/fake-desktop-engine.py
python -m py_compile tests/fake-desktop-engine.py
```

The targeted run is a subset of the portable run. The two Modal modules in the
last run are separate from that portable selection. `python-initial-*` preserves
the earlier missing-dependency failures instead of discarding them.

C++ existing cache-key regression compiled with
`-std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined
-fno-omit-frame-pointer -I native/ffi native/tests/kernel_cache_key.cpp` and passed.
Its log is not a new benchmark result. The bridge fixture check records its
actual bounded subprocess smoke; Swift parser/lifecycle tests require macOS CI.

The first real macOS link exposed an availability-builtin compiler-rt dependency
missing from Rust's link; `initial-macos-link-error.log` retains the exact failure.
The replacement production parser is compiled by the same sanitizer command with
`native/tests/metal_os_capabilities.cpp`; it checks1,190 cases. The runtime syscall
and native link still require macOS CI. OS flags and their Rust contract remain
unchanged.
