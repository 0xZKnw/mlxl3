#!/bin/zsh
# Q36-MTP-DEFAULT-T3C static checks, CPU/lib/bin suites, targeted Kani and Python (no GPU model loaded).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
C=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-default-t3c-20261115/checks
cd $W && source build/env.sh >/dev/null
[[ -z "$(git status --porcelain)" ]] || { echo "worktree not clean"; exit 2; }
git rev-parse HEAD > $C/source-commit.txt
unset MLXL3_MTP_POST_NORM MLXL3_MTP_EVICT MLXL3_Q36_HEAD_LANE MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE
rc=0
run() { local name=$1; shift; [[ -e $C/$name.log ]] && { echo "refusing to overwrite $name"; rc=1; return; }; "$@" > $C/$name.log 2>&1; local code=$?; echo "$name exit=$code"; [[ $code -eq 0 ]] || rc=1; }
run fmt cargo fmt --check
run clippy-mlx-chat cargo clippy --all-targets --features mlx,chat -- -D warnings
run clippy-cpu cargo clippy --all-targets --no-default-features -- -D warnings
run clippy-cpu-chat cargo clippy --all-targets --no-default-features --features chat -- -D warnings
run clippy-cpu-hybrid cargo clippy --all-targets --no-default-features --features hybrid-dispatch -- -D warnings
run clippy-direct-metal-chat cargo clippy --all-targets --features direct-metal,chat -- -D warnings
run test-cpu-lib cargo test --lib --no-default-features
run test-cpu-chat cargo test --no-default-features --features chat
run test-cpu-hybrid cargo test --no-default-features --features hybrid-dispatch
run test-mlxchat-lib cargo test --lib --features mlx,chat
run test-mlxchat-bin cargo test --bin mlxl3-rs --features mlx,chat
for h in mtp_post_norm_setting_is_strict mtp_evict_setting_is_strict evict_band_is_total_and_monotone evict_width_is_bounded_and_rejects_invalid_input; do
  run kani-$h cargo kani --lib --no-default-features --harness $h
done
run py-compare-native $MLXL3_BUILD_PYTHON -m pytest -q tests/test_compare_native.py
run py-kani-manifest env PYTHONPATH=. $MLXL3_BUILD_PYTHON -m pytest -q tests/test_kani_manifest.py tests/test_kani_ci.py
run py-native-coverage $MLXL3_BUILD_PYTHON scripts/check_native_coverage.py
run py-native-coverage-tests $MLXL3_BUILD_PYTHON -m unittest discover -s tests -p test_native_coverage.py
run ruff $MLXL3_BUILD_PYTHON -m ruff check benchmarks/compare_native.py tests/test_compare_native.py tests/test_kani_ci.py
grep -h "test result" $C/test-*.log | sort | uniq -c
grep -h "VERIFICATION:\|successfully verified\|failures" $C/kani-*.log
[[ -z "$(git status --porcelain)" ]] || { echo "worktree changed during checks"; rc=1; }
echo checks_status=$rc
exit $rc
