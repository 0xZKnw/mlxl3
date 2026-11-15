#!/bin/zsh
# Q36-MTP-EVICT-T3B static checks and CPU/lib/bin suites (no GPU model loaded).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
C=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3b-20261115/checks
cd $W && source build/env.sh >/dev/null
rc=0
run() { local name=$1; shift; "$@" > $C/$name.log 2>&1; local code=$?; echo "$name exit=$code"; [[ $code -eq 0 ]] || rc=1; }
run fmt cargo fmt --check
run clippy-mlx-chat cargo clippy --all-targets --features mlx,chat -- -D warnings
run clippy-cpu cargo clippy --all-targets --no-default-features -- -D warnings
run clippy-cpu-chat cargo clippy --all-targets --no-default-features --features chat -- -D warnings
run clippy-cpu-hybrid cargo clippy --all-targets --no-default-features --features hybrid-dispatch -- -D warnings
run clippy-direct-metal-chat cargo clippy --all-targets --features direct-metal,chat -- -D warnings
run test-cpu-lib cargo test --lib --no-default-features
run test-mlxchat-lib cargo test --lib --features mlx,chat
run test-mlxchat-bin cargo test --bin mlxl3-rs --features mlx,chat
grep -h "test result" $C/test-*.log
echo checks_status=$rc
exit $rc
