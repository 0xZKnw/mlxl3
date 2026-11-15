#!/bin/zsh
# Q36-MTP-DEFAULT-T3C physical oracles O1-O5 on the new default, one GPU process at a time (pre-registered in opti.md).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/rel-2.0.11-beta2/mix-stq1_0
T0=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
T1B=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-postnorm-t1b-20261115
T3A=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3a-20261115
T3B=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3b-20261115
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-default-t3c-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
BIN=$W/build/t3c/physical
cd $W && source build/env.sh
[[ -z "$(git status --porcelain)" ]] || { echo "worktree not clean"; exit 2; }
unset MLXL3_MTP_POST_NORM MLXL3_MTP_EVICT MLXL3_MTP_DIAG_HIDDEN MLXL3_MTP_DIAG_COMPACT MLXL3_MTP_ADAPTIVE
rc=0
# Build the release test binary from the committed source.
[[ ! -e $BIN ]] || { echo "$BIN exists; preserve it"; exit 2; }
mkdir -p build/t3c $D/oracles
cargo test --release --no-run --features mlx,chat --lib > $D/oracles/build-release-test.log 2>&1 || { echo "build failed"; exit 1; }
exe=$(sed -n 's/.*Executable unittests native\/src\/lib.rs (\(.*\))/\1/p' $D/oracles/build-release-test.log | tail -1)
[[ -x $exe ]] || { echo "test executable not found"; exit 1; }
cp $exe $BIN
EXPECTED=$(shasum -a 256 $BIN | cut -d' ' -f1)
{ git rev-parse HEAD; shasum -a 256 $BIN; } | tee $D/oracles/binaries.sha256
diag() {  # name confidence, then NAME=VALUE environment
  local name=$1 conf=$2; shift 2
  env "$@" T0_PHYSICAL_BIN=$BIN T0_EXPECTED_PHYSICAL=$EXPECTED zsh $T0/physical.sh $D/oracles/$name $T0/prompts 256 128 $conf 0 || rc=1
  sleep 2
}
check() {  # name, then comparator arguments
  local name=$1; shift
  "$@" > $D/oracles/$name/compare.log 2>&1 || rc=1
  echo "compare $name: $(tail -1 $D/oracles/$name/compare.log)"
}
unit() {  # name test-path, then NAME=VALUE environment
  local name=$1 test=$2; shift 2
  mkdir -p $D/oracles/units
  [[ -e $D/oracles/units/$name.log ]] && { echo "refusing to overwrite $name"; rc=1; return; }
  env "$@" $BIN $test --exact --ignored --nocapture --test-threads=1 > $D/oracles/units/$name.log 2>&1 &
  local pid=$!
  ( sleep 1200; kill -9 $pid 2>/dev/null ) &
  local dog=$!
  wait $pid; local code=$?
  kill $dog 2>/dev/null; wait $dog 2>/dev/null
  if grep -q 'test result: ok. 1 passed' $D/oracles/units/$name.log && [[ $code -eq 0 ]]; then echo "unit $name ok"; else echo "unit $name FAILED ($code)"; rc=1; fi
  sleep 2
}
# O1: variables unset = POST_NORM=1 EVICT=1, exactly the T3b evict1-post1 diagnostic, and the policy recomputed.
diag default 1
check default $PY -I $D/scripts/compare_t3c.py $D/oracles/default/diag.json $T3B/oracles/evict1-post1/diag.json post 1
$PY -I $T3B/scripts/compare_t3b.py $D/oracles/default/diag.json $T3A/post/diag.json evict post > $D/oracles/default/compare-policy.log 2>&1 || rc=1
echo "policy default: $(tail -1 $D/oracles/default/compare-policy.log)"
# O2: both off = previous default (T1b default, pre-norm).
diag off 0 MLXL3_MTP_POST_NORM=0 MLXL3_MTP_EVICT=0
check off $PY -I $T3B/scripts/compare_t3b.py $D/oracles/off/diag.json $T1B/oracles/default/diag.json same pre
# O3: EVICT=0 alone keeps post-norm on: T3a post.
diag evict0 1 MLXL3_MTP_EVICT=0
check evict0 $PY -I $T3B/scripts/compare_t3b.py $D/oracles/evict0/diag.json $T3A/post/diag.json same post
# O4: session and cache exactness with the variables unset.
unit recursive qwen35::tests::mtp_recursive_sessions_match_greedy_and_exact_caches
unit lookup qwen35::tests::mtp_lookup_accepts_rejects_and_repairs_exact_caches
unit cache mtp::native::cache_tests::cache_only_matches_full_head_and_next_prediction
# O5: strictness at Head::load.
mkdir -p $D/oracles/reject
for spec in MLXL3_MTP_EVICT=2:MLXL3_MTP_EVICT MLXL3_MTP_EVICT=true:MLXL3_MTP_EVICT MLXL3_MTP_POST_NORM=2:MLXL3_MTP_POST_NORM; do
  assign=${spec%%:*} var=${spec##*:} tag=${assign//=/-}
  env $assign MLXL3_MTP_DIAG_PROMPTS=$T0/prompts MLXL3_MTP_DIAG_OUTPUT=$D/oracles/reject/never.json \
    $BIN mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $D/oracles/reject/$tag.log 2>&1
  code=$?
  grep -q "$var must be 0 or 1" $D/oracles/reject/$tag.log && [[ $code -ne 0 && ! -e $D/oracles/reject/never.json ]] && echo "reject $tag ok" || { echo "reject $tag FAILED"; rc=1; }
done
[[ -z "$(git status --porcelain)" ]] || { echo "worktree changed"; rc=1; }
echo oracles_status=$rc
exit $rc
