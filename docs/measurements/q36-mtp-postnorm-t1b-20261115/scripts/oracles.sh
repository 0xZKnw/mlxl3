#!/bin/zsh
# Q36-MTP-POSTNORM-T1B physical oracles, one GPU process at a time (pre-registered in opti.md).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
T0=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-postnorm-t1b-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
BIN=${T1B_BIN:-build/t1b/physical}
EXPECTED=${1:?expected test binary sha256}
cd $W && source build/env.sh
unset MLXL3_MTP_POST_NORM MLXL3_MTP_DIAG_HIDDEN MLXL3_MTP_DIAG_COMPACT MLXL3_MTP_ADAPTIVE
rc=0
diag() {  # name, then NAME=VALUE environment
  local name=$1; shift
  env "$@" T0_PHYSICAL_BIN=$BIN T0_EXPECTED_PHYSICAL=$EXPECTED zsh $T0/physical.sh $D/oracles/$name $T0/prompts 256 128 0 0 || rc=1
  sleep 2
}
compare() {  # name reference mode variant
  $PY -I $D/scripts/compare_diag.py $D/oracles/$1/diag.json $2 $3 $4 > $D/oracles/$1/compare.log 2>&1 || rc=1
  echo "compare $1: $(tail -1 $D/oracles/$1/compare.log)"
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
[[ "$(shasum -a 256 $BIN | cut -d' ' -f1)" == "$EXPECTED" ]] || { echo "test binary changed"; exit 2; }
diag default
compare default $T0/t1/pre/diag.json same pre
diag post0 MLXL3_MTP_POST_NORM=0
compare post0 $T0/t1/pre/diag.json same pre
diag post1 MLXL3_MTP_POST_NORM=1
compare post1 $T0/t1/post/diag.json same post
diag seed MLXL3_MTP_DIAG_HIDDEN=seed
compare seed $T0/t1/seed/diag.json same seed
diag chain MLXL3_MTP_DIAG_HIDDEN=chain
compare chain $T0/t1/chain/diag.json same chain
diag full-pre MLXL3_MTP_DIAG_COMPACT=0
diag full-post MLXL3_MTP_DIAG_COMPACT=0 MLXL3_MTP_POST_NORM=1
for v in 1 default; do
  if [[ $v == 1 ]]; then set -- MLXL3_MTP_POST_NORM=1; else set -- MLXL3_MTP_TEST_DUMMY=1; fi
  unit recursive-post$v qwen35::tests::mtp_recursive_sessions_match_greedy_and_exact_caches "$@"
  unit lookup-post$v qwen35::tests::mtp_lookup_accepts_rejects_and_repairs_exact_caches "$@"
  unit cache-post$v mtp::native::cache_tests::cache_only_matches_full_head_and_next_prediction "$@"
done
# Strictness at the production entry (Head::load), then the diagnostic exclusivity guard.
mkdir -p $D/oracles/reject
MLXL3_MTP_POST_NORM=2 MLXL3_MTP_DIAG_PROMPTS=$T0/prompts MLXL3_MTP_DIAG_OUTPUT=$D/oracles/reject/never.json \
  $BIN mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $D/oracles/reject/post-norm-2.log 2>&1
code=$?; grep -q 'MLXL3_MTP_POST_NORM must be 0 or 1' $D/oracles/reject/post-norm-2.log && [[ $code -ne 0 && ! -e $D/oracles/reject/never.json ]] && echo "reject post-norm-2 ok" || { echo "reject post-norm-2 FAILED"; rc=1; }
MLXL3_MTP_POST_NORM=1 MLXL3_MTP_DIAG_HIDDEN=seed MLXL3_MTP_DIAG_PROMPTS=$T0/prompts MLXL3_MTP_DIAG_OUTPUT=$D/oracles/reject/never.json \
  $BIN mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $D/oracles/reject/exclusive.log 2>&1
code=$?; grep -q 'are exclusive' $D/oracles/reject/exclusive.log && [[ $code -ne 0 ]] && echo "reject exclusive ok" || { echo "reject exclusive FAILED"; rc=1; }
echo oracles_status=$rc
exit $rc
