#!/bin/zsh
# Q36-MTP-EVICT-T3B physical oracles 1-4, one GPU process at a time (pre-registered in opti.md).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
T0=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
T1B=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-postnorm-t1b-20261115
T3A=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3a-20261115
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3b-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
BIN=build/t3b/physical
CAND=$W/build/t3b/candidate
EXPECTED=${1:?expected test binary sha256}
EXPECTED_CAND=${2:?expected candidate sha256}
cd $W && source build/env.sh
unset MLXL3_MTP_POST_NORM MLXL3_MTP_EVICT MLXL3_MTP_DIAG_HIDDEN MLXL3_MTP_DIAG_COMPACT MLXL3_MTP_ADAPTIVE
rc=0
diag() {  # name confidence, then NAME=VALUE environment
  local name=$1 conf=$2; shift 2
  env "$@" T0_PHYSICAL_BIN=$BIN T0_EXPECTED_PHYSICAL=$EXPECTED zsh $T0/physical.sh $D/oracles/$name $T0/prompts 256 128 $conf 0 || rc=1
  sleep 2
}
compare() {  # name reference mode variant
  $PY -I $D/scripts/compare_t3b.py $D/oracles/$1/diag.json $2 $3 $4 > $D/oracles/$1/compare.log 2>&1 || rc=1
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
[[ "$(shasum -a 256 $CAND | cut -d' ' -f1)" == "$EXPECTED_CAND" ]] || { echo "candidate changed"; exit 2; }
# Oracle 1: EVICT absent or 0 leaves pre and post paths unchanged.
diag default 0
compare default $T1B/oracles/default/diag.json same pre
diag evict0 0 MLXL3_MTP_EVICT=0
compare evict0 $T1B/oracles/default/diag.json same pre
diag evict0-post1 1 MLXL3_MTP_EVICT=0 MLXL3_MTP_POST_NORM=1
compare evict0-post1 $T3A/post/diag.json same post
# Oracle 2: EVICT on, post-norm, confidence; tokens equal the AR-checked T3a post tokens.
diag evict1-post1 1 MLXL3_MTP_EVICT=1 MLXL3_MTP_POST_NORM=1
compare evict1-post1 $T3A/post/diag.json evict post
diag evict1-pre 1 MLXL3_MTP_EVICT=1
compare evict1-pre $T1B/oracles/default/diag.json evict pre
# Oracle 3: session/cache exactness with EVICT on (post and pre).
for v in 1 default; do
  if [[ $v == 1 ]]; then set -- MLXL3_MTP_EVICT=1 MLXL3_MTP_POST_NORM=1; else set -- MLXL3_MTP_EVICT=1; fi
  unit recursive-post$v qwen35::tests::mtp_recursive_sessions_match_greedy_and_exact_caches "$@"
  unit lookup-post$v qwen35::tests::mtp_lookup_accepts_rejects_and_repairs_exact_caches "$@"
  unit cache-post$v mtp::native::cache_tests::cache_only_matches_full_head_and_next_prediction "$@"
done
# Strictness at the production entry (Head::load).
mkdir -p $D/oracles/reject
for bad in 2 true; do
  MLXL3_MTP_EVICT=$bad MLXL3_MTP_DIAG_PROMPTS=$T0/prompts MLXL3_MTP_DIAG_OUTPUT=$D/oracles/reject/never.json \
    $BIN mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $D/oracles/reject/evict-$bad.log 2>&1
  code=$?; grep -q 'MLXL3_MTP_EVICT must be 0 or 1' $D/oracles/reject/evict-$bad.log && [[ $code -ne 0 && ! -e $D/oracles/reject/never.json ]] && echo "reject evict-$bad ok" || { echo "reject evict-$bad FAILED"; rc=1; }
done
# Oracle 4: bridge parity MTP1/2/3, cancellation, prefix reuse and Tune key, EVICT 0 then 1 (post-norm).
mkdir -p $D/tune
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/tune/conditions-before.txt 2>&1
for flag in 0 1; do
  MLXL3_MTP_POST_NORM=1 MLXL3_MTP_EVICT=$flag timeout 1800 $MLXL3_BUILD_PYTHON scripts/check-mtp-depths.py $CAND $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tune \
    > $D/tune/tune-evict$flag.jsonl 2> $D/tune/tune-evict$flag.err
  code=$?; echo "tune evict$flag exit=$code"; [[ $code -eq 0 ]] || rc=1
  sleep 2
done
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/tune/conditions-after.txt 2>&1
$MLXL3_BUILD_PYTHON -I - $D/tune <<'PY' | tee $D/tune/tune-summary.txt || rc=1
import json, sys
from pathlib import Path
D = Path(sys.argv[1]); ok = True; keys = {}
for flag in ("0", "1"):
    lines = [json.loads(l) for l in (D / f"tune-evict{flag}.jsonl").read_text().splitlines() if l.startswith("{")]
    ready = [e for e in lines if e.get("type") == "ready"]
    done = [e for e in lines if e.get("type") == "mtp_tune_complete"]
    final = [e for e in lines if e.get("tuner_verified") is True]
    if len(ready) != 1 or len(done) != 1 or len(final) != 1:
        print(flag, "missing ready/tune/final", len(ready), len(done), len(final)); ok = False; continue
    f = final[0]
    for k in ("all_depths_parity", "prefix_parity", "cancel_recovery", "tuner_verified"):
        if f.get(k) is not True:
            print(flag, k, f.get(k)); ok = False
    key = ready[0]["mtp_tuning_key"]; keys[flag] = key
    if f"mtppostnorm-v1=1:mtpevict-v1={flag}:" not in key:
        print(flag, "key lacks flags", key); ok = False
    r = done[0]
    print(f"evict={flag} best_depth={r['best_depth']} key_ok={r['tuning_key'] == key}")
    for row in r["rows"]:
        print(f"  depth {row['depth']} eligible={row['eligible']} decode_tps={row['decode_tps']:.2f} accepted={row.get('accepted_tokens')}")
if len(keys) == 2 and keys["0"] == keys["1"]:
    print("keys collide"); ok = False
print("tune_ok", ok)
sys.exit(0 if ok else 1)
PY
echo oracles_status=$rc
exit $rc
