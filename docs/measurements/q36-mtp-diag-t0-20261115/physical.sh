#!/bin/zsh
# Q36-MTP T0a/T0b physical runs of mtp::cycle_diagnostics::mtp_cycle_diagnostics (release test binary).
# Usage: physical.sh <out dir> <prompts dir> <tokens> <warmup> <confidence 0|1> <dylib 0|1>
# No /usr/bin/timeout: it is SIP-protected and strips DYLD_INSERT_LIBRARIES (P8). 1200 s zsh watchdog.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
BIN=${T0_PHYSICAL_BIN:-build/t0/physical}
OUT=$1 PROMPTS=$2 TOKENS=$3 WARMUP=$4 CONF=$5 INJECT=$6
DYLIB=/Users/justin/.codex/worktrees/q36-t1-epilogue/mix-stq1_0/build/cbtime/libcb_timing.dylib
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE MLXL3_MTP_ADAPTIVE
mkdir -p $OUT
[[ -e $OUT/diag.json || -e $OUT/cb.bin ]] && { echo "refusing to overwrite $OUT"; exit 2; }
shasum -a 256 $BIN $DYLIB > $OUT/binaries.sha256
EXPECTED_PHYSICAL=${T0_EXPECTED_PHYSICAL:-1df4b0223a34ade8890c4c5d60ec834c5b8f50f6ad5d271f660186d536a3a905}
grep -q "^$EXPECTED_PHYSICAL " $OUT/binaries.sha256 || { echo "test binary changed"; exit 2; }
grep -q '^22c83ffa316279333a237e37b6783d4aa795c5cfb0cc558d4e5d51b2b4097aa5 ' $OUT/binaries.sha256 || { echo "dylib changed"; exit 2; }
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $OUT/conditions-before.txt 2>&1
export MLXL3_MTP_DIAG_PROMPTS=$PROMPTS MLXL3_MTP_DIAG_OUTPUT=$OUT/diag.json MLXL3_MTP_DIAG_TOKENS=$TOKENS \
  MLXL3_MTP_DIAG_WARMUP=$WARMUP MLXL3_MTP_DIAG_CONFIDENCE=$CONF MLXL3_MTP_DIAG_DEPTHS=1,2,3
if [[ $INJECT -eq 1 ]]; then
  DYLD_INSERT_LIBRARIES=$DYLIB MLXL3_CB_TIMING_PATH=$OUT/cb.bin $BIN \
    mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $OUT/test.log 2>&1 &
else
  $BIN mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $OUT/test.log 2>&1 &
fi
pid=$!
( sleep 1200; kill -9 $pid 2>/dev/null ) &
dog=$!
wait $pid
code=$?
kill $dog 2>/dev/null; wait $dog 2>/dev/null
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $OUT/conditions-after.txt 2>&1
rc=0
grep -q 'test result: ok. 1 passed' $OUT/test.log && [[ $code -eq 0 && -s $OUT/diag.json ]] || rc=1
if [[ $INJECT -eq 1 ]]; then
  grep -q 'cb_timing: hooked' $OUT/test.log && [[ -s $OUT/cb.bin ]] || { echo "CB dylib not loaded or no records"; rc=1; }
  grep -q ' 0 short writes' $OUT/test.log || { echo "short writes or missing report"; rc=1; }
else
  grep -q 'cb_timing' $OUT/test.log && { echo "dylib unexpectedly loaded"; rc=1; }
fi
echo "physical $OUT exit=$code status=$rc"
exit $rc
