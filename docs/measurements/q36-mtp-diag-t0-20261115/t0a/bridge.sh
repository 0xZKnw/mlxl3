#!/bin/zsh
# Q36-MTP-ACCEPT-POS-T0A bridge check: A = extracted release 2.0.11 beta 1, B = candidate with histogram counters.
# REL protocol (compare_native, short FR 27 + code 43, ctx default, greedy, prompt cache off, 128 warmup + 128).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115/t0a
P=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115/prompts/b-code.txt
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE MLXL3_MTP_ADAPTIVE
A=$W/build/rel/engine-2.0.11-beta.1/mlxl3
B=$W/build/t0/candidate
shasum -a 256 $A $B | tee $D/bridge-binaries.sha256
rc=0
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/bridge-conditions-before.txt 2>&1
for depth in 1 2 3; do
  timeout 1800 $MLXL3_BUILD_PYTHON benchmarks/compare_native.py $MLXL3_QWEN_TEST_MODEL \
    --baseline $A --candidate $B --mtp-head $MLXL3_MTP_TEST_HEAD --prompt-file $P \
    --tokens 128 --warmup-tokens 128 --repeats 1 --settle-seconds 2 --order ABBA \
    --mtp-depth $depth --campaign-timeout 1700 --output $D/bridge-mtp$depth > $D/bridge-mtp$depth.log 2>&1
  code=$?
  echo "mtp$depth exit=$code"
  [[ $code -eq 0 ]] || rc=1
done
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/bridge-conditions-after.txt 2>&1
$MLXL3_BUILD_PYTHON -I $D/check_bridge.py $D || rc=1
echo bridge_status=$rc
exit $rc
