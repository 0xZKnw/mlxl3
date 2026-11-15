#!/bin/zsh
# Q36-PIPELINE-STRIDE-P7, second launch: re-checks the existing per-k parity logs with a corrected
# stride pattern (the first p7.sh used a BRE "\|" that never matches under zsh), then runs the CB stage unchanged.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-pipeline-default/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-pipeline-stride-p7-20261114
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
DYLIB=/Users/justin/.codex/worktrees/q36-t1-epilogue/mix-stq1_0/build/cbtime/libcb_timing.dylib
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE
rc=0
for k in 2 8 16 64; do
  n=$(grep -c 'pipeline exact' $D/parity-k$k.log)
  grep -qE "pipeline stride=$k( |\$)" $D/parity-k$k.log && s=1 || s=0
  grep -q 'test result: ok. 1 passed' $D/parity-k$k.log && t=1 || t=0
  echo "recheck parity k=$k exact=$n stride_applied=$s test_ok=$t"
  [[ $n -eq 3 && $s -eq 1 && $t -eq 1 ]] || rc=1
done
grep -q 'MLXL3_QWEN_PIPELINE_STRIDE must be' $D/invalid-stride0.log || rc=1
[[ $rc -eq 0 ]] || { echo p7_status=parity-failed; exit 1; }
pmset -g batt | head -1 > $D/power.txt
i=0
for k in 4 8 16 64 2 64 16 8 4; do
  timeout 400 $MLXL3_BUILD_PYTHON $D/cb_bridge.py $W/build/p6/candidate $MLXL3_QWEN_TEST_MODEL $P $D/p$i-k$k $k 1 $DYLIB
  code=$?
  echo "run p$i k=$k exit=$code"
  [[ $code -eq 0 ]] || rc=1
  i=$((i+1))
  sleep 2
done
pmset -g batt | head -1 >> $D/power.txt
echo p7_status=$rc
exit $rc
