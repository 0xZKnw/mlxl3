#!/bin/zsh
# Q36-PIPELINE-STRIDE-P7: parity per stride, then CB diagnostic k=4,8,16,64,2,64,16,8,4 (prompt code).
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
  MLXL3_QWEN_PIPELINE_STRIDE=$k timeout 600 build/p6/physical qwen35::tests::layer_pipeline_matches_unscheduled_logits_and_states \
    --exact --ignored --nocapture --test-threads=1 > $D/parity-k$k.log 2>&1
  code=$?
  echo "parity k=$k exit=$code $(grep -c 'pipeline exact' $D/parity-k$k.log) exact $(grep -o 'pipeline stride=[0-9]*' $D/parity-k$k.log)"
  [[ $code -eq 0 && $(grep -c 'pipeline exact' $D/parity-k$k.log) -eq 3 ]] || rc=1
  grep -q "pipeline stride=$k\$\|pipeline stride=$k " $D/parity-k$k.log || { echo "stride k=$k not applied"; rc=1; }
done
MLXL3_QWEN_PIPELINE_STRIDE=0 timeout 120 build/p6/candidate forward $MLXL3_QWEN_TEST_MODEL --tokens 1 > $D/invalid-stride0.log 2>&1
echo "invalid stride 0 exit=$? $(grep -c 'MLXL3_QWEN_PIPELINE_STRIDE must be' $D/invalid-stride0.log)"
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
