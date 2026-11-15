#!/bin/zsh
# Q36-GPU-ATTRIBUTION-P8: two independent processes of the timeline test with CB timing, then analysis.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-pipeline-default/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-gpu-attribution-p8-20261115
DYLIB=/Users/justin/.codex/worktrees/q36-t1-epilogue/mix-stq1_0/build/cbtime/libcb_timing.dylib
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE
rc=0
pmset -g batt | head -1 > $D/power.txt
for r in run1 run2; do
  mkdir -p $D/$r
  # No /usr/bin/timeout here: it is SIP-protected and strips DYLD_INSERT_LIBRARIES (first launch recorded no CB).
  DYLD_INSERT_LIBRARIES=$DYLIB MLXL3_CB_TIMING_PATH=$D/$r/cb.bin MLXL3_Q36_TIMELINE_PATH=$D/$r/timeline.json \
    build/p8/physical qwen35::tests::q36_decode_block_timeline --exact --ignored --nocapture --test-threads=1 > $D/$r/test.log 2>&1 &
  pid=$!
  ( sleep 900; kill -9 $pid 2>/dev/null ) &
  dog=$!
  wait $pid
  code=$?
  kill $dog 2>/dev/null; wait $dog 2>/dev/null
  grep -q 'cb_timing: hooked' $D/$r/test.log && [[ -s $D/$r/cb.bin ]] || { echo "p8 $r: CB dylib not loaded or no records"; rc=1; }
  echo "p8 $r exit=$code $(grep -c 'test result: ok. 1 passed' $D/$r/test.log) passed"
  [[ $code -eq 0 ]] && grep -q 'test result: ok. 1 passed' $D/$r/test.log || rc=1
  sleep 2
done
pmset -g batt | head -1 >> $D/power.txt
[[ $rc -eq 0 ]] && { $MLXL3_BUILD_PYTHON $D/analyze_timeline.py $D/run1 $D/run2 > $D/analysis.log 2>&1 || rc=1; }
echo p8_status=$rc
exit $rc
