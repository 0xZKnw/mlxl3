#!/bin/zsh
# Q36-PIPELINE-DEFAULT-P6 speed: baseline 6c62ad5f (old default) vs candidate (pipeline default everywhere).
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-pipeline-default/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-pipeline-default-p6-20261114
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
DRAFT=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-DFlash2
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE
A=$W/build/p6/baseline; B=$W/build/p6/candidate
common=(--baseline $A --candidate $B --prompt-file $P --tokens 128 --warmup-tokens 128 --repeats 2 --settle-seconds 2)
rc=0
run() {  # name order extra...
  local name=$1 order=$2; shift 2
  pmset -g batt | head -1 >> $D/speed-power.txt
  timeout 1700 $MLXL3_BUILD_PYTHON benchmarks/compare_native.py $MLXL3_QWEN_TEST_MODEL $common --order $order \
    --campaign-timeout 1600 --output $D/speed-$name-$order "$@" > $D/speed-$name-$order.log 2>&1
  local code=$?
  echo "$name $order exit=$code"
  [[ $code -eq 0 ]] || rc=1
}
run ar BAAB
run ar ABBA
run mtp2 ABBA --mtp-depth 2 --mtp-head $MLXL3_MTP_TEST_HEAD
run dflash3 ABBA --dflash-mode 3 --dflash-draft $DRAFT
run dflash3 BAAB --dflash-mode 3 --dflash-draft $DRAFT
pmset -g batt | head -1 >> $D/speed-power.txt
echo speed_status=$rc
exit $rc
