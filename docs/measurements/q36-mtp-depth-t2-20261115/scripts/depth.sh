#!/bin/zsh
# Q36-MTP-DEPTH-T2: same release binary, MLXL3_MTP_POST_NORM=1 in both arms.
# A = fixed D1; B in {D2, D3, ADA (MLXL3_MTP_ADAPTIVE=1, start depth 1)}. Pre-registered in opti.md.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-depth-t2-20261115/runs; mkdir -p $D
PR=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115/prompts
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE MLXL3_MTP_POST_NORM MLXL3_MTP_ADAPTIVE MLXL3_MTP_LOOKUP
B=$W/build/t1b/candidate
[[ "$(shasum -a 256 $B | cut -d" " -f1)" == "${1:?expected candidate sha256}" ]] || { echo "candidate changed"; exit 2; }
shasum -a 256 $B $PR/b-code.txt $PR/c-prose-fr.txt $PR/d-doc-summary.txt benchmarks/compare_native.py | tee $D/inputs.sha256
rc=0
conditions() {
  { date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; top -l 1 -n 6 -o cpu | sed -n 1,18p; } > $1 2>&1
}
conditions $D/conditions-before.txt
run() {  # document arm order
  local doc=$1 arm=$2 order=$3 name=$1-$2-$3
  local extra=()
  case $arm in
    d2) extra=(--candidate-mtp-depth 2) ;;
    d3) extra=(--candidate-mtp-depth 3) ;;
    ada) extra=(--candidate-env MLXL3_MTP_ADAPTIVE=1) ;;
  esac
  pmset -g batt | head -1 >> $D/power.txt
  timeout 2400 $MLXL3_BUILD_PYTHON benchmarks/compare_native.py $MLXL3_QWEN_TEST_MODEL \
    --baseline $B --candidate $B --baseline-env MLXL3_MTP_POST_NORM=1 --candidate-env MLXL3_MTP_POST_NORM=1 $extra \
    --mtp-depth 1 --mtp-head $MLXL3_MTP_TEST_HEAD --prompt-file $PR/$doc.txt --tokens 256 --warmup-tokens 128 \
    --repeats 2 --settle-seconds 2 --order $order --campaign-timeout 2300 --output $D/$name > $D/$name.log 2>&1
  local code=$?
  echo "$name exit=$code"
  [[ $code -eq 0 ]] || rc=1
}
for doc in b-code c-prose-fr d-doc-summary; do
  for arm in d2 d3 ada; do
    run $doc $arm BAAB
    run $doc $arm ABBA
  done
done
pmset -g batt | head -1 >> $D/power.txt
conditions $D/conditions-after.txt
echo "depth_status=$rc"
exit $rc
