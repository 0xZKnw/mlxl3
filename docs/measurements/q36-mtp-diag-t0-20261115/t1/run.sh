#!/bin/zsh
# Q36-MTP-POSTNORM-T1: one process per hidden wiring, same prompts/budgets as T0a, confidence off.
set -u
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
EXPECTED=${1:?expected test binary sha256}
rc=0
for v in pre seed chain post; do
  MLXL3_MTP_DIAG_HIDDEN=$v T0_PHYSICAL_BIN=build/t1/physical T0_EXPECTED_PHYSICAL=$EXPECTED zsh $D/physical.sh $D/t1/$v $D/prompts 256 128 0 0 || rc=1
  sleep 2
done
$PY -I $D/t1/analyze_t1.py $D/t1/analysis.json $D/t1 $D/t0a/run1 > $D/t1/analysis.log 2>&1 || rc=1
echo t1_status=$rc
exit $rc
