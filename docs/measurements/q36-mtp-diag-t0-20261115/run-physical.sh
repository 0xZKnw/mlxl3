#!/bin/zsh
# Sequential physical campaign, one GPU model at a time: T0a confidence x2, then T0b without and with dylib.
set -u
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
rc=0
zsh $D/physical.sh $D/t0a/run1 $D/prompts 256 128 1 0 || rc=1
sleep 2
zsh $D/physical.sh $D/t0a/run2 $D/prompts 256 128 1 0 || rc=1
sleep 2
$PY -I $D/t0a/analyze_t0a.py $D/t0a/analysis.json $D/t0a/run1 $D/t0a/run2 > $D/t0a/analysis.log 2>&1 || rc=1
zsh $D/physical.sh $D/t0b/nodylib $D/t0b/prompts 128 128 0 0 || rc=1
sleep 2
zsh $D/physical.sh $D/t0b/run1 $D/t0b/prompts 128 128 0 1 || rc=1
sleep 2
zsh $D/physical.sh $D/t0b/run2 $D/t0b/prompts 128 128 0 1 || rc=1
$PY -I $D/t0b/analyze_t0b.py $D/t0b/analysis.json $D/t0b/nodylib $D/t0b/run1 $D/t0b/run2 > $D/t0b/analysis.log 2>&1 || rc=1
echo physical_status=$rc
exit $rc
