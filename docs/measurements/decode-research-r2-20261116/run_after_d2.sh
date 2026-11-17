#!/bin/zsh
# Sequential follow-up: D4 collect + analyze (LFM3), then D2 on the 35B, then the simulations.
# One GPU model at a time; each phase logs its own status.
M=/Users/justin/Documents/mix-stq1_0/docs/measurements/decode-research-r2-20261116
W=/Users/justin/.codex/worktrees/lfm3-prompt-lookup/mix-stq1_0
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
ENGINE=$W/build/engine-v2.0.13-check/mlxl3
LFM3="$HOME/Library/Application Support/io.mlxl3.desktop/Models/LFM3-1.7B-EXL3--8bpw"
Q36=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-EXL3-2.49bpw
MTP=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-MTP-4bit
cond() { { date; uptime; pmset -g batt | tail -1; ps aux | grep -E 'mlxl3|swift-frontend|cargo|rustc' | grep -v grep | awk '{print $3, $11}' | sort -rn | head -5; } >> $M/$1 2>&1 }
until grep -q 'exit=' $M/d2-lfm3.out 2>/dev/null; do sleep 5; done
echo "phase d2-lfm3 $(grep exit= $M/d2-lfm3.out)"
cond d4-conditions.txt
$PY $M/d4.py collect --engine $ENGINE --model "$LFM3" --d2 $M/d2-lfm3/results.json --d2-script $M/d2.py --out $M/d4 > $M/d4-collect.log 2>&1
echo "phase d4-collect exit=$?"
$PY $M/d4.py analyze --model "$LFM3" --out $M/d4 > $M/d4-analyze.log 2>&1
echo "phase d4-analyze exit=$?"
cond d4-conditions.txt
if ps aux | grep -E '[m]lxl3(-rs)? ' | grep -v grep > /dev/null; then
  echo "phase d2-q36 skipped: another engine process is running"
else
  cond d2-q36-conditions.txt
  $PY $M/d2.py --scripts $W/scripts --engine $ENGINE --model $Q36 --mtp-head $MTP --out $M/d2-q36 --plan q36 --tokens 256 --cases fr,code,chat,edit > $M/d2-q36.out 2>&1
  echo "phase d2-q36 exit=$?"
  cond d2-q36-conditions.txt
fi
$PY $M/sim.py --tokenizer "$LFM3/tokenizer.json" --d2 $M/d2-lfm3/results.json --d2-script $M/d2.py --labels A --costs lfm3_p300,lfm3_p2000,lfm3_hyp8 --out $M/sim-lfm3-d2.json > $M/sim-lfm3-d2.log 2>&1
echo "phase sim-lfm3 exit=$?"
if [ -f $M/d2-q36/results.json ]; then
  $PY $M/sim.py --tokenizer $Q36/tokenizer.json --d2 $M/d2-q36/results.json --d2-script $M/d2.py --labels AR --costs q36 --out $M/sim-q36-d2.json > $M/sim-q36-d2.log 2>&1
  echo "phase sim-q36 exit=$?"
fi
echo "phase all done"
