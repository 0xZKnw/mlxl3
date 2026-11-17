#!/bin/zsh
# D5 (MTP at the Qwen3.6 catalog presets), then the extra D4 screens (CPU).
M=/Users/justin/Documents/mix-stq1_0/docs/measurements/decode-research-r2-20261116
W=/Users/justin/.codex/worktrees/lfm3-prompt-lookup/mix-stq1_0
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
ENGINE=$W/build/engine-v2.0.13-check/mlxl3
LFM3="$HOME/Library/Application Support/io.mlxl3.desktop/Models/LFM3-1.7B-EXL3--8bpw"
Q36=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-EXL3-2.49bpw
MTP=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-MTP-4bit
cond() { { date; uptime; pmset -g batt | tail -1; } >> $M/$1 2>&1 }
# Wait for the previous chain: its last step writes sim-q36-d2.json (or the 35B run was skipped/failed).
until [ -f $M/sim-q36-d2.json ] || { grep -q '"status"' $M/d2-q36.out 2>/dev/null && ! pgrep -f run_after_d2.sh > /dev/null; }; do sleep 5; done
sleep 3
if pgrep -f 'engine-v2.0.13-check/mlxl3|mlxl3-rs bridge' > /dev/null; then
  echo "phase d5 skipped: an engine process is running"
else
  cond d5-conditions.txt
  $PY $M/d2.py --scripts $W/scripts --engine $ENGINE --model $Q36 --mtp-head $MTP --out $M/d5-q36-presets --plan q36-presets --tokens 256 --cases fr,code,chat,edit > $M/d5-q36-presets.out 2>&1
  echo "phase d5 exit=$?"
  cond d5-conditions.txt
fi
$PY $M/d4.py analyze --model "$LFM3" --out $M/d4 --proxies int3_g64,int3_g32,int4_g32,int4_g128 --report analysis-extra.json > $M/d4-analyze-extra.log 2>&1
echo "phase d4-extra exit=$?"
echo "phase d5 chain done"
