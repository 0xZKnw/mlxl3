#!/bin/zsh
# Q36-HEAD-LANE-P24 speed: same release binary, A = MLXL3_Q36_HEAD_LANE=0, B = MLXL3_Q36_HEAD_LANE=1.
# Pre-registered in opti.md: MTP2 BAAB then ABBA, MTP3 BAAB then ABBA, MTP1 BAAB control.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-head-lane-p24-20261115
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE
B=$W/build/p24/candidate
shasum -a 256 $B | tee $D/speed-binary.sha256
common=(--baseline $B --candidate $B --baseline-env MLXL3_Q36_HEAD_LANE=0 --candidate-env MLXL3_Q36_HEAD_LANE=1
  --mtp-head $MLXL3_MTP_TEST_HEAD --prompt-file $P --tokens 128 --warmup-tokens 128 --repeats 2 --settle-seconds 2)
rc=0
conditions() {
  { date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; top -l 1 -n 6 -o cpu | sed -n 1,18p; } > $1 2>&1
}
conditions $D/speed-conditions-before.txt
run() {  # depth order
  local depth=$1 order=$2 name=mtp$1-$2
  pmset -g batt | head -1 >> $D/speed-power.txt
  timeout 2400 $MLXL3_BUILD_PYTHON benchmarks/compare_native.py $MLXL3_QWEN_TEST_MODEL $common --order $order \
    --mtp-depth $depth --campaign-timeout 2300 --output $D/speed-$name > $D/speed-$name.log 2>&1
  local code=$?
  echo "$name exit=$code"
  [[ $code -eq 0 ]] || rc=1
}
run 2 BAAB
run 2 ABBA
run 3 BAAB
run 3 ABBA
run 1 BAAB
pmset -g batt | head -1 >> $D/speed-power.txt
conditions $D/speed-conditions-after.txt
$MLXL3_BUILD_PYTHON - $D <<'PY' | tee $D/speed-summary.txt || rc=1
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
ok = True
for name in ["mtp2-BAAB", "mtp2-ABBA", "mtp3-BAAB", "mtp3-ABBA", "mtp1-BAAB"]:
    path = D / f"speed-{name}" / "results.json"
    if not path.exists():
        print(name, "missing results.json"); ok = False; continue
    r = json.loads(path.read_text())
    if r.get("status") != "complete" or r.get("parity") is not True:
        print(name, "status", r.get("status"), "parity", r.get("parity"), r.get("error")); ok = False; continue
    env = {k: v["environment"] for k, v in r["binaries"].items()}
    assert env == {"A": {"MLXL3_Q36_HEAD_LANE": "0"}, "B": {"MLXL3_Q36_HEAD_LANE": "1"}}, env
    assert r["binaries"]["A"]["sha256"] == r["binaries"]["B"]["sha256"]
    keys = {p["label"]: p["ready"].get("mtp_tuning_key", "") for p in r["passes"]}
    for prompt, s in r["summary"].items():
        d, pf, tt = s["decode_tps"], s["prefill_tps"], s["ttft_seconds"]
        print(f"{name:10s} {prompt:8s} decode A {d['A']:7.2f} B {d['B']:7.2f} ({d['change_percent']:+.2f} %)"
              f" | prefill {pf['change_percent']:+.2f} % | ttft {tt['change_percent']:+.2f} %")
    for prompt in r["summary"]:
        raw = [(p["label"], round(p["prompts"][prompt]["medians"]["decode_tps"], 2)) for p in r["passes"]]
        print(f"{name:10s} {prompt:8s} per-pass decode medians (order) {raw}")
print("summary_ok", ok)
sys.exit(0 if ok else 1)
PY
echo speed_status=$rc
exit $rc
