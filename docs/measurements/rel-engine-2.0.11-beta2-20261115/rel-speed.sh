#!/bin/zsh
# REL-ENGINE-2.0.11-BETA2 speed: A = extracted published 2.0.11 beta 1, B = extracted 2.0.11 beta 2.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/rel-2.0.11-beta2/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/rel-engine-2.0.11-beta2-20261115
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE MLXL3_MTP_POST_NORM MLXL3_MTP_EVICT MLXL3_MTP_ADAPTIVE
A=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0/build/rel/engine-2.0.11-beta.1/mlxl3
B=$W/build/rel/engine-2.0.11-beta.2/mlxl3
shasum -a 256 $A $B | tee $D/speed-binaries.sha256
common=(--baseline $A --candidate $B --mtp-head $MLXL3_MTP_TEST_HEAD --prompt-file $P
  --tokens 128 --warmup-tokens 128 --repeats 2 --settle-seconds 2 --order BAAB)
rc=0
conditions() {
  { date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; top -l 1 -n 6 -o cpu | sed -n 1,18p; } > $1 2>&1
}
conditions $D/speed-conditions-before.txt
for depth in 0 1 2 3; do
  name=$([[ $depth -eq 0 ]] && echo ar || echo mtp$depth)
  pmset -g batt | head -1 >> $D/speed-power.txt
  timeout 2400 $MLXL3_BUILD_PYTHON benchmarks/compare_native.py $MLXL3_QWEN_TEST_MODEL $common \
    --mtp-depth $depth --campaign-timeout 2300 --output $D/speed-$name-BAAB > $D/speed-$name-BAAB.log 2>&1
  code=$?
  echo "$name BAAB exit=$code"
  [[ $code -eq 0 ]] || rc=1
done
pmset -g batt | head -1 >> $D/speed-power.txt
conditions $D/speed-conditions-after.txt
$MLXL3_BUILD_PYTHON - $D <<'PY' | tee $D/speed-summary.txt || rc=1
import json, sys
from pathlib import Path
D = Path(sys.argv[1]); ok = True
for name in ["ar", "mtp1", "mtp2", "mtp3"]:
    path = D / f"speed-{name}-BAAB" / "results.json"
    if not path.exists():
        print(name, "missing results.json"); ok = False; continue
    r = json.loads(path.read_text())
    if r.get("status") != "complete" or r.get("parity") is not True:
        print(name, "status", r.get("status"), "parity", r.get("parity"), r.get("error")); ok = False; continue
    versions = {p["label"]: p["ready"].get("engine_version", p["ready"].get("version")) for p in r["passes"]}
    for prompt, s in r["summary"].items():
        d, pf, tt, el = s["decode_tps"], s["prefill_tps"], s["ttft_seconds"], s["elapsed_seconds"]
        print(f"{name:5s} {prompt:8s} decode A {d['A']:7.2f} B {d['B']:7.2f} ({d['change_percent']:+.2f} %)"
              f" | prefill {pf['change_percent']:+.2f} % | ttft {tt['change_percent']:+.2f} % | elapsed {el['change_percent']:+.2f} %")
        raw = [(p["label"], round(p["prompts"][prompt]["medians"]["decode_tps"], 2)) for p in r["passes"]]
        print(f"{name:5s} {prompt:8s} per-pass decode medians (order) {raw}")
    print(f"{name:5s} ready versions {versions}")
print("summary_ok", ok)
sys.exit(0 if ok else 1)
PY
echo speed_status=$rc
exit $rc
