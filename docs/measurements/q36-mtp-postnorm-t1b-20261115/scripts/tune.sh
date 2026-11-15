#!/bin/zsh
# Q36-MTP-POSTNORM-T1B indicative Tune pass: check-mtp-depths.py --tune, flag 0 then 1, one GPU process at a time.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-postnorm-t1b-20261115/tune; mkdir -p $D
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE MLXL3_MTP_POST_NORM MLXL3_MTP_ADAPTIVE
B=$W/build/t1b/candidate
[[ "$(shasum -a 256 $B | cut -d" " -f1)" == "${1:?expected candidate sha256}" ]] || { echo "candidate changed"; exit 2; }
rc=0
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/conditions-before.txt 2>&1
for flag in 0 1; do
  pmset -g batt | head -1 >> $D/power.txt
  MLXL3_MTP_POST_NORM=$flag timeout 1800 $MLXL3_BUILD_PYTHON scripts/check-mtp-depths.py $B $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tune \
    > $D/tune-post$flag.jsonl 2> $D/tune-post$flag.err
  code=$?
  echo "post$flag exit=$code"
  [[ $code -eq 0 ]] || rc=1
done
{ date; pmset -g batt | head -2; pmset -g therm; sysctl vm.swapusage; } > $D/conditions-after.txt 2>&1
$MLXL3_BUILD_PYTHON -I - $D <<'PY' | tee $D/tune-summary.txt || rc=1
import json, sys
from pathlib import Path
D = Path(sys.argv[1]); ok = True; keys = {}
for flag in ("0", "1"):
    lines = [json.loads(l) for l in (D / f"tune-post{flag}.jsonl").read_text().splitlines() if l.startswith("{")]
    ready = [e for e in lines if e.get("type") == "ready"]
    done = [e for e in lines if e.get("type") == "mtp_tune_complete"]
    final = [e for e in lines if e.get("tuner_verified") is True]
    if len(ready) != 1 or len(done) != 1 or len(final) != 1:
        print(flag, "missing ready/tune/final", len(ready), len(done), len(final)); ok = False; continue
    key = ready[0]["mtp_tuning_key"]; keys[flag] = key
    if f"mtppostnorm-v1={flag}" not in key:
        print(flag, "key lacks flag", key); ok = False
    r = done[0]
    print(f"post={flag} best_depth={r['best_depth']} key_ok={r['tuning_key'] == key}")
    for row in r["rows"]:
        print(f"  depth {row['depth']} eligible={row['eligible']} decode_tps={row['decode_tps']:.2f} accepted={row.get('accepted_tokens')}")
if len(keys) == 2 and keys["0"] == keys["1"]:
    print("keys collide"); ok = False
print("tune_ok", ok)
sys.exit(0 if ok else 1)
PY
echo "tune_status=$rc"
exit $rc
