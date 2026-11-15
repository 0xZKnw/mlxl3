#!/bin/zsh
# Q36-HEAD-LANE-P24: real CLI and bridge paths of the release candidate, flag 1 (default) and flag 0.
set -u
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-head-lane-p24-20261115
PY=$MLXL3_BUILD_PYTHON
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
DRAFT=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-DFlash2
REF_CLI=/Users/justin/.codex/worktrees/q36-post209-qualified/mix-stq1_0/build/qualified/final-cli.log
REF_BRIDGE=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-t1-finish-p2-20261114/bridge-mtp1-baseline.log
B=$W/build/p24/candidate
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE
rc=0
shasum -a 256 $B | tee $D/parity-binary.sha256

# 1. CLI logits (M=1 path) against the qualified witness, unset (default), 1 and 0.
for flag in unset 1 0; do
  if [[ $flag == unset ]]; then
    timeout 300 $B forward $MLXL3_QWEN_TEST_MODEL --tokens 1,53,106,159 > $D/cli-flag$flag.log 2>&1; echo cli-flag$flag=$?
  else
    MLXL3_Q36_HEAD_LANE=$flag timeout 300 $B forward $MLXL3_QWEN_TEST_MODEL --tokens 1,53,106,159 > $D/cli-flag$flag.log 2>&1; echo cli-flag$flag=$?
  fi
  $PY - $REF_CLI $D/cli-flag$flag.log <<'PY' | tee $D/cli-flag$flag-compare.log || rc=1
import json, sys, numpy as np
from pathlib import Path
rows=lambda f:[json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
a=rows(sys.argv[1]); b=rows(sys.argv[2])
assert [i['token'] for i in a]==[i['token'] for i in b]==[1,53,106,159]
for x,y in zip(a,b,strict=True):
    x,y=[np.asarray(r['logits'],dtype='<f2') for r in (x,y)]
    assert x.shape==y.shape==(248320,) and np.isfinite(x).all() and x.tobytes()==y.tobytes()
print('4 x 248320 finite logits bit-identical to final-cli.log')
PY
done
for bad in 2 "" true " 1"; do
  MLXL3_Q36_HEAD_LANE=$bad timeout 120 $B forward $MLXL3_QWEN_TEST_MODEL --tokens 1 > $D/cli-flag-invalid.log 2>&1; code=$?
  echo "cli-flag-invalid [$bad] exit=$code $(grep -o 'MLXL3_Q36_HEAD_LANE must be 0 or 1' $D/cli-flag-invalid.log | head -1)" | tee -a $D/cli-flag-invalid-summary.log
  [[ $code -ne 0 ]] && grep -q "MLXL3_Q36_HEAD_LANE must be 0 or 1" $D/cli-flag-invalid.log || rc=1
done

# 2. AR/MTP1 bridge hashes against the witness (flag default).
timeout 900 $PY scripts/check-mtp-bridge.py $B $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tokens 1,2,3,17,64,128 --prompt-file $P > $D/bridge-mtp1.log 2>&1; echo bridge=$?
$PY - $REF_BRIDGE $D/bridge-mtp1.log <<'PY' | tee $D/bridge-signatures-compare.log || rc=1
import json, sys
from pathlib import Path
def sig(f):
    rows=[json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
    assert rows[-1].get('status')=='validated', rows[-1]
    return {r['budget']:r['token_hash'] for r in rows if 'budget' in r}
a=sig(sys.argv[1]); b=sig(sys.argv[2])
print(a); print(b); assert a==b and sorted(a)==[1,2,3,17,64,128] and b[128]=='04d09d6803d2329b'
print('default AR/MTP1 token hashes identical to the witness for all budgets')
PY

# 3. MTP1/2/3 against AR (budgets, cancellation, prefix, tuner), flag 1 and flag 0.
for flag in 1 0; do
  MLXL3_Q36_HEAD_LANE=$flag timeout 1800 $PY scripts/check-mtp-depths.py $B $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tune > $D/depths-flag$flag.log 2>&1; echo depths-flag$flag=$?
done
$PY - $D/depths-flag0.log $D/depths-flag1.log <<'PY' | tee $D/depths-compare.log || rc=1
import json, sys
from pathlib import Path
def load(f, flag):
    rows=[json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
    final=rows[-1]
    assert final=={"all_depths_parity":True,"prefix_parity":True,"cancel_recovery":True,"tuner_verified":True}, final
    ready=[r for r in rows if r.get('type')=='ready']
    assert len(ready)==1 and f':q36headlane-v1={flag}:' in ready[0]['mtp_tuning_key'], ready
    done=[(r['request_id'], r['token_hash'], r['stats']['generated_tokens']) for r in rows if r.get('type')=='complete']
    assert done and all(isinstance(h,str) and len(h)==16 for _,h,_ in done)
    tune=[r for r in rows if r.get('type')=='mtp_tune_complete']
    assert len(tune)==1
    return done, tune[0]
a,ta=load(sys.argv[1],0); b,tb=load(sys.argv[2],1)
assert a==b, 'flag 1 completions differ from flag 0'
assert [r['token_hashes'] for r in ta['rows']]==[r['token_hashes'] for r in tb['rows']]
print(len(a),'completions identical (request id, token_hash, count); tuner hashes identical')
print('tune flag0', [(r['depth'], round(r['decode_tps'],2), r['accepted_tokens']) for r in ta['rows']], 'best', ta['best_depth'])
print('tune flag1', [(r['depth'], round(r['decode_tps'],2), r['accepted_tokens']) for r in tb['rows']], 'best', tb['best_depth'])
PY

# 4. DFlash2 bridge, flag 0 against flag 1.
for flag in 0 1; do
  MLXL3_Q36_HEAD_LANE=$flag timeout 1200 $PY scripts/check_dflash2.py $B $MLXL3_QWEN_TEST_MODEL $DRAFT --skip-tune --timeout 1100 --output $D/dflash-flag$flag.json > $D/dflash-flag$flag.log 2>&1; echo dflash-flag$flag=$?
done
$PY - $D/dflash-flag0.json $D/dflash-flag1.json <<'PY' | tee $D/dflash-compare.log || rc=1
import json, sys
from pathlib import Path
def hashes(f):
    r=json.loads(Path(f).read_text())
    assert r['status']=='complete' and r['parity'] is True, (f, r.get('status'), r.get('error'))
    ev=[(e['request_id'], e['token_hash'], e['stats']['generated_tokens']) for e in r['events'] if e.get('type')=='complete']
    assert ev and all(isinstance(h,str) and len(h)==16 and n>0 for _,h,n in ev)
    return ev
a=hashes(sys.argv[1]); b=hashes(sys.argv[2])
print(len(a), 'completions each'); assert a==b
print('DFlash2 bridge: flag 1 completions identical to flag 0 (request id, token_hash, count)')
PY
echo parity_status=$rc
exit $rc
