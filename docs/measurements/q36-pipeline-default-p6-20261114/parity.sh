#!/bin/zsh
# Q36-PIPELINE-DEFAULT-P6: CLI, AR/MTP1 bridge and DFlash2 bridge parity, candidate (default) vs witnesses.
set -u
W=/Users/justin/.codex/worktrees/q36-pipeline-default/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-pipeline-default-p6-20261114
PY=$MLXL3_BUILD_PYTHON
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
DRAFT=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-DFlash2
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE
rc=0

timeout 300 build/p6/candidate forward $MLXL3_QWEN_TEST_MODEL --tokens 1,53,106,159 > $D/cli-forward-candidate.log 2>&1; echo cli=$?
$PY - <<EOF | tee $D/cli-forward-compare.log || rc=1
import json, numpy as np
from pathlib import Path
rows=lambda f:[json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
a=rows('/Users/justin/.codex/worktrees/q36-post209-qualified/mix-stq1_0/build/qualified/final-cli.log'); b=rows('$D/cli-forward-candidate.log')
assert [i['token'] for i in a]==[i['token'] for i in b]==[1,53,106,159]
for x,y in zip(a,b,strict=True):
    x,y=[np.asarray(r['logits'],dtype='<f2') for r in (x,y)]
    assert x.shape==y.shape==(248320,) and np.isfinite(x).all() and x.tobytes()==y.tobytes()
print('candidate default: 4 x 248320 finite logits bit-identical to final-cli.log')
EOF

timeout 900 $PY scripts/check-mtp-bridge.py $W/build/p6/candidate $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tokens 1,2,3,17,64,128 --prompt-file $P > $D/bridge-candidate.log 2>&1; echo bridge=$?
$PY - <<EOF | tee $D/bridge-signatures-compare.log || rc=1
import json
from pathlib import Path
def sig(f):
    rows=[json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
    assert rows[-1].get('status')=='validated', rows[-1]
    return {r['budget']:r['token_hash'] for r in rows if 'budget' in r}
a=sig('/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-t1-finish-p2-20261114/bridge-mtp1-baseline.log'); b=sig('$D/bridge-candidate.log')
print(a); print(b); assert a==b and sorted(a)==[1,2,3,17,64,128]
print('candidate default AR/MTP1 token hashes identical to witness 8caeae84 for all budgets')
EOF

for side in baseline candidate; do
  timeout 1200 $PY scripts/check_dflash2.py $W/build/p6/$side $MLXL3_QWEN_TEST_MODEL $DRAFT --skip-tune --timeout 1100 --output $D/dflash-$side.json > $D/dflash-$side.log 2>&1; echo dflash-$side=$?
done
$PY - <<EOF | tee $D/dflash-compare.log || rc=1
import json
from pathlib import Path
def hashes(side):
    r=json.loads(Path('$D/dflash-'+side+'.json').read_text())
    assert r['status']=='complete' and r['parity'] is True, (side, r.get('status'), r.get('error'))
    ev=[(e['request_id'], e['token_hash'], e['stats']['generated_tokens']) for e in r['events'] if e.get('type')=='complete']
    assert ev and all(isinstance(h,str) and len(h)==16 and n>0 for _,h,n in ev)
    return ev
a=hashes('baseline'); b=hashes('candidate')
print(len(a), 'completions each'); assert a==b
print('DFlash2 bridge: candidate default completions identical to baseline (request id, token_hash, count)')
EOF
echo parity_status=$rc
exit $rc
