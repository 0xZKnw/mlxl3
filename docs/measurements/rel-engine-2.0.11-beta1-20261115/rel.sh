#!/bin/zsh
# REL-ENGINE-2.0.11-BETA1: archive integrity and real-path correctness of the extracted engine.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
cd $W && source build/env.sh
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/rel-engine-2.0.11-beta1-20261115
P24=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-head-lane-p24-20261115
PY=$MLXL3_BUILD_PYTHON
P=/Users/justin/.codex/worktrees/q36-decode-r5-live/mix-stq1_0/docs/measurements/mtp-dense-1.4.0/code-prompt.txt
DRAFT=/Users/justin/Documents/mix-stq1_0/models/Qwen3.6-35B-A3B-DFlash2
REF_CLI=/Users/justin/.codex/worktrees/q36-post209-qualified/mix-stq1_0/build/qualified/final-cli.log
REF_BRIDGE=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-t1-finish-p2-20261114/bridge-mtp1-baseline.log
ARCHIVE=$W/dist/MLXL3-Engine-v2.0.11-beta.1-arm64.tar.gz
X=$W/build/rel/engine-2.0.11-beta.1
B=$X/mlxl3
SOURCE=$(git rev-parse HEAD)
unset MLXL3_QWEN_PIPELINE MLXL3_QWEN_PIPELINE_STRIDE MLXL3_Q36_HEAD_LANE
rc=0
fail() { echo "FAIL: $*"; rc=1; }

# 1. Archive, manifest, signatures, runtime identity.
[[ ! -e $X ]] || { echo "extraction directory exists; preserve it"; exit 2; }
mkdir -p $X
shasum -a 256 $ARCHIVE $ARCHIVE.sha256 | tee $D/archive.sha256
(cd ${ARCHIVE:h} && shasum -a 256 -c ${ARCHIVE:t}.sha256) | tee $D/archive-check.log || fail "archive checksum file"
tar -xzf $ARCHIVE -C $X || fail "extract"
ls -la $X | tee $D/extracted-files.txt
$PY - $X $SOURCE <<'PY' | tee $D/manifest-check.log || fail "manifest"
import hashlib, json, sys
from pathlib import Path
x = Path(sys.argv[1]); m = json.loads((x / "engine.json").read_text())
assert m["version"] == "2.0.11" and m["release_tag"] == "engine-v2.0.11-beta.1" and m["architecture"] == "arm64", m
assert set(m["files"]) == {"mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"}
assert {p.name for p in x.iterdir()} == set(m["files"]) | {"engine.json"}, sorted(p.name for p in x.iterdir())
for name, rec in sorted(m["files"].items()):
    data = (x / name).read_bytes()
    assert len(data) == rec["size"] and hashlib.sha256(data).hexdigest() == rec["sha256"], name
    print(f"{name} {rec['size']} {rec['sha256']}")
print("manifest ok", m["version"], m["release_tag"], "minimum_app", m["minimum_app_version"], "minimum_macos", m["minimum_macos"])
PY
: > $D/signatures.log
for f in mlxl3 libmlx.dylib libjaccl.dylib; do
  if codesign --verify --strict $X/$f >> $D/signatures.log 2>&1 && lipo $X/$f -verify_arch arm64 >> $D/signatures.log 2>&1; then
    echo "$f signature+arm64 ok" | tee -a $D/signatures.log
  else
    fail "$f signature/arch"
  fi
done
$B runtime-info > $D/runtime-info.json 2> $D/runtime-info.stderr || fail "runtime-info"
$PY - $D/runtime-info.json $SOURCE <<'PY' | tee $D/runtime-info-check.log || fail "runtime identity"
import json, sys
info = json.load(open(sys.argv[1])); source = sys.argv[2]
assert info["version"] == "2.0.11" and info["mlx_enabled"] is True and info["chat_enabled"] is True, info
assert info["architecture"] == "aarch64", info
assert source.startswith(info["commit"]) or info["commit"].startswith(source[:12]), (info["commit"], source)
print("runtime ok", {k: info[k] for k in ("version", "commit", "backend", "architecture")})
PY

# 2. CLI logits against the qualified witness.
timeout 300 $B forward $MLXL3_QWEN_TEST_MODEL --tokens 1,53,106,159 > $D/cli.log 2>&1 || fail "cli exit"
$PY - $REF_CLI $D/cli.log <<'PY' | tee $D/cli-compare.log || fail "cli logits"
import json, sys, numpy as np
from pathlib import Path
rows = lambda f: [json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
a = rows(sys.argv[1]); b = rows(sys.argv[2])
assert [i['token'] for i in a] == [i['token'] for i in b] == [1, 53, 106, 159]
for x, y in zip(a, b, strict=True):
    x, y = [np.asarray(r['logits'], dtype='<f2') for r in (x, y)]
    assert x.shape == y.shape == (248320,) and np.isfinite(x).all() and x.tobytes() == y.tobytes()
print('4 x 248320 finite logits bit-identical to final-cli.log')
PY

# 3. AR/MTP1 bridge hashes against the witness.
timeout 900 $PY scripts/check-mtp-bridge.py $B $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tokens 1,2,3,17,64,128 --prompt-file $P > $D/bridge-mtp1.log 2>&1 || fail "bridge exit"
$PY - $REF_BRIDGE $D/bridge-mtp1.log <<'PY' | tee $D/bridge-compare.log || fail "bridge hashes"
import json, sys
from pathlib import Path
def sig(f):
    rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
    assert rows[-1].get('status') == 'validated', rows[-1]
    return {r['budget']: r['token_hash'] for r in rows if 'budget' in r}
a = sig(sys.argv[1]); b = sig(sys.argv[2])
assert a == b and sorted(a) == [1, 2, 3, 17, 64, 128] and b[128] == '04d09d6803d2329b', (a, b)
print('AR/MTP1 token hashes identical to the witness for budgets', sorted(b))
PY

# 4. MTP depths, cancellation, prefix and tuner against P24 flag 1.
timeout 1800 $PY scripts/check-mtp-depths.py $B $MLXL3_QWEN_TEST_MODEL $MLXL3_MTP_TEST_HEAD --tune > $D/depths.log 2>&1 || fail "depths exit"
$PY - $P24/depths-flag1.log $D/depths.log <<'PY' | tee $D/depths-compare.log || fail "depths"
import json, sys
from pathlib import Path
def load(f):
    rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.startswith('{')]
    assert rows[-1] == {"all_depths_parity": True, "prefix_parity": True, "cancel_recovery": True, "tuner_verified": True}, rows[-1]
    done = [(r['request_id'], r['token_hash'], r['stats']['generated_tokens']) for r in rows if r.get('type') == 'complete']
    tune = [r for r in rows if r.get('type') == 'mtp_tune_complete']
    assert done and len(tune) == 1
    return done, tune[0]
a, ta = load(sys.argv[1]); b, tb = load(sys.argv[2])
assert a == b, 'completions differ from P24 flag 1'
assert [r['token_hashes'] for r in ta['rows']] == [r['token_hashes'] for r in tb['rows']]
print(len(b), 'completions identical to P24 flag 1; tuner hashes identical')
print('tune', [(r['depth'], round(r['decode_tps'], 2), r['accepted_tokens']) for r in tb['rows']], 'best', tb['best_depth'])
PY

# 5. DFlash2 bridge against P24 flag 1.
timeout 1200 $PY scripts/check_dflash2.py $B $MLXL3_QWEN_TEST_MODEL $DRAFT --skip-tune --timeout 1100 --output $D/dflash.json > $D/dflash.log 2>&1 || fail "dflash exit"
$PY - $P24/dflash-flag1.json $D/dflash.json <<'PY' | tee $D/dflash-compare.log || fail "dflash"
import json, sys
from pathlib import Path
def hashes(f):
    r = json.loads(Path(f).read_text())
    assert r['status'] == 'complete' and r['parity'] is True, (f, r.get('status'), r.get('error'))
    ev = [(e['request_id'], e['token_hash'], e['stats']['generated_tokens']) for e in r['events'] if e.get('type') == 'complete']
    assert ev and all(isinstance(h, str) and len(h) == 16 and n > 0 for _, h, n in ev)
    return ev
a = hashes(sys.argv[1]); b = hashes(sys.argv[2])
assert a == b
print(len(b), 'DFlash2 completions identical to P24 flag 1')
PY

# 6. Two real text turns through the Desktop protocol.
timeout 600 $PY scripts/smoke-release.py $B $MLXL3_QWEN_TEST_MODEL > $D/smoke.log 2>&1 || fail "smoke"
grep -c '"turn"' $D/smoke.log | tee $D/smoke-turns.txt
[[ $(grep -c '"turn"' $D/smoke.log) -eq 2 ]] || fail "smoke turns"
echo rel_status=$rc
exit $rc
