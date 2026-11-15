#!/bin/zsh
# Q36-MTP-EVICT-T3B oracle 5: policy mutant (k = drafted width, bands ignored) must be detected; then restore.
set -u
setopt pipefail
W=/Users/justin/.codex/worktrees/q36-head-lane-p24/mix-stq1_0
T0=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-diag-t0-20261115
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3b-20261115
PY=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
EXPECTED=${1:?expected restored test binary sha256}
cd $W && source build/env.sh
unset MLXL3_MTP_POST_NORM MLXL3_MTP_EVICT MLXL3_MTP_DIAG_HIDDEN MLXL3_MTP_DIAG_COMPACT MLXL3_MTP_ADAPTIVE
mkdir -p $D/mutation
rc=0
F=native/src/mtp.rs
cp $F $D/mutation/mtp.rs.orig
$PY -I - $F <<'PY' || exit 2
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = """                let kept = crate::contracts::evict_width(&bands, proposals.len())
                    .context("invalid MTP EVICT width")?;"""
new = """                let kept = crate::contracts::evict_width(&bands, proposals.len())
                    .map(|_| proposals.len())
                    .context("invalid MTP EVICT width")?;"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
git diff --no-index $D/mutation/mtp.rs.orig $F > $D/mutation/mutant.diff
cargo test --release --features mlx,chat --lib --no-run > $D/mutation/build-mutant.log 2>&1 || rc=1
T=$(grep -o "build/target-p24/release/deps/mlxl3_native-[0-9a-f]*" $D/mutation/build-mutant.log | tail -1)
cp $T build/t3b/mutant && shasum -a 256 build/t3b/mutant > $D/mutation/mutant.sha256
cp $D/mutation/mtp.rs.orig $F
cmp -s $F $D/mutation/mtp.rs.orig || { echo "restore failed"; exit 2; }
# Mutant run: the diagnostic's own policy check must reject it (no report written).
OUT=$D/mutation/run; mkdir -p $OUT
{ MLXL3_MTP_EVICT=1 MLXL3_MTP_POST_NORM=1 MLXL3_MTP_DIAG_PROMPTS=$T0/prompts MLXL3_MTP_DIAG_OUTPUT=$OUT/diag.json \
  MLXL3_MTP_DIAG_TOKENS=256 MLXL3_MTP_DIAG_WARMUP=0 MLXL3_MTP_DIAG_CONFIDENCE=1 MLXL3_MTP_DIAG_DEPTHS=3 \
  build/t3b/mutant mtp::cycle_diagnostics::mtp_cycle_diagnostics --exact --ignored --nocapture --test-threads=1 > $OUT/test.log 2>&1 & }
pid=$!; ( sleep 1200; kill -9 $pid 2>/dev/null ) & dog=$!
wait $pid; code=$?; kill $dog 2>/dev/null; wait $dog 2>/dev/null
if [[ $code -ne 0 && ! -e $OUT/diag.json ]] && grep -q "recorded EVICT width is not the policy's" $OUT/test.log; then
  echo "mutant detected by the diagnostic"; else echo "mutant NOT detected ($code)"; rc=1; fi
# Independent Python oracle: the same mutation applied to a real EVICT report must be rejected.
$PY -I - $D/oracles/evict1-post1/diag.json $D/mutation/forged.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for c in r["cases"]:
    for e in c["trace"]["evictions"]:
        e["kept"] = e["drafted"]
json.dump(r, open(sys.argv[2], "w"))
PY
$PY -I $D/scripts/compare_t3b.py $D/mutation/forged.json /Users/justin/Documents/mix-stq1_0/docs/measurements/q36-mtp-evict-t3a-20261115/post/diag.json evict post > $D/mutation/forged-compare.log 2>&1
[[ $? -ne 0 ]] && grep -q "k mismatches" $D/mutation/forged-compare.log && echo "forged k detected by Python oracle" || { echo "forged k NOT detected"; rc=1; }
# Restore: rebuild and require the original test binary.
cargo test --release --features mlx,chat --lib --no-run > $D/mutation/build-restored.log 2>&1 || rc=1
T=$(grep -o "build/target-p24/release/deps/mlxl3_native-[0-9a-f]*" $D/mutation/build-restored.log | tail -1)
shasum -a 256 $T | tee $D/mutation/restored.sha256
[[ "$(shasum -a 256 $T | cut -d' ' -f1)" == "$EXPECTED" ]] && echo "restored binary identical" || { echo "restored binary differs"; rc=1; }
echo mutation_status=$rc
exit $rc
