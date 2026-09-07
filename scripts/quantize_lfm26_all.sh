#!/bin/bash
# Sequential local conversions; rerunning resumes the converter's artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/lfm26-quants
for lfm_bits in 4.1 5 3 8; do
    lfm_output="models/LFM2.5-2.6B-EXL3-${lfm_bits}bpw"
    lfm_work="build/lfm26-quants/${lfm_bits}bpw"
    mkdir -p "$lfm_work"
    echo "Starting LFM2.5-2.6B ${lfm_bits} bpw: $(date -u)"
    # Same source, corpus and capture settings across all four variants.
    if [[ "$lfm_bits" != 4.1 && ! -f "$lfm_work/calibration.safetensors" ]]; then
        cp -c build/lfm26-quants/4.1bpw/calibration.safetensors "$lfm_work/calibration.safetensors"
    fi
    .venv/bin/python scripts/quantize_lfm26.py \
        --in-dir models/source/LFM2.5-2.6B --out-dir "$lfm_output" \
        --work-dir "$lfm_work" --bits "$lfm_bits" \
        --calibration-rows 2048 --calibration-seq-len 1024 \
        --max-workers 1 --search-backend metal \
        > "$lfm_work/conversion.log" 2>&1
    .venv/bin/python benchmarks/eval_winogrande.py mlxl3 "$lfm_output" \
        --limit 64 --output "$lfm_work/winogrande64.json" \
        > "$lfm_work/evaluation.log" 2>&1
    .venv/bin/mlxl3 run "$lfm_output" 'Salut ! Combien font deux plus deux ?' \
        --temperature 0 --max-tokens 256 > "$lfm_work/chat.log" 2>&1
    echo "Completed LFM2.5-2.6B ${lfm_bits} bpw: $(date -u)"
done
