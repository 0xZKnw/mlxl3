#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
python_bin="${MLXL3_BUILD_PYTHON:-$PWD/.venv/bin/python}"
vendor="$PWD/apps/MLXL3Studio/Vendor/PonyExl3"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/benchmarks:$vendor${PYTHONPATH:+:$PYTHONPATH}"
"${python_bin}" -m PyInstaller --noconfirm --clean --onedir --name mlxl3-quantize \
    --distpath build/quantizer --workpath build/quantizer-work --specpath build \
    --paths "$PWD/src" --paths "$PWD/scripts" --paths "$PWD/benchmarks" --paths "$vendor" \
    --collect-submodules ponyexl3 --collect-data ponyexl3 \
    --collect-submodules mlx_lm.models --collect-all mlx --collect-data mlxl3_quantizer \
    --hidden-import quantize_lfm26 --hidden-import quantize_lfm25_moe --hidden-import quantize_ling \
    --hidden-import eval_lfm25_quality --hidden-import huggingface_hub \
    --add-data "$vendor/LICENSE:PonyExl3Notices" --add-data "$vendor/NOTICE:PonyExl3Notices" \
    scripts/quantization_studio.py
"${python_bin}" - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
paths=[root/'scripts/quantization_studio.py',root/'scripts/quantize_lfm26.py',root/'scripts/quantize_lfm25_moe.py',root/'scripts/quantize_ling.py',*(root/'src/mlxl3_quantizer').glob('*.py'), *(root/'apps/MLXL3Studio/Vendor/PonyExl3').rglob('*.py'), root/'scripts/patches/ponyexl3-mlxl3-conversion.patch']
metadata={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'ponyexl3':'8e7fa6b1556f59fc669e25087903b279b9b0346f','files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(root/'build/quantizer/mlxl3-quantize/build-info.json').write_text(json.dumps(metadata,indent=2)+'\n')
PY
