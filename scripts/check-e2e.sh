#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"

python_bin="${MLXL3_TEST_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        python_bin="$PWD/.venv/bin/python"
    else
        python_bin="$(command -v python3)"
    fi
fi
export PYTHONPATH="$PWD/src:$PWD/apps/MLXL3Studio/Vendor/PonyExl3${PYTHONPATH:+:$PYTHONPATH}"
"${python_bin}" -c 'from pathlib import Path; import mlxl3_quantizer; assert Path(mlxl3_quantizer.__file__).resolve().is_relative_to((Path.cwd() / "src").resolve()), "Tests resolved a different checkout"'
"${python_bin}" -m compileall -q src
"${python_bin}" -m pytest -q
export MLXL3_TEST_PYTHON="${python_bin}"
scripts/check-desktop.sh
print "MLXL3 full E2E suite passed"
