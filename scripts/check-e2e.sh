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
"${python_bin}" -m compileall -q src
"${python_bin}" -m pytest -q
export MLXL3_TEST_PYTHON="${python_bin}"
scripts/check-desktop.sh
print "MLXL3 full E2E suite passed"
