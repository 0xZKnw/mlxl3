#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
test_dir="$(mktemp -d -t mlxl3-studios)"
trap 'rm -rf "${test_dir}"' EXIT
test_python="${MLXL3_TEST_PYTHON:-$(command -v python3)}"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/benchmarks:$PWD/apps/MLXL3Studio/Vendor/PonyExl3${PYTHONPATH:+:$PYTHONPATH}"
"${test_python}" -m pytest -q tests/test_quantization_studio.py tests/test_lfm_conversion_adapter.py
swift_args=(--package-path apps/MLXL3Studio)
swiftc_args=(-swift-version 6 -warnings-as-errors -parse-as-library -module-cache-path "${test_dir}/modules")
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
    swift_args+=(--sdk "${MLXL3_MACOS_SDK}")
    swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
fi
swift build "${swift_args[@]}"
binary_dir="$(swift build "${swift_args[@]}" --show-bin-path)"
swiftc_args+=(-I "${binary_dir}" -I "${binary_dir}/Modules")
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
if [[ -f "${binary_dir}/SwiftMath.o" ]]; then
    math_objects=("${binary_dir}/SwiftMath.o")
else
    math_objects=("${binary_dir}/SwiftMath.build"/*.swift.o)
fi
swiftc "${swiftc_args[@]}" "${sources[@]}" "${math_objects[@]}" tests/studios-check.swift -o "${test_dir}/studios-check"
MLXL3_STUDIOS_TMP="${test_dir}" "${test_dir}/studios-check"
swiftc "${swiftc_args[@]}" apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift tests/agent-sandbox-check.swift -o "${test_dir}/agent-check"
"${test_dir}/agent-check"
"${test_python}" tests/check-agent-protocol.py "${test_dir}/agent-check"
