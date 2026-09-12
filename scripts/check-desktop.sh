#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
test_python="${MLXL3_TEST_PYTHON:-}"
if [[ -z "${test_python}" ]]; then
    test_python="$(command -v python3)"
fi
test_dir="$(mktemp -d -t mlxl3-check)"
trap 'rm -rf "${test_dir}"' EXIT

# CI normally selects the active Xcode toolchain. A local Command Line Tools
# install can opt into an installed SDK explicitly (for example 26.5).
swift_args=(--package-path apps/MLXL3Studio)
swiftc_args=(-swift-version 6 -parse-as-library -module-cache-path "${test_dir}/modules")
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
    swift_args+=(--sdk "${MLXL3_MACOS_SDK}")
    swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
fi
swift build "${swift_args[@]}"
binary_dir="$(swift build "${swift_args[@]}" --show-bin-path)"
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
swiftc_args+=(-I "${binary_dir}")
[[ -d "${binary_dir}/Modules" ]] && swiftc_args+=(-I "${binary_dir}/Modules")
if [[ -f "${binary_dir}/SwiftMath.o" ]]; then
    math_objects=("${binary_dir}/SwiftMath.o")
else
    math_objects=("${binary_dir}/SwiftMath.build"/*.swift.o)
fi

swiftc "${swiftc_args[@]}" \
    "${sources[@]}" "${math_objects[@]}" \
    tests/studio-hardening-check.swift -o "${test_dir}/check"
"${test_dir}/check" "$PWD/tests/fake-desktop-engine.py"

"${binary_dir}/MLXL3Studio" --check-chat-timeline
"${binary_dir}/MLXL3Studio" --check-mcp-preferences

# Exercise the real Swift bridge against a subprocess: callback ordering,
# stale-generation suppression, reload, cancellation and crash handling.
awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift \
    tests/bridge-dispatch-check.swift > "${test_dir}/BridgeDispatchCombined.swift"
bridge_sources=("${sources[@]}")
bridge_sources=("${(@)bridge_sources:#*/MLXL3Bridge.swift}")
swiftc "${swiftc_args[@]}" -O \
    "${bridge_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/BridgeDispatchCombined.swift" -o "${test_dir}/bridge-check"
"${test_dir}/bridge-check" "$PWD/tests/fake-desktop-engine.py"

# Exercise incremental Markdown/code rendering and Unicode chunk boundaries
# through AppKit/SwiftUI, not only the pure Python parser tests.
swiftc "${swiftc_args[@]}" -O \
    "${sources[@]}" "${math_objects[@]}" tests/streaming-render-check.swift \
    -o "${test_dir}/streaming-check"
"${test_dir}/streaming-check"

# Exercise the cancellable CLICommand transport independently of the model
# bridge (large pipe output, line progress, cancellation and stderr errors).
swiftc "${swiftc_args[@]}" -O apps/MLXL3Studio/Sources/MLXL3Studio/CLICommand.swift \
    tests/cli-command-check.swift -o "${test_dir}/cli-command-check"
"${test_dir}/cli-command-check" "${test_python}"

print "Desktop E2E checks passed: lifecycle, bridge, streaming/rendering and CLI transport"
