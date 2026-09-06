#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
swift build --package-path apps/MLXL3Studio
test_dir="$(mktemp -d -t mlxl3-check)"
trap 'rm -rf "${test_dir}"' EXIT
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
swiftc -swift-version 6 -parse-as-library \
    -I apps/MLXL3Studio/.build/arm64-apple-macosx/debug/Modules \
    "${sources[@]}" apps/MLXL3Studio/.build/arm64-apple-macosx/debug/SwiftMath.build/*.swift.o \
    tests/studio-hardening-check.swift -o "${test_dir}/check"
"${test_dir}/check" "$PWD/tests/fake-desktop-engine.py"
apps/MLXL3Studio/.build/debug/MLXL3Studio --check-chat-timeline
apps/MLXL3Studio/.build/debug/MLXL3Studio --check-mcp-preferences
