#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
test_dir="$(mktemp -d -t mlxl3-workspace-check)"
trap 'rm -rf "${test_dir}"' EXIT
swift_args=(--package-path apps/MLXL3Studio)
swiftc_args=(-swift-version 6 -warnings-as-errors -parse-as-library)
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
  swift_args+=(--sdk "${MLXL3_MACOS_SDK}")
  swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
fi
swift build "${swift_args[@]}"
bin="$(swift build "${swift_args[@]}" --show-bin-path)"
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
if [[ -f "${bin}/SwiftMath.o" ]]; then math_objects=("${bin}/SwiftMath.o"); else math_objects=("${bin}/SwiftMath.build"/*.swift.o); fi
swiftc "${swiftc_args[@]}" -I "$bin" -I "$bin/Modules" "${sources[@]}" "${math_objects[@]}" tests/studio-workspace-check.swift -o "${test_dir}/workspace-check"
if [[ "${MLXL3_STUDIO_INTERACTIVE:-}" == "1" ]]; then
  app_dir="$PWD/build/MLXL3 Workspace Check.app"
  mkdir -p "$app_dir/Contents/MacOS"
  cp "${test_dir}/workspace-check" "$app_dir/Contents/MacOS/workspace-check"
  cat > "$app_dir/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>io.mlxl3.workspace-check</string><key>CFBundleName</key><string>MLXL3 Workspace Check</string><key>CFBundleExecutable</key><string>workspace-check</string><key>CFBundlePackageType</key><string>APPL</string><key>CFBundleVersion</key><string>1</string><key>CFBundleShortVersionString</key><string>1.0</string><key>NSPrincipalClass</key><string>NSApplication</string></dict></plist>
PLIST
  "$app_dir/Contents/MacOS/workspace-check"
else
  "${test_dir}/workspace-check"
fi
