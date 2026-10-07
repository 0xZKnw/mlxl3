#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h:h:h}"
out="$PWD/docs/measurements/pr28-review"
build_dir="$(mktemp -d -t mlxl3-pr28-build)"
trap 'rm -rf "$build_dir"' EXIT
sdk=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
binary_dir="$(swift build --package-path apps/MLXL3Studio --sdk "$sdk" --show-bin-path)"
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
sources=("${(@)sources:#*/StudioModel.swift}")
if [[ -f "${binary_dir}/SwiftMath.o" ]]; then
    math_objects=("${binary_dir}/SwiftMath.o")
else
    math_objects=("${binary_dir}/SwiftMath.build"/*.swift.o)
fi
awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    "$out/review-idle-check.swift" > "$build_dir/ReviewCombined.swift"
swiftc -swift-version 6 -strict-concurrency=complete -warnings-as-errors -parse-as-library \
    -sdk "$sdk" -module-cache-path "$build_dir/modules" -I "$binary_dir" -I "$binary_dir/Modules" \
    "${sources[@]}" "${math_objects[@]}" "$build_dir/ReviewCombined.swift" -o "$build_dir/review-check"
"$build_dir/review-check" "$out/fault-engine.py" "$out/review-repros.json" valid error malformed
