#!/bin/zsh
set -euo pipefail
run_dir="$PWD/build/desktop-preferences-20261102"
binary_dir="$PWD/apps/MLXL3Studio/.build/debug"
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
sources=("${(@)sources:#*/StudioModel.swift}")
sources=("${(@)sources:#*/DesktopPreferences.swift}")
math_objects=("${binary_dir}/SwiftMath.o")
cat apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift tests/desktop-preferences-check.swift > "${run_dir}/PreferencesCombined.swift"
args=(-swift-version 6 -parse-as-library -warnings-as-errors -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -module-cache-path "${run_dir}/modules" -I "${binary_dir}")
swiftc "${args[@]}" "${sources[@]}" "${math_objects[@]}" apps/MLXL3Studio/Sources/MLXL3Studio/DesktopPreferences.swift "${run_dir}/PreferencesCombined.swift" -o "${run_dir}/preferences-final"
MLXL3_SKIP_METAL_CHECK=1 MLXL3_PREFERENCE_SNAPSHOTS="${run_dir}" "${run_dir}/preferences-final" "$PWD/tests/fake-desktop-engine.py"
# Independent negative control: an incorrect default must be detected before any engine starts.
sed 's/else { return true }/else { return false }/' apps/MLXL3Studio/Sources/MLXL3Studio/DesktopPreferences.swift > "${run_dir}/BadPreferences.swift"
swiftc "${args[@]}" "${sources[@]}" "${math_objects[@]}" "${run_dir}/BadPreferences.swift" "${run_dir}/PreferencesCombined.swift" -o "${run_dir}/preferences-mutant"
ulimit -c 0
if "${run_dir}/preferences-mutant" "$PWD/tests/fake-desktop-engine.py" > "${run_dir}/mutation.log" 2>&1; then
    print -u2 'ERROR: incorrect default escaped the regression test'
    exit 1
fi
grep -q 'Keep existing auto-load default' "${run_dir}/mutation.log"
print 'Incorrect startup default detected; production source never mutated.'
