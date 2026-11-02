#!/bin/zsh
# Build only the Desktop changes; retain the installed engine that supports Flash Next.
set -euo pipefail
repo_dir="${0:A:h:h}"
app_dir="${repo_dir}/dist/MLXL3 Desktop Test.app"
launch=false
if [[ "${1:-}" == --launch && $# == 1 ]]; then
    launch=true
elif (( $# != 0 )); then
    print -u2 "Usage: $0 [--launch]"
    exit 2
fi
if ${launch} && /usr/bin/pgrep -x MLXL3Studio >/dev/null; then
    print -u2 "Ferme MLXL3 Desktop, puis relance cette commande."
    exit 1
fi
cd "${repo_dir}"
args=(--configuration release --package-path apps/MLXL3Studio)
sdk="${MLXL3_MACOS_SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
[[ -d "${sdk}" ]] && args+=(--sdk "${sdk}")
swift build "${args[@]}"
binary_dir="$(swift build "${args[@]}" --show-bin-path)"
mkdir -p build dist
stage="$(mktemp -d "${repo_dir}/build/desktop-test.XXXXXX")"
trap 'rm -rf "${stage}"' EXIT
bundle="${stage}/MLXL3 Desktop Test.app"
mkdir -p "${bundle}/Contents/MacOS" "${bundle}/Contents/Resources"
cp "${binary_dir}/MLXL3Studio" "${bundle}/Contents/MacOS/MLXL3Studio"
cp apps/MLXL3Studio/Resources/Info.plist "${bundle}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleDisplayName MLXL3 Desktop Test' "${bundle}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleName MLXL3 Desktop Test' "${bundle}/Contents/Info.plist"
# The helper checks the installed runtime manifest and every file hash before copying.
python3 scripts/copy-test-runtime.py "${bundle}"
ditto apps/MLXL3Studio/Vendor/SwiftMath/Sources/SwiftMath/mathFonts.bundle "${bundle}/Contents/Resources/mathFonts.bundle"
if [[ -f '/Applications/MLXL3 Desktop.app/Contents/Resources/AppIcon.icns' ]]; then
    cp '/Applications/MLXL3 Desktop.app/Contents/Resources/AppIcon.icns' "${bundle}/Contents/Resources/AppIcon.icns"
fi
codesign --force --deep --sign - --no-strict "${bundle}"
codesign --verify --deep --strict "${bundle}"
# Only replace this script's local test artifact, after the full stage is valid.
rm -rf "${app_dir}"
mv "${bundle}" "${app_dir}"
print "Version de test créée : ${app_dir}"
if ${launch}; then
    open -n "${app_dir}"
fi
