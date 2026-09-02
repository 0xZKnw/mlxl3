#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
package_dir="${repo_dir}/apps/MLXL3Studio"
app_dir="${repo_dir}/dist/MLXL3 Desktop.app"
legacy_app_dir="${repo_dir}/dist/MLXL3 Studio.app"

swift build --configuration release --package-path "${package_dir}"
binary_dir="$(swift build --configuration release --package-path "${package_dir}" --show-bin-path)"

if [[ "${app_dir}" != "${repo_dir}/dist/MLXL3 Desktop.app" ]]; then
    print -u2 "Refus de remplacer un chemin d’application inattendu."
    exit 2
fi

rm -rf "${app_dir}" "${legacy_app_dir}"
install -d "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
install -m 755 "${binary_dir}/MLXL3Studio" "${app_dir}/Contents/MacOS/MLXL3Studio"
install -m 644 "${package_dir}/Resources/Info.plist" "${app_dir}/Contents/Info.plist"

icon_work_dir="$(mktemp -d)"
trap 'rm -rf "${icon_work_dir}"' EXIT
iconset_dir="${icon_work_dir}/AppIcon.iconset"
install -d "${iconset_dir}"
sips -s format png "${package_dir}/Resources/AppIcon.svg" --out "${icon_work_dir}/icon-1024.png" >/dev/null
for size in 16 32 128 256 512; do
    sips -z "${size}" "${size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null
    double_size=$((size * 2))
    sips -z "${double_size}" "${double_size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "${iconset_dir}" -o "${app_dir}/Contents/Resources/AppIcon.icns"

# SwiftPM's Bundle.module accessor looks beside Bundle.main when a package is
# embedded in a standalone executable app. Preserve that layout for SwiftMath's
# local fonts; there is no runtime download. Copy the explicit bundle so stale
# build artifacts from removed dependencies never enter the application.
math_bundle="${binary_dir}/SwiftMath_SwiftMath.bundle"
if [[ -d "${math_bundle}" ]]; then
    ditto "${math_bundle}" "${app_dir}/${math_bundle:t}"
fi

codesign --force --deep --sign - --no-strict "${app_dir}"

print "Application créée : ${app_dir}"
