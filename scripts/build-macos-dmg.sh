#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
app_dir="${repo_dir}/dist/MLXL3 Desktop.app"
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
    "${repo_dir}/apps/MLXL3Studio/Resources/Info.plist")"
dmg_path="${repo_dir}/dist/MLXL3-Desktop-v${version}-Apple-Silicon.dmg"

"${script_dir}/build-macos-app.sh"

staging_dir="$(mktemp -d)"
trap 'rm -rf "${staging_dir}"' EXIT
ditto "${app_dir}" "${staging_dir}/MLXL3 Desktop.app"
ln -s /Applications "${staging_dir}/Applications"

if [[ -e "${dmg_path}" ]]; then
    rm "${dmg_path}"
fi
hdiutil create \
    -volname "MLXL3 Desktop" \
    -srcfolder "${staging_dir}" \
    -format UDZO \
    -imagekey zlib-level=9 \
    "${dmg_path}"

print "DMG autonome créé : ${dmg_path}"
