#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
package_dir="${repo_dir}/apps/MLXL3Studio"
app_dir="${repo_dir}/dist/MLXL3 Studio.app"

swift build --configuration release --package-path "${package_dir}"
binary_dir="$(swift build --configuration release --package-path "${package_dir}" --show-bin-path)"

if [[ "${app_dir}" != "${repo_dir}/dist/MLXL3 Studio.app" ]]; then
    print -u2 "Refus de remplacer un chemin d’application inattendu."
    exit 2
fi

rm -rf "${app_dir}"
install -d "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
install -m 755 "${binary_dir}/MLXL3Studio" "${app_dir}/Contents/MacOS/MLXL3Studio"
install -m 644 "${package_dir}/Resources/Info.plist" "${app_dir}/Contents/Info.plist"
codesign --force --deep --sign - "${app_dir}"

print "Application créée : ${app_dir}"
