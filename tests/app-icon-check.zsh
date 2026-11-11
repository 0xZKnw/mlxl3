#!/bin/zsh
set -euo pipefail
repo_dir="${0:A:h:h}"
source_image="${repo_dir}/apps/MLXL3Studio/Resources/AppIcon.png"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
expected_sha=e9bae3204b80cae5959d72dfd3bc6c791774b297c7f485e4e635e225d1bcddb1
[[ "$(shasum -a 256 "${source_image}" | cut -d ' ' -f 1)" == "${expected_sha}" ]]
"${repo_dir}/scripts/build-app-icon.sh" "${source_image}" "${work_dir}/AppIcon.icns"
iconutil -c iconset "${work_dir}/AppIcon.icns" -o "${work_dir}/AppIcon.iconset"
sips -z 1024 1024 "${source_image}" --out "${work_dir}/expected.png" >/dev/null
sips -s format bmp "${work_dir}/expected.png" --out "${work_dir}/expected.bmp" >/dev/null
sips -s format bmp "${work_dir}/AppIcon.iconset/icon_512x512@2x.png" --out "${work_dir}/actual.bmp" >/dev/null
cmp "${work_dir}/expected.bmp" "${work_dir}/actual.bmp"
if "${repo_dir}/scripts/build-app-icon.sh" "${work_dir}/missing.png" "${work_dir}/invalid.icns" >/dev/null 2>&1; then
    print -u2 'Missing icon source was accepted'
    exit 1
fi
[[ ! -e "${work_dir}/invalid.icns" ]]
print 'App icon checks passed: supplied artwork, generated ICNS, missing source'
