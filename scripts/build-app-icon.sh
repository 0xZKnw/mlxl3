#!/bin/zsh
set -euo pipefail
if (( $# != 2 )); then
    print -u2 "Usage: $0 source.png output.icns"
    exit 2
fi
icon_work_dir="$(mktemp -d)"
trap 'rm -rf "${icon_work_dir}"' EXIT
iconset_dir="${icon_work_dir}/AppIcon.iconset"
install -d "${iconset_dir}"
width="$(sips -g pixelWidth "$1" | awk '/pixelWidth/ { print $2 }')"
height="$(sips -g pixelHeight "$1" | awk '/pixelHeight/ { print $2 }')"
[[ "${width}" == "${height}" && "${width}" -ge 32 ]] || { print -u2 'Icon source must be square and at least 32 pixels'; exit 1; }
sips -z 1024 1024 "$1" --out "${icon_work_dir}/icon-1024.png" >/dev/null
for size in 16 32 128 256 512; do
    sips -z "${size}" "${size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null
    double_size=$((size * 2))
    sips -z "${double_size}" "${double_size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "${iconset_dir}" -o "$2"
