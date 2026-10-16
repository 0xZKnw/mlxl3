#!/bin/zsh
set -euo pipefail
# Release-only assembly using the exact already-published engine archive.
repo_dir="${0:A:h:h:h:h}"
script_dir="${repo_dir}/scripts"
package_dir="${repo_dir}/apps/MLXL3Studio"
app_dir="${repo_dir}/dist/MLXL3 Desktop.app"
legacy_app_dir="${repo_dir}/dist/MLXL3 Studio.app"
runtime_dist_dir="${repo_dir}/build/published-engine-1.4.8"
python_bin="${MLXL3_BUILD_PYTHON:-python3}"
version="1.4.7"
build="29"
dmg_path="${repo_dir}/dist/MLXL3-Desktop-v${version}-b${build}-Apple-Silicon.dmg"
install -d "${repo_dir}/dist"
cd "${repo_dir}"
"${python_bin}" - <<'VERIFY'
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
assert not subprocess.check_output(['git','diff','HEAD','--name-only']), 'Tracked sources must be committed before packaging'
runtime=root/'build/published-engine-1.4.8'
manifest=runtime/'engine.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='cdd4421ba021e3ab5263a7b4c148337656cc9136978186e57d44a21e873b176f'
for name,record in json.loads(manifest.read_text())['files'].items():
 file=runtime/name
 assert file.stat().st_size==record['size'] and hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'],name
VERIFY
swift_args=(--configuration release --package-path "${package_dir}")
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
    swift_args+=(--sdk "${MLXL3_MACOS_SDK}")
fi
swift build "${swift_args[@]}"
binary_dir="$(swift build "${swift_args[@]}" --show-bin-path)"

if [[ "${app_dir}" != "${repo_dir}/dist/MLXL3 Desktop.app" ]]; then
    print -u2 "Refus de remplacer un chemin d’application inattendu."
    exit 2
fi

rm -rf "${app_dir}" "${legacy_app_dir}"
install -d "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
install -m 755 "${binary_dir}/MLXL3Studio" "${app_dir}/Contents/MacOS/MLXL3Studio"
install -m 644 "${package_dir}/Resources/Info.plist" "${app_dir}/Contents/Info.plist"
ditto "${runtime_dist_dir}" "${app_dir}/Contents/Resources/runtime"
"${python_bin}" "${script_dir}/build_manifest.py" "${app_dir}/Contents/Resources/build-info.json"

icon_work_dir="$(mktemp -d)"
trap 'rm -rf "${icon_work_dir}"' EXIT
iconset_dir="${icon_work_dir}/AppIcon.iconset"
install -d "${iconset_dir}"
sips -z 1024 1024 "${package_dir}/Resources/AppIcon.png" --out "${icon_work_dir}/icon-1024.png" >/dev/null
for size in 16 32 128 256 512; do
    sips -z "${size}" "${size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null
    double_size=$((size * 2))
    sips -z "${double_size}" "${double_size}" "${icon_work_dir}/icon-1024.png" --out "${iconset_dir}/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "${iconset_dir}" -o "${app_dir}/Contents/Resources/AppIcon.icns"

# Keep LaTeX fonts in the app's standard signed Resources directory. SwiftMath
# is vendored only to make this release layout independent of SwiftPM's build
# directory conventions.
math_bundle="${package_dir}/Vendor/SwiftMath/Sources/SwiftMath/mathFonts.bundle"
ditto "${math_bundle}" "${app_dir}/Contents/Resources/mathFonts.bundle"

codesign --force --sign - --no-strict "${app_dir}"
codesign --verify --deep --strict "${app_dir}"

"${app_dir}/Contents/Resources/runtime/mlxl3" list --json >/dev/null

print "Application créée : ${app_dir}"

"${python_bin}" - <<'VERIFY'
from pathlib import Path
import hashlib,json
runtime=Path('dist/MLXL3 Desktop.app/Contents/Resources/runtime')
manifest=runtime/'engine.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='cdd4421ba021e3ab5263a7b4c148337656cc9136978186e57d44a21e873b176f'
for name,record in json.loads(manifest.read_text())['files'].items():
 file=runtime/name
 assert file.stat().st_size==record['size'] and hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'],name
print('Published engine and manifest remained byte-identical after app signing.')
VERIFY
rm -rf "${icon_work_dir}"
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
