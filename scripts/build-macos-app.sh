#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
package_dir="${repo_dir}/apps/MLXL3Studio"
app_dir="${repo_dir}/dist/MLXL3 Desktop.app"
legacy_app_dir="${repo_dir}/dist/MLXL3 Studio.app"
runtime_dist_dir="${repo_dir}/build/pyinstaller-dist/runtime"
runtime_work_dir="${repo_dir}/build/pyinstaller-work"
python_bin="${MLXL3_BUNDLE_PYTHON:-${repo_dir}/.venv/bin/python}"

if [[ "$(uname -m)" != "arm64" ]]; then
    print -u2 "Le bundle MLX doit être construit sur un Mac Apple Silicon."
    exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
    print -u2 "Python de build introuvable : ${python_bin}"
    exit 2
fi

"${python_bin}" -c 'import PyInstaller, mlx, mlx_lm, mlxl3' 2>/dev/null || {
    print -u2 'Dépendances de bundle absentes. Lance : pip install -e ".[bundle]"'
    exit 2
}

"${python_bin}" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "${repo_dir}/build/pyinstaller-dist" \
    --workpath "${runtime_work_dir}" \
    "${repo_dir}/packaging/mlxl3-runtime.spec"

if [[ ! -x "${runtime_dist_dir}/mlxl3" ]]; then
    print -u2 "Le runtime autonome n’a pas été produit."
    exit 2
fi

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
ditto "${runtime_dist_dir}" "${app_dir}/Contents/Resources/runtime"

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

# Keep LaTeX fonts in the app's standard signed Resources directory. SwiftMath
# is vendored only to make this release layout independent of SwiftPM's build
# directory conventions.
math_bundle="${package_dir}/Vendor/SwiftMath/Sources/SwiftMath/mathFonts.bundle"
ditto "${math_bundle}" "${app_dir}/Contents/Resources/mathFonts.bundle"

codesign --force --deep --sign - --no-strict "${app_dir}"
codesign --verify --deep --strict "${app_dir}"

"${app_dir}/Contents/Resources/runtime/mlxl3" list --json >/dev/null

print "Application créée : ${app_dir}"
