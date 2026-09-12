#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
package_dir="${repo_dir}/apps/MLXL3Studio"
app_dir="${repo_dir}/dist/MLXL3 Desktop.app"
legacy_app_dir="${repo_dir}/dist/MLXL3 Studio.app"
runtime_dist_dir="${repo_dir}/build/rust-runtime"
mlx_root="${MLXL3_MLX_ROOT:-}"
python_bin="${MLXL3_BUILD_PYTHON:-${repo_dir}/.venv/bin/python}"
cargo_bin="${MLXL3_CARGO:-${HOME}/.cargo/bin/cargo}"

if [[ -z "${MLXL3_MACOS_SDK:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
    export MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    print -u2 "Le bundle MLX doit être construit sur un Mac Apple Silicon."
    exit 2
fi
if [[ -z "${mlx_root}" ]]; then
    mlx_candidates=("${repo_dir}"/.venv/lib/python*/site-packages/mlx(N))
    mlx_root="${mlx_candidates[1]:-}"
fi
if [[ ! -f "${mlx_root}/lib/libmlx.dylib" || ! -f "${mlx_root}/lib/mlx.metallib" ]]; then
    print -u2 "MLX natif introuvable. Définis MLXL3_MLX_ROOT vers le paquet MLX 0.32.2."
    exit 2
fi
if [[ ! -x "${cargo_bin}" ]]; then
    cargo_bin="$(command -v cargo || true)"
fi
if [[ ! -x "${cargo_bin}" ]]; then
    print -u2 "Cargo introuvable. Installe Rust avec rustup."
    exit 2
fi

MLXL3_MLX_ROOT="${mlx_root}" MACOSX_DEPLOYMENT_TARGET=26.2 \
    "${cargo_bin}" build --release --locked --features mlx,chat --manifest-path "${repo_dir}/Cargo.toml"
rm -rf "${runtime_dist_dir}"
install -d "${runtime_dist_dir}"
install -m 755 "${repo_dir}/target/release/mlxl3-rs" "${runtime_dist_dir}/mlxl3"
install -m 755 "${mlx_root}/lib/libmlx.dylib" "${runtime_dist_dir}/libmlx.dylib"
install -m 755 "${mlx_root}/lib/libjaccl.dylib" "${runtime_dist_dir}/libjaccl.dylib"
install -m 644 "${mlx_root}/lib/mlx.metallib" "${runtime_dist_dir}/mlx.metallib"

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
