#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_dir="${script_dir:h}"
runtime_dist_dir="${repo_dir}/build/rust-runtime"
mlx_root="${MLXL3_MLX_ROOT:-}"
python_bin="${MLXL3_BUILD_PYTHON:-${repo_dir}/.venv/bin/python}"
cargo_bin="${MLXL3_CARGO:-${HOME}/.cargo/bin/cargo}"

if [[ -z "${MLXL3_MACOS_SDK:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
    export MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    print -u2 "Le moteur doit être construit sur un Mac Apple Silicon."
    exit 2
fi
if [[ ! -x "${cargo_bin}" ]]; then
    cargo_bin="$(command -v cargo || true)"
fi
if [[ ! -x "${cargo_bin}" ]]; then
    print -u2 "Cargo introuvable. Installe Rust avec rustup."
    exit 2
fi

backend="${MLXL3_ENGINE_BACKEND:-mlx}"
case "${backend}" in
    metal)
        MACOSX_DEPLOYMENT_TARGET=26.2 "${cargo_bin}" build --release --locked --features chat --manifest-path "${repo_dir}/Cargo.toml"
        ;;
    mlx)
        if [[ -z "${mlx_root}" ]]; then
            mlx_candidates=("${repo_dir}"/.venv/lib/python*/site-packages/mlx(N))
            mlx_root="${mlx_candidates[1]:-}"
        fi
        if [[ ! -f "${mlx_root}/lib/libmlx.dylib" || ! -f "${mlx_root}/lib/mlx.metallib" ]]; then
            print -u2 "MLX natif introuvable. Définis MLXL3_MLX_ROOT vers le paquet MLX 0.32.2."
            exit 2
        fi
        MLXL3_MLX_ROOT="${mlx_root}" MACOSX_DEPLOYMENT_TARGET=26.2 \
            "${cargo_bin}" build --release --locked --features mlx,chat --manifest-path "${repo_dir}/Cargo.toml"
        ;;
    *) print -u2 "MLXL3_ENGINE_BACKEND doit être metal ou mlx."; exit 2 ;;
esac
rm -rf "${runtime_dist_dir}"
install -d "${runtime_dist_dir}"
install -m 755 "${CARGO_TARGET_DIR:-${repo_dir}/target}/release/mlxl3-rs" "${runtime_dist_dir}/mlxl3"
components=(mlxl3)
if [[ "${backend}" == mlx ]]; then
    install -m 755 "${mlx_root}/lib/libmlx.dylib" "${runtime_dist_dir}/libmlx.dylib"
    install -m 755 "${mlx_root}/lib/libjaccl.dylib" "${runtime_dist_dir}/libjaccl.dylib"
    install -m 644 "${mlx_root}/lib/mlx.metallib" "${runtime_dist_dir}/mlx.metallib"
    components+=(libmlx.dylib libjaccl.dylib)
fi
for component in "${components[@]}"; do
    codesign --force --sign - "${runtime_dist_dir}/${component}"
    lipo "${runtime_dist_dir}/${component}" -verify_arch arm64
done
if [[ "${backend}" == metal ]]; then
    otool -L "${runtime_dist_dir}/mlxl3" > "${runtime_dist_dir}/../native-dependencies.txt"
    DYLD_PRINT_LIBRARIES=1 "${runtime_dist_dir}/mlxl3" runtime-info > "${runtime_dist_dir}/../native-runtime.json" 2> "${runtime_dist_dir}/../native-dyld.txt"
    "${python_bin}" - "${runtime_dist_dir}/.." <<'CHECK'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
info=json.loads((root/'native-runtime.json').read_text())
assert info['backend']=='metal' and info['mlx_enabled'] is False and info['chat_enabled'] is True
assert info['architecture']=='aarch64'
linked=(root/'native-dependencies.txt').read_text().lower()
loaded=(root/'native-dyld.txt').read_text().lower()
assert 'metal.framework' in linked
assert loaded.strip(), 'dynamic loader did not emit its dependency evidence'
assert not any(name in linked+loaded for name in ('libmlx.', 'libjaccl.', 'mlx.metallib'))
CHECK
fi
"${python_bin}" "${script_dir}/package_engine.py" "${runtime_dist_dir}" "${repo_dir}/dist" --backend "${backend}"
