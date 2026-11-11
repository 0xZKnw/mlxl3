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
if [[ "${backend}" == metal || "${backend}" == hybrid ]]; then
    print -u2 "Les moteurs metal et hybrid sont désactivés pour la v2. Utilise MLXL3_ENGINE_BACKEND=mlx."
    exit 2
fi
engine_binary="${CARGO_TARGET_DIR:-${repo_dir}/target}/release/mlxl3-rs"
package_backend="${backend}"
case "${backend}" in
    metal)
        MACOSX_DEPLOYMENT_TARGET=26.2 "${cargo_bin}" build --release --locked --no-default-features --features direct-metal,chat --manifest-path "${repo_dir}/Cargo.toml"
        ;;
    mlx|hybrid)
        if [[ -z "${mlx_root}" ]]; then
            mlx_candidates=("${repo_dir}"/.venv/lib/python*/site-packages/mlx(N))
            mlx_root="${mlx_candidates[1]:-}"
        fi
        if [[ ! -f "${mlx_root}/lib/libmlx.dylib" || ! -f "${mlx_root}/lib/mlx.metallib" ]]; then
            print -u2 "MLX natif introuvable. Définis MLXL3_MLX_ROOT vers le paquet MLX 0.32.2."
            exit 2
        fi
        MLXL3_MLX_ROOT="${mlx_root}" MACOSX_DEPLOYMENT_TARGET=26.2 \
            "${cargo_bin}" build --release --locked --no-default-features --features mlx,chat --bin mlxl3-rs --manifest-path "${repo_dir}/Cargo.toml"
        if [[ "${backend}" == hybrid ]]; then
            helper_dir="${repo_dir}/build/hybrid-helpers"
            install -d "${helper_dir}"
            install -m 755 "${engine_binary}" "${helper_dir}/mlx"
            MACOSX_DEPLOYMENT_TARGET=26.2 "${cargo_bin}" build --release --locked --no-default-features --features direct-metal,chat --bin mlxl3-rs --manifest-path "${repo_dir}/Cargo.toml"
            install -m 755 "${engine_binary}" "${helper_dir}/metal"
            for helper in mlx metal; do
                codesign --force --sign - "${helper_dir}/${helper}"
                lipo "${helper_dir}/${helper}" -verify_arch arm64
                codesign --verify --strict "${helper_dir}/${helper}"
                otool -L "${helper_dir}/${helper}" > "${helper_dir}/${helper}-dependencies.txt"
                DYLD_LIBRARY_PATH="${mlx_root}/lib" DYLD_PRINT_LIBRARIES=1 "${helper_dir}/${helper}" runtime-info > "${helper_dir}/${helper}-runtime.json" 2> "${helper_dir}/${helper}-dyld.txt"
            done
            "${python_bin}" - "${helper_dir}" <<'CHECK'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
infos={name:json.loads((root/f'{name}-runtime.json').read_text()) for name in ('mlx','metal')}
assert infos['mlx']['version']==infos['metal']['version']
assert infos['mlx']['commit']==infos['metal']['commit']
for name,info in infos.items():
    assert info['backend']==name and info['mlx_enabled']==(name=='mlx')
    assert info['chat_enabled'] and info['architecture']=='aarch64'
    linked=(root/f'{name}-dependencies.txt').read_text().lower()
    loaded=(root/f'{name}-dyld.txt').read_text().lower()
    assert loaded.strip()
    if name=='metal':
        assert 'metal.framework' in linked
        assert not any(lib in linked+loaded for lib in ('libmlx.','libjaccl.','mlx.metallib'))
    else:
        assert 'libmlx.' in linked and 'libmlx.' in loaded
CHECK
            MLXL3_MLX_HELPER="${helper_dir}/mlx" MLXL3_METAL_HELPER="${helper_dir}/metal" MACOSX_DEPLOYMENT_TARGET=26.2 \
                "${cargo_bin}" build --release --locked --no-default-features --features hybrid-dispatch --bin mlxl3-hybrid --manifest-path "${repo_dir}/Cargo.toml"
            engine_binary="${CARGO_TARGET_DIR:-${repo_dir}/target}/release/mlxl3-hybrid"
            package_backend=mlx
        fi
        ;;
    *) print -u2 "MLXL3_ENGINE_BACKEND doit être metal, mlx ou hybrid."; exit 2 ;;
esac
rm -rf "${runtime_dist_dir}"
install -d "${runtime_dist_dir}"
install -m 755 "${engine_binary}" "${runtime_dist_dir}/mlxl3"
components=(mlxl3)
if [[ "${package_backend}" == mlx ]]; then
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
identity_args=()
if [[ -n "${MLXL3_ENGINE_RELEASE_TAG:-}" ]]; then
    identity_args+=(--release-tag "${MLXL3_ENGINE_RELEASE_TAG}")
fi
"${python_bin}" "${script_dir}/package_engine.py" "${runtime_dist_dir}" "${repo_dir}/dist" --backend "${package_backend}" "${identity_args[@]}"
