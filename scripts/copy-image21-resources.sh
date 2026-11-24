#!/bin/sh
# Both Desktop bundles run the same Python worker and Metal sources.
set -eu
repo_dir=$(dirname "$0")/..
destination=${1:?Expected the Image21 resource directory}
mkdir -p "$destination"
install -m 644 \
    "$repo_dir/apps/MLXL3Studio/Resources/Image21/render.py" \
    "$repo_dir/native/shaders/_qmm_tensor_kernel.metal" \
    "$repo_dir/native/shaders/_qmm_tensor_window_kernel.metal" \
    "$repo_dir/docs/images/studio-image21-m5.png" \
    "$destination/"
