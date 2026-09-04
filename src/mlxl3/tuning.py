"""Opt-in, version/device-bound offline TensorOps tuning profiles.

No timing, file polling, or learning occurs on the decode hot path. Shape
changes are Q1 candidates, never silently enabled as bit-exact optimizations.
"""
import json
import os
import platform
from functools import cache

import mlx.core as mx

# Bump whenever TensorOps addressing, reductions or codebook arithmetic changes.
# A profile is not portable merely because the device and MLX match.
TENSOR_KERNEL_ABI = 1


def signature():
    return {'architecture': mx.device_info().get('architecture'),
            'device': mx.device_info().get('device_name'),
            'mlx': mx.__version__, 'os': platform.mac_ver()[0],
            'tensor_kernel_abi': TENSOR_KERNEL_ABI}


def shape_key(rows, input_dims, output_dims, bits, mode):
    return ':'.join(map(str, (rows, input_dims, output_dims, bits, int(mode))))


def validate_tile(tile, input_dims, output_dims):
    if not isinstance(tile, (list, tuple)) or len(tile) != 3:
        return False
    bm, bn, bk = tile
    return (bm in (8, 16, 32) and bn in (16, 32, 64) and bk in (16, 32, 64)
            and input_dims % bk == 0 and output_dims % bn == 0)


@cache
def _profile():
    path = os.environ.get('MLXL3_TENSOR_PROFILE')
    if not path or os.environ.get('MLXL3_ALLOW_Q1_TUNING', '0') != '1':
        return {}
    with open(path) as handle:
        profile = json.load(handle)
    if profile.get('schema') != 1 or profile.get('signature') != signature():
        raise ValueError('TensorOps tuning profile is incompatible with this runtime/device')
    return profile.get('entries', {})


def tensor_tile(rows, input_dims, output_dims, bits, mode):
    entry = _profile().get(shape_key(rows, input_dims, output_dims, bits, mode))
    if entry is None:
        return 32, 32, 16
    tile = entry.get('tile')
    if not entry.get('heldout_checked') or not validate_tile(tile, input_dims, output_dims):
        raise ValueError('invalid/unvalidated TensorOps tuning profile entry')
    return tuple(tile)
