"""Owned FP32 Hadamard boundaries for conversion weights and calibration rows."""

import math

import numpy as np

HAD_DIM = 128
HAD_SCALE = np.float32(1.0 / math.sqrt(HAD_DIM))


def _flush_subnormals(values):
    # Metal FP32 arithmetic flushes subnormals and retains the zero sign.
    words = values.view(np.uint32)
    small = (words & np.uint32(0x7F800000)) == 0
    words[small] &= np.uint32(0x80000000)


def hadamard128(values: np.ndarray, *, axis: int = 1) -> np.ndarray:
    """Apply normalized 128-point butterflies to a finite FP32 matrix axis.

    raises: ValueError
    post: __return__.shape == values.shape
    post: np.isfinite(__return__).all()
    """
    source = np.asarray(values, dtype=np.float32)
    if (
        source.ndim != 2
        or axis not in (0, 1)
        or any(size == 0 for size in source.shape)
        or source.shape[axis] % HAD_DIM
        or not np.isfinite(source).all()
    ):
        raise ValueError("Hadamard requires a finite nonempty matrix with a 128-aligned axis")
    aligned = np.ascontiguousarray(source.T if axis == 0 else source)
    blocks = aligned.reshape(-1, HAD_DIM).copy()
    _flush_subnormals(blocks)
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        pairs = blocks.reshape(-1, HAD_DIM // (2 * stride), 2, stride)
        left, right = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0] = left + right
        pairs[:, :, 1] = left - right
        _flush_subnormals(blocks)
    result = (blocks * HAD_SCALE).reshape(aligned.shape)
    _flush_subnormals(result)
    if not np.isfinite(result).all():
        raise ValueError("Hadamard output overflowed")
    return result.T if axis == 0 else result


def public_matrix_to_inner_native(values, *, suh=None, svh=None):
    """Apply the conversion oracle's divide/column-H128/divide/row-H128 order."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2 or any(size == 0 for size in result.shape) or not np.isfinite(result).all():
        raise ValueError("expected a finite nonempty public weight matrix")
    for scale, axis in [(svh, 1), (suh, 0)]:
        if scale is None:
            continue
        scale = np.asarray(scale, dtype=np.float32)
        if (
            scale.shape != (result.shape[axis],)
            or not np.isfinite(scale).all()
            or (scale == 0).any()
        ):
            raise ValueError("invalid conversion scale")
        _flush_subnormals(result)
        result /= scale.reshape((1, -1) if axis == 1 else (-1, 1))
        result = hadamard128(result, axis=axis)
    return result


def public_activations_to_inner_native(values, scale):
    """Prescale calibration rows with the FP32 Metal flush boundaries."""
    values = np.asarray(values, dtype=np.float32).copy()
    scale = np.asarray(scale, dtype=np.float32).copy()
    if (
        values.ndim != 2
        or scale.shape != (values.shape[1],)
        or not np.isfinite(values).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("invalid calibration rows or scale")
    _flush_subnormals(values)
    _flush_subnormals(scale)
    return hadamard128(values * scale)
