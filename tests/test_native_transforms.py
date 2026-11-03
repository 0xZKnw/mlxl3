"""Conversion Hadamard checks with an independent sign-matrix oracle."""

import numpy as np
import pytest

from mlxl3_quantizer.transforms import HAD_SCALE, hadamard128, public_matrix_to_inner_native


def test_every_basis_vector_has_the_independent_walsh_sign():
    expected = (
        np.array(
            [[(-1) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)],
            dtype=np.float32,
        )
        * HAD_SCALE
    )
    np.testing.assert_array_equal(hadamard128(np.eye(128, dtype=np.float32)), expected)
    np.testing.assert_array_equal(hadamard128(np.eye(128, dtype=np.float32), axis=0), expected)


@pytest.mark.parametrize(
    "shape,axis", [((4, 256), 1), ((256, 4), 0), ((256, 384), 1), ((256, 384), 0)]
)
def test_strided_blocks_preserve_the_mathematical_transform(shape, axis):
    values = np.random.default_rng(75).normal(size=shape).astype(np.float32)[:, ::-1]
    signs = np.array(
        [[(-1) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)], dtype=np.float64
    ) / np.sqrt(128)
    expected = values.astype(np.float64).copy()
    for start in range(0, shape[axis], 128):
        if axis == 1:
            expected[:, start : start + 128] = (
                values[:, start : start + 128].astype(np.float64) @ signs
            )
        else:
            expected[start : start + 128] = signs @ values[start : start + 128].astype(np.float64)
    np.testing.assert_allclose(hadamard128(values, axis=axis), expected, rtol=2e-5, atol=1e-6)


def test_public_matrix_basis_preserves_requested_axis_and_scale_order():
    x = np.eye(128, dtype=np.float32)
    sv = np.full(128, 2, dtype=np.float32)
    su = np.full(128, -4, dtype=np.float32)
    actual = public_matrix_to_inner_native(x, suh=su, svh=sv)
    np.testing.assert_allclose(actual, x * np.float32(-0.125), rtol=0, atol=3e-8)


@pytest.mark.parametrize(
    "values,axis",
    [
        (np.zeros((0, 128)), 1),
        (np.zeros((128, 0)), 0),
        (np.zeros((1, 127)), 1),
        (np.zeros(128), 1),
        (np.full((1, 128), np.nan), 1),
        (np.zeros((1, 128)), 2),
    ],
)
def test_bad_shapes_and_values_reject_before_transform(values, axis):
    with pytest.raises(ValueError):
        hadamard128(values, axis=axis)


@pytest.mark.parametrize("scale", [np.zeros(128), np.full(128, np.inf), np.ones(127)])
def test_invalid_scales_reject(scale):
    with pytest.raises(ValueError):
        public_matrix_to_inner_native(np.zeros((128, 128)), suh=scale)


def test_signed_subnormal_rows_flush_before_the_butterfly_can_amplify_them():
    values = np.empty((4, 128), dtype=np.float32)
    values[0] = np.nextafter(np.float32(0), np.float32(1))
    values[1] = -values[0]
    values[2] = np.finfo(np.float32).tiny / 4
    values[3] = -values[2]
    actual = hadamard128(values)
    expected = np.zeros((4, 128), dtype=np.float32)
    expected[[1, 3], 0] = np.float32(-0.0)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert values[0, 0] != 0


def test_activation_prescale_cannot_amplify_subnormal_inputs():
    from mlxl3_quantizer.transforms import public_activations_to_inner_native

    values = np.full((1, 128), np.finfo(np.float32).tiny / 4, dtype=np.float32)
    actual = public_activations_to_inner_native(values, np.full(128, 2**20))
    np.testing.assert_array_equal(actual, np.zeros_like(values))


def test_eight_independent_metal_fixtures_are_bit_exact_without_mlx():
    import hashlib
    import json
    from pathlib import Path

    from mlxl3_quantizer.transforms import public_activations_to_inner_native

    root = Path(__file__).parent / "fixtures/native-converter-host"
    report = json.loads((root / "hadamard.json").read_text())
    assert hashlib.sha256((root / "hadamard.npz").read_bytes()).hexdigest() == report["sha256"]
    count = 0
    with np.load(root / "hadamard.npz", allow_pickle=False) as corpus:
        for case in report["cases"]:
            name, axis = case["name"], case["axis"]
            x, expected = corpus[f"{name}-input"], corpus[f"{name}-output"]
            if axis is None:
                actual = public_matrix_to_inner_native(
                    x, suh=corpus[f"{name}-su"], svh=corpus[f"{name}-sv"]
                )
            elif axis == -1:
                actual = public_activations_to_inner_native(x, corpus[f"{name}-su"])
            else:
                actual = hadamard128(x, axis=axis)
            assert x.shape == expected.shape == actual.shape == tuple(case["shape"])
            assert x.size == case["count"] and np.isfinite(actual).all()
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            count += actual.size
    assert len(report["cases"]) == 8 and count == report["words"] == 197888


def test_finite_input_that_overflows_a_butterfly_never_returns_infinite_weights():
    values = np.full((1, 128), np.finfo(np.float32).max, dtype=np.float32)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="overflowed"),
    ):
        hadamard128(values)
