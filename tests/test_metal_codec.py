from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode, decode_codewords
from mlxl3.codec.trellis import pack_trellis, unpack_trellis
from mlxl3.kernels import decode_trellis_metal, pack_trellis_metal


def test_mul1_swar_all_states_bit_exact(monkeypatch):
    from mlxl3.kernels.common import specialized_codebook_header
    outputs = []
    words = mx.arange(65536, dtype=mx.uint32)
    for enabled in ('0', '1'):
        monkeypatch.setenv('MLXL3_MUL1_SWAR', enabled)
        kernel = mx.fast.metal_kernel(
            name='mlxl3_test_mul1_swar_' + enabled,
            input_names=['words'], output_names=['values'],
            header=specialized_codebook_header(2),
            source='uint i=thread_position_in_grid.x; values[i]=half(mlxl3_decode_codeword(words[i],2));',
            compile_options={'math_mode': 'safe'},
        )
        values = kernel(inputs=[words], grid=(65536,1,1), threadgroup=(256,1,1),
                        output_shapes=[(65536,)], output_dtypes=[mx.float16])[0]
        mx.eval(values)
        outputs.append(np.asarray(values).view(np.uint16))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    expected = decode_codewords(np.arange(65536, dtype=np.uint16), CodebookMode.MUL1)
    np.testing.assert_array_equal(outputs[1], expected.view(np.uint16))


@pytest.mark.parametrize("k", range(1, 9))
def test_metal_pack_matches_cpu(k: int) -> None:
    rng = np.random.default_rng(700 + k)
    encoded = rng.integers(0, 1 << k, size=(3, 256), dtype=np.uint16)
    actual = pack_trellis_metal(mx.array(encoded), k)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), pack_trellis(encoded, k))


@pytest.mark.parametrize("k", range(1, 9))
@pytest.mark.parametrize("mode", list(CodebookMode))
def test_metal_fused_decode_matches_cpu(k: int, mode: CodebookMode) -> None:
    rng = np.random.default_rng(9000 + 10 * k + int(mode))
    encoded = rng.integers(0, 1 << k, size=(2, 256), dtype=np.uint16)
    packed = pack_trellis(encoded, k)
    expected_words = unpack_trellis(packed, k)
    expected_values = decode_codewords(expected_words, mode)

    actual_words, actual_values = decode_trellis_metal(mx.array(packed), k, mode)
    mx.eval(actual_words, actual_values)
    np.testing.assert_array_equal(np.asarray(actual_words), expected_words)
    # Compare fp16 payloads, not a loose floating-point tolerance.
    np.testing.assert_array_equal(
        np.asarray(actual_values).view(np.uint16),
        expected_values.view(np.uint16),
    )
