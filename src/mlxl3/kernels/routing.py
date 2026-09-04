"""Integer-only routed-token permutation helpers."""

from functools import cache

import mlx.core as mx


@cache
def _inverse_kernel():
    return mx.fast.metal_kernel(
        name="mlxl3_inverse_permutation_v1",
        input_names=["order"],
        output_names=["inverse"],
        source="""
            uint i = thread_position_in_grid.x;
            if (i < uint(order_shape[0])) inverse[order[i]] = i;
        """,
    )


def inverse_permutation(order: mx.array) -> mx.array:
    """Invert a permutation (e.g. argsort output), without sorting it again.

    The caller must supply every index in [0, size) exactly once. This internal
    precondition avoids a validation/synchronization pass in the hot path.
    """
    if order.ndim != 1 or order.dtype != mx.uint32:
        raise ValueError("inverse_permutation requires a uint32 permutation vector")
    if order.size == 0:
        return order
    return _inverse_kernel()(
        inputs=[order], grid=(order.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[order.shape], output_dtypes=[mx.uint32],
    )[0]
