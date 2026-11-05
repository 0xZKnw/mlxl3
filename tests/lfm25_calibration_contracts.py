"""Symbolic searches exercise the production alignment guard without MLX or I/O."""

from quantize_lfm25_moe import input_row_stride


def accepts_broadcast_or_sorted_inputs(input_rows: int, stride: int) -> int:
    """
    pre: input_rows >= 1
    pre: stride >= 1
    post: _ == stride
    """
    return input_row_stride(input_rows, input_rows * stride)


def rejects_nonpositive_inputs(input_rows: int, route_slots: int) -> bool:
    """
    pre: input_rows <= 0 or route_slots <= 0
    post: _
    """
    try:
        input_row_stride(input_rows, route_slots)
    except ValueError:
        return True
    return False


def rejects_misaligned_inputs(input_rows: int, route_slots: int) -> bool:
    """
    pre: input_rows >= 1
    pre: route_slots >= 1
    pre: route_slots % input_rows != 0
    post: _
    """
    try:
        input_row_stride(input_rows, route_slots)
    except ValueError:
        return True
    return False
