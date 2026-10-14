"""Source guard only; MLX allocation, concurrency and RAM are outside this model."""

from mlxl3_quantizer.memory import should_trim_conversion_cache


def hot_conversion_cache_is_retained(cached: int) -> bool:
    """
    pre: 0 <= cached <= 1073741824
    post: _
    """
    return not should_trim_conversion_cache(cached)


def excess_conversion_cache_is_released(cached: int) -> bool:
    """
    pre: 1073741824 < cached <= 18446744073709551615
    post: _
    """
    return should_trim_conversion_cache(cached)
