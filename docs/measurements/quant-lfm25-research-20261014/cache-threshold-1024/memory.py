"""Keep reusable converter buffers between LDLQ checkpoints."""


def should_trim_conversion_cache(cached_bytes: int) -> bool:
    return cached_bytes > 1024 * 1024 * 1024
