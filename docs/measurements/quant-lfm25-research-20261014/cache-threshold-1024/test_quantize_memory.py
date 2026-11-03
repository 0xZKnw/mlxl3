import pytest

from mlxl3_quantizer.memory import should_trim_conversion_cache


@pytest.mark.parametrize(
    "cached,expected",
    [
        (0, False),
        (512 << 20, False),
        ((1024 << 20) - 1, False),
        (1024 << 20, False),
        ((1024 << 20) + 1, True),
        ((1 << 64) - 1, True),
    ],
)
def test_conversion_cache_threshold_keeps_hot_pool_and_releases_excess(cached, expected):
    assert should_trim_conversion_cache(cached) is expected
