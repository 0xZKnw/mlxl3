"""CrossHair entry points call the patched production converter, without I/O."""

from ponyexl3.cli.convert import _fixed_uniform_bits


def uniform_without_head(k: int, head_bits: int) -> int | None:
    """
    pre: 2 <= k <= 8
    pre: 1 <= head_bits <= 8
    post: _ == k
    """
    return _fixed_uniform_bits(float(k), head_bits, [k], {}, [0.0])


def uniform_with_head(k: int, head_bits: int) -> int | None:
    """
    pre: 2 <= k <= 8
    pre: 1 <= head_bits <= 8
    post: (_ == k) == (head_bits == k)
    """
    return _fixed_uniform_bits(float(k), head_bits, [k], {"lm_head": [head_bits]}, [0.0])


def benchmark_rejects_empty_matrix(repeats: int, iterations: int) -> bool:
    """
    pre: 1 <= repeats <= 8
    pre: 1 <= iterations <= 8
    post: _
    """
    from bench_quantize_packing import run

    try:
        run("unused-report.json", repeats=repeats, iterations=iterations, tiles=())
    except ValueError:
        return True
    return False


def benchmark_rejects_invalid_bits(k: int) -> bool:
    """
    pre: k < 1 or k > 8
    post: _
    """
    from bench_quantize_packing import run

    try:
        run("unused-report.json", repeats=1, iterations=1, tiles=(1,), bits=(k,))
    except ValueError:
        return True
    return False
