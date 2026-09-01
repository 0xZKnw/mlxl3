"""Metal source fragments shared by EXL3 kernels."""

from __future__ import annotations

from mlxl3.codec.perm import permutation, permutation_inverse

CODEBOOK_HEADER = r"""
inline half mlxl3_half_from_bits(ushort bits) {
    return as_type<half>(bits);
}

inline half mlxl3_decode_codeword(uint x, int cb) {
    x &= 0xffffu;
    if (cb == 0) {
        x = x * 89226354u + 64248484u;
        x = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
        half lo = mlxl3_half_from_bits(ushort(x & 0xffffu));
        half hi = mlxl3_half_from_bits(ushort(x >> 16));
        return lo + hi;
    }
    if (cb == 1) {
        x *= 0xCBAC1FEDu;
        x = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
        half lo = mlxl3_half_from_bits(ushort(x & 0xffffu));
        half hi = mlxl3_half_from_bits(ushort(x >> 16));
        return lo + hi;
    }
    x *= 0x83DCD12Du;
    uint sum = 0x6400u;
    sum += x & 0xffu;
    sum += (x >> 8) & 0xffu;
    sum += (x >> 16) & 0xffu;
    sum += (x >> 24) & 0xffu;
    half h = mlxl3_half_from_bits(ushort(sum));
    half inv = mlxl3_half_from_bits(ushort(0x1EEEu));
    half bias = mlxl3_half_from_bits(ushort(0xC931u));
    return fma(h, inv, bias);
}
"""


def specialized_codebook_header(mode: int) -> str:
    """Branch-free decode source for a kernel already specialized by codebook."""

    if mode == 0:
        body = """
    uint bits = x * 89226354u + 64248484u;
    bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);
    half2 values = as_type<half2>(bits);
    return float(values.x + values.y);
"""
    elif mode == 1:
        body = """
    uint bits = x * 0xCBAC1FEDu;
    bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);
    half2 values = as_type<half2>(bits);
    return float(values.x + values.y);
"""
    elif mode == 2:
        body = """
    uint bits = x * 0x83DCD12Du;
    uint sum = 0x6400u;
    sum += bits & 0xffu;
    sum += (bits >> 8) & 0xffu;
    sum += (bits >> 16) & 0xffu;
    sum += (bits >> 24) & 0xffu;
    half value = as_type<half>(ushort(sum));
    half inv = as_type<half>(ushort(0x1EEEu));
    half bias = as_type<half>(ushort(0xC931u));
    return float(value * inv + bias);
"""
    else:
        raise ValueError(f"unknown EXL3 codebook mode {mode}")
    return f"""
inline float mlxl3_decode_codeword(uint x, int unused_cb) {{
    x &= 0xffffu;
{body}
}}
"""


def permutation_header() -> str:
    values = ", ".join(str(int(value)) for value in permutation_inverse())
    return f"constant ushort mlxl3_perm_inv[256] = {{{values}}};\n"


def forward_permutation_header() -> str:
    values = ", ".join(str(int(value)) for value in permutation())
    return f"constant ushort mlxl3_perm[256] = {{{values}}};\n"
