"""EXL3 inference and conversion for Apple Silicon."""

from .codec.codebook import CodebookMode, decode_codeword, decode_codewords
from .codec.trellis import pack_trellis, unpack_trellis
from .linear import EXL3Linear
from .moe import EXL3SwitchGLU

__all__ = [
    "CodebookMode",
    "EXL3Linear",
    "EXL3SwitchGLU",
    "decode_codeword",
    "decode_codewords",
    "pack_trellis",
    "unpack_trellis",
]

__version__ = "0.4.0"
