"""Reuse immutable vocabulary metadata, never a live streaming text state."""
import weakref
from copy import copy

from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, SPMStreamingDetokenizer


def cache_detokenizer_metadata(tokenizer):
    if getattr(tokenizer, '_mlxl3_detokenizer_metadata_cached', False):
        return False
    original = getattr(tokenizer, '_detokenizer_class', None)
    if original is None:
        return False
    prototype = original(tokenizer)
    if type(prototype) not in (BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        return False
    prototype.tokenmap = tuple(prototype.tokenmap)
    owner = weakref.ref(tokenizer)

    def fresh(current):
        if current is not owner():
            return original(current)
        result = copy(prototype)
        result.reset()
        return result

    tokenizer._detokenizer_class = fresh
    tokenizer._mlxl3_detokenizer_metadata_cached = True
    return True
