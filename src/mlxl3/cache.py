"""Explicit ownership for lazy copy-on-write forks of ordinary KV caches.

MLX full slices create distinct array handles sharing immutable storage; a
subsequent indexed update detaches storage. This is not a paged-attention
implementation: first write can still copy the whole retained allocation.
"""
from dataclasses import dataclass

from mlx_lm.models.cache import KVCache


@dataclass(eq=False)
class StorageOwner:
    nbytes: int


class SharedKVCache(KVCache):
    def __init__(self):
        super().__init__()
        self.storage_owner = None

    @classmethod
    def adopt(cls, cache):
        if type(cache) is KVCache:
            cache.__class__ = cls
            cache.storage_owner = None
        if type(cache) is not cls:
            raise TypeError('only ordinary KVCache can adopt COW ownership')
        return cache

    def update_and_fetch(self, keys, values):
        # The old owner's other forks remain valid. Count this writer
        # independently even if MLX is able to avoid the physical copy.
        self.storage_owner = None
        return super().update_and_fetch(keys, values)

    @property
    def state(self):
        return super().state

    @state.setter
    def state(self, values):
        self.storage_owner = None
        KVCache.state.fset(self, values)

    def fork(self):
        fork = SharedKVCache()
        if self.keys is not None:
            if self.storage_owner is None:
                self.storage_owner = StorageOwner(self.keys.nbytes + self.values.nbytes)
            # Do not alias the Python array handles themselves: indexed
            # assignment changes a handle even when its storage is immutable.
            fork.keys = self.keys[:]
            fork.values = self.values[:]
            fork.storage_owner = self.storage_owner
        fork.offset = self.offset
        return fork


def compact_conv_states(cache, prefill_tokens):
    """Detach small convolution-state slices from long prefill backing arrays.

    Limit to the small B,L,D state layout used by LFM/Qwen convolution caches;
    never copy their large recurrent matrices. mx.array deliberately copies,
    unlike contiguous() which can keep a contiguous slice's large parent alive.
    """
    if prefill_tokens < 128:
        return
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    for item in cache:
        if type(item) is not ArraysCache:
            continue
        for index, value in enumerate(item.cache):
            if isinstance(value, mx.array) and value.ndim == 3 and 0 < value.shape[1] <= 16:
                item[index] = mx.array(value)
