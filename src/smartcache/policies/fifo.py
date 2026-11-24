from __future__ import annotations

from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class FIFOPolicy:
    name = "FIFO"

    def __init__(self) -> None:
        self.insertion_order: dict[str, int] = {}
        self.counter = 0

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        # In CacheSimulator, cache is a list where items are appended on insertion.
        # Thus cache[0] is the oldest item currently in the cache.
        victim_key = cache[0].key
        return victim_key

