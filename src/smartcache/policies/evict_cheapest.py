from __future__ import annotations

from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class EvictCheapestPolicy:
    """Evict the resident object that is cheapest to fetch again.

    Ties are broken by evicting the least recently used object. This is the
    simplest possible cost-aware rule and acts as a sanity baseline: a learned
    policy should beat it before claiming to have learned anything.
    """

    name = "EvictCheapest"

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        return min(
            cache, key=lambda entry: (entry.retrieval_cost_ms, entry.last_access)
        ).key