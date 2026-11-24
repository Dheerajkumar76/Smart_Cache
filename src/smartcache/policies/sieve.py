from __future__ import annotations

from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class SievePolicy:
    """SIEVE eviction (Zhang et al., NSDI 2024).

    Objects sit in a FIFO queue, newest at the head. Each object has a
    "visited" bit that is set when it is hit. A hand moves from the tail
    (oldest) towards the head: a visited object has its bit cleared and is
    skipped; the first unvisited object is evicted. The hand remembers its
    position between evictions and wraps back to the tail at the head.

    The simulator appends new entries to the end of `cache`, so the list order
    is the insertion order (index 0 = oldest = tail). Hits are detected from
    entry.frequency, which the simulator increments on every hit.
    """

    name = "SIEVE"

    def __init__(self) -> None:
        # Number of hits already "used up" by clearing the visited bit.
        self._cleared_hits: dict[str, int] = {}
        # Key the hand points to (None = start from the tail).
        self._hand: str | None = None

    def _visited(self, entry: CacheEntry) -> bool:
        return entry.frequency - 1 > self._cleared_hits.get(entry.key, 0)

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")

        keys = [entry.key for entry in cache]
        position = keys.index(self._hand) if self._hand in keys else 0
        # At most two passes: the first may clear every visited bit.
        for _ in range(2 * len(cache)):
            entry = cache[position]
            if self._visited(entry):
                self._cleared_hits[entry.key] = entry.frequency - 1
                position = (position + 1) % len(cache)
                continue
            # Evict this one; the hand moves on to the next newer object.
            following = position + 1
            self._hand = keys[following] if following < len(keys) else None
            self._cleared_hits.pop(entry.key, None)
            return entry.key
        raise RuntimeError("SIEVE failed to find a victim")