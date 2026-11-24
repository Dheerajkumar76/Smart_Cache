from __future__ import annotations

from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class GDSFPolicy:
    """GreedyDual-Size-Frequency (GDSF).

    Each object gets priority H = L + frequency * cost / size, computed when
    the object is inserted or last accessed, using the value of L at that
    moment. On eviction the lowest-H object is removed and L is raised to its
    H. Objects that stop being accessed keep their old H while L keeps rising,
    so they age out instead of staying in the cache forever.

    The simulator only calls the policy at eviction time, so accesses are
    detected lazily: if an entry's last_access changed since the policy last
    saw it, it was touched. L only changes during evictions, so every access
    since the previous eviction happened under the current L. That makes this
    lazy update exactly equivalent to updating H on every access.
    """

    name = "GDSF"

    def __init__(self) -> None:
        self.inflation_value = 0.0
        # key -> (last_access seen, priority H)
        self._priorities: dict[str, tuple[int, float]] = {}

    def _value(self, entry: CacheEntry) -> float:
        freq = max(1, entry.frequency)
        size_kb = max(0.5, entry.size_bytes / 1024.0)
        return freq * entry.retrieval_cost_ms / size_kb

    def _priority(self, entry: CacheEntry) -> float:
        stored = self._priorities.get(entry.key)
        if stored is None or stored[0] != entry.last_access:
            # Inserted or accessed since the last eviction: refresh H.
            stored = (entry.last_access, self.inflation_value + self._value(entry))
            self._priorities[entry.key] = stored
        return stored[1]

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")

        victim = min(cache, key=self._priority)
        self.inflation_value = self._priorities.pop(victim.key)[1]

        # Forget objects that are no longer resident.
        resident = {entry.key for entry in cache}
        for key in list(self._priorities):
            if key not in resident:
                del self._priorities[key]
        return victim.key