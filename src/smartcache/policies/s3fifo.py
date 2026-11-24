from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class S3FIFOPolicy:
    """S3-FIFO eviction (Yang et al., SOSP 2023).

    The cache is split into three FIFO queues:

    * small (S, about 10% of the slots): every new object starts here;
    * main (M, the rest): objects that proved useful;
    * ghost (G): keys only (no data) of objects recently evicted from S.

    Evicting from S: an object hit at least once moves to M, otherwise it is
    evicted and its key is remembered in G. A new object whose key is in G
    goes straight to M. Evicting from M: an object with remaining hits is
    reinserted (its hit count is decremented, max 3), otherwise it is evicted.
    This quickly removes "one-hit wonders" (questions asked only once), which
    are very common in real traffic.

    Hits are detected from entry.frequency, which the simulator increments on
    every hit, so the policy only needs to be called at eviction time.
    """

    name = "S3-FIFO"

    def __init__(self, small_fraction: float = 0.1, max_frequency: int = 3) -> None:
        if not 0 < small_fraction < 1:
            raise ValueError("small_fraction must be between 0 and 1")
        self.small_fraction = small_fraction
        self.max_frequency = max_frequency
        self.small: list[str] = []
        self.main: list[str] = []
        self.ghost: OrderedDict[str, None] = OrderedDict()
        # frequency = min(max_frequency, entry.frequency - 1 - offset[key])
        self._offset: dict[str, int] = {}

    def _frequency(self, entry: CacheEntry) -> int:
        hits = entry.frequency - 1 - self._offset.get(entry.key, 0)
        return max(0, min(self.max_frequency, hits))

    def _set_frequency(self, entry: CacheEntry, value: int) -> None:
        self._offset[entry.key] = entry.frequency - 1 - value

    def _sync(self, cache: Sequence[CacheEntry]) -> None:
        """Place objects inserted since the last eviction (in insertion order)."""
        tracked = set(self.small) | set(self.main)
        for entry in cache:
            if entry.key in tracked:
                continue
            # Count every hit since insertion (the simulator starts each new
            # entry at frequency 1), including hits before we first saw it.
            self._offset[entry.key] = 0
            if entry.key in self.ghost:
                del self.ghost[entry.key]
                self.main.append(entry.key)
            else:
                self.small.append(entry.key)

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        self._sync(cache)
        entries = {entry.key: entry for entry in cache}
        small_target = max(1, int(len(cache) * self.small_fraction))
        ghost_limit = max(1, len(cache) - small_target)

        # Each pass either evicts or moves one object, so this terminates.
        for _ in range(4 * len(cache) * (self.max_frequency + 1)):
            if self.small and (len(self.small) >= small_target or not self.main):
                key = self.small.pop(0)
                entry = entries[key]
                if self._frequency(entry) >= 1:
                    self._set_frequency(entry, 0)
                    self.main.append(key)
                    continue
                self.ghost[key] = None
                while len(self.ghost) > ghost_limit:
                    self.ghost.popitem(last=False)
                self._offset.pop(key, None)
                return key

            key = self.main.pop(0)
            entry = entries[key]
            frequency = self._frequency(entry)
            if frequency > 0:
                self._set_frequency(entry, frequency - 1)
                self.main.append(key)
                continue
            self._offset.pop(key, None)
            return key
        raise RuntimeError("S3-FIFO failed to find a victim")