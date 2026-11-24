from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from smartcache.core.interfaces import CacheEntry
from smartcache.policies.gdsf import GDSFPolicy


class AdaptiveGDSFPolicy(GDSFPolicy):
    """GDSF whose frequency/cost exponents adapt to recent traffic.

    priority value = frequency**alpha * cost**beta / size

    * Skewed traffic (a few very popular objects) raises alpha so frequency
      matters more.
    * A sudden drop in hit rate is treated as drift: L is boosted so that
      stale objects age out faster.

    The policy is only called on evictions, so hit/miss telemetry is
    reconstructed from timestamps: every request between two evictions was a
    hit, and the request that triggered this eviction is a miss.
    """

    name = "AdaptiveGDSF"

    def __init__(self, window_size: int = 200, drift_boost: float = 1.3) -> None:
        super().__init__()
        self.alpha = 1.2
        self.beta = 1.0
        self.drift_boost = drift_boost
        self.window: deque[int] = deque(maxlen=window_size)
        self._last_eviction_timestamp: int | None = None
        self.drift_events = 0

    def _value(self, entry: CacheEntry) -> float:
        freq = max(1, entry.frequency)
        size_kb = max(0.5, entry.size_bytes / 1024.0)
        return (freq**self.alpha) * (entry.retrieval_cost_ms**self.beta) / size_kb

    def _record(self, timestamp: int, resident: int) -> None:
        if self._last_eviction_timestamp is None:
            # First eviction: each resident object entered through one miss,
            # every other earlier request was a hit.
            hits = max(0, timestamp - 1 - resident)
            self.window.extend([0] * resident)
        else:
            hits = max(0, timestamp - self._last_eviction_timestamp - 1)
        self.window.extend([1] * hits)
        self.window.append(0)
        self._last_eviction_timestamp = timestamp

    def _adapt(self, context: Mapping[str, Any] | None) -> None:
        if context is not None:
            counts = list(context.get("frequency_counts", {}).values())
            if len(counts) > 1:
                cv = float(np.std(counts) / max(1.0, np.mean(counts)))
                if cv > 1.2:
                    self.alpha, self.beta = 1.8, 0.9
                else:
                    self.alpha, self.beta = 1.0, 1.2

        if len(self.window) == self.window.maxlen:
            quarter = self.window.maxlen // 4
            values = list(self.window)
            older = float(np.mean(values[:quarter]))
            recent = float(np.mean(values[-quarter:]))
            if older > 0.4 and recent < older * 0.6:
                self.inflation_value *= self.drift_boost
                self.drift_events += 1
                self.window.clear()  # avoid firing again on the same drop

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        self._record(timestamp, len(cache))
        self._adapt(context)
        return super().choose_victim(cache, incoming, timestamp, context)