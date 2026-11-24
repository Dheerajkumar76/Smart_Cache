"""Safety guard (circuit breaker) for a learned eviction policy.

A learned policy can misbehave: the traffic changes in a way it never saw,
the model file is wrong, or it simply makes worse choices than a simple
rule. Production systems therefore never trust a model blindly.

GuardedPolicy runs the learned (primary) policy and, alongside it, a cheap
"shadow" LRU cache that only tracks keys. Both see every request, so the
guard knows the miss cost the real cache paid and the miss cost LRU WOULD
have paid over a recent window.

* If the primary pays clearly more than the shadow LRU (more than `margin`)
  over the last `window` requests, the guard switches to the fallback
  policy (LRU) and records an event.
* After `cooldown` requests it gives the primary another chance.
* If the primary raises an error, the fallback decides that eviction.

While the fallback decides, the primary is told about each decision
(record_decision) so its history stays correct for when it comes back.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry
from smartcache.policies.lru import LRUPolicy

logger = logging.getLogger(__name__)


class GuardedPolicy:
    def __init__(
        self,
        primary: Any,
        capacity: int,
        fallback: Any | None = None,
        window: int = 2000,
        margin: float = 0.05,
        cooldown: int = 5000,
        check_every: int = 250,
    ):
        self.primary = primary
        self.fallback = fallback or LRUPolicy()
        self.name = f"Guarded{getattr(primary, 'name', 'Policy')}"
        self.capacity = capacity
        self.window = window
        self.margin = margin
        self.cooldown = cooldown
        self.check_every = check_every

        self.using_fallback = False
        self.retry_at = 0
        self.events: list[dict[str, Any]] = []
        self.errors = 0
        # Recent miss costs of the real cache and of the shadow LRU cache.
        self._real: deque[float] = deque(maxlen=window)
        self._shadow: deque[float] = deque(maxlen=window)
        self._shadow_keys: OrderedDict[str, None] = OrderedDict()

    # -- monitoring -------------------------------------------------------
    def observe_request(self, key: str, cost: float, hit: bool, timestamp: int) -> None:
        self._real.append(0.0 if hit else cost)

        if key in self._shadow_keys:
            self._shadow_keys.move_to_end(key)
            self._shadow.append(0.0)
        else:
            self._shadow.append(cost)
            self._shadow_keys[key] = None
            if len(self._shadow_keys) > self.capacity:
                self._shadow_keys.popitem(last=False)

        if timestamp % self.check_every == 0:
            self._check(timestamp)

    def _check(self, timestamp: int) -> None:
        if self.using_fallback:
            if timestamp >= self.retry_at:
                self.using_fallback = False
                self._record("retry_primary", timestamp)
            return
        if len(self._real) < self.window:
            return
        real, shadow = sum(self._real), sum(self._shadow)
        if real > shadow * (1 + self.margin):
            self.using_fallback = True
            self.retry_at = timestamp + self.cooldown
            self._record("switch_to_fallback", timestamp, real=real, shadow=shadow)

    def _record(self, event: str, timestamp: int, **details: float) -> None:
        self.events.append({"event": event, "timestamp": timestamp, **details})
        logger.warning("Cache guard %s at request %d %s", event, timestamp, details)

    # -- decisions --------------------------------------------------------
    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not self.using_fallback:
            try:
                return self.primary.choose_victim(cache, incoming, timestamp, context)
            except (ValueError, KeyError, RuntimeError, FloatingPointError) as error:
                self.errors += 1
                logger.exception("Primary policy failed; using fallback: %s", error)

        victim = self.fallback.choose_victim(cache, incoming, timestamp, context)
        record = getattr(self.primary, "record_decision", None)
        if record is not None:
            record(cache, incoming, timestamp, victim)
        return victim