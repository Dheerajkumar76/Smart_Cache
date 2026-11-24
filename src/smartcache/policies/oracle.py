"""Offline optimum for the fixed-slot cache.

This is not a policy that can be deployed: it reads the whole future trace.
It answers one question: what is the lowest total miss cost that ANY policy
could possibly achieve on this trace? Benchmarks use it as the 100% mark.
"""

from __future__ import annotations

from math import comb
from typing import Any, Mapping, Sequence

MAX_STATES = 20_000


def optimal_miss_cost(
    requests: Sequence[Mapping[str, Any]],
    capacity: int,
) -> float:
    """Exact minimum total miss cost with forced admission.

    Dynamic programming over every possible cache content. After each request
    we keep, for every reachable set of cached keys, the cheapest way to get
    there. On a miss with a full cache, every resident key is tried as the
    victim. Only feasible when the number of distinct objects is small.
    """
    keys = {str(request["key"]) for request in requests}
    if comb(len(keys), min(capacity, len(keys))) > MAX_STATES:
        raise ValueError(
            f"{len(keys)} objects with capacity {capacity} is too large "
            "for the exact optimum"
        )

    best: dict[frozenset[str], float] = {frozenset(): 0.0}
    for request in requests:
        key = str(request["key"])
        cost = float(request["cost"])
        following: dict[frozenset[str], float] = {}
        for cached, total in best.items():
            if key in cached:
                options = [(cached, total)]
            elif len(cached) < capacity:
                options = [(cached | {key}, total + cost)]
            else:
                options = [
                    ((cached - {victim}) | {key}, total + cost) for victim in cached
                ]
            for state, value in options:
                if value < following.get(state, float("inf")):
                    following[state] = value
        best = following
    return min(best.values())



class _FutureIndex:
    """Looks up the next time each key will be requested."""

    def __init__(self, requests: Sequence[Mapping[str, Any]]):
        from collections import defaultdict

        positions: dict[str, list[int]] = defaultdict(list)
        # CacheSimulator timestamps start at 1 for the first request.
        for timestamp, request in enumerate(requests, start=1):
            positions[str(request["key"])].append(timestamp)
        self._positions = dict(positions)

    def distance(self, key: str, timestamp: int) -> float:
        from bisect import bisect_right

        positions = self._positions.get(key, [])
        index = bisect_right(positions, timestamp)
        if index == len(positions):
            return float("inf")
        return float(positions[index] - timestamp)


class BeladyPolicy:
    """Offline reference: evict the object whose next request is furthest away.

    Belady's rule maximises hit ratio, but ignores cost. It reads the future,
    so it can only be used as a yardstick, never deployed.
    """

    name = "Belady"

    def __init__(self, requests: Sequence[Mapping[str, Any]]):
        self._future = _FutureIndex(requests)

    def choose_victim(self, cache, incoming, timestamp, context=None) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        return max(
            cache, key=lambda entry: self._future.distance(entry.key, timestamp)
        ).key


class CostBeladyPolicy:
    """Offline reference: evict the object with the lowest cost per request of
    waiting (retrieval_cost / distance to next request). Objects never needed
    again go first. Like Belady, it reads the future and is only a yardstick.
    """

    name = "CostBelady"

    def __init__(self, requests: Sequence[Mapping[str, Any]]):
        self._future = _FutureIndex(requests)

    def choose_victim(self, cache, incoming, timestamp, context=None) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")

        def keep_value(entry) -> float:
            distance = self._future.distance(entry.key, timestamp)
            if distance == float("inf"):
                return -1.0
            return entry.retrieval_cost_ms / distance

        return min(cache, key=keep_value).key