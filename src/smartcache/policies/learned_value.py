"""Learned value-based eviction that works for any cache size.

Idea
----
Instead of one network output per cache slot (which only works for a fixed,
tiny cache), a small network looks at ONE cached object at a time and
predicts how valuable it is to keep:

    keep_value(object) = retrieval_cost * predicted_reuse(object)

predicted_reuse estimates gamma ** (requests until this object is asked
for again), i.e. the discounted return of keeping it, in the reinforcement
learning sense. It is close to 1 if the object will be needed soon and close
to 0 if it will not be needed for a long time (or never). On every eviction
all cached objects are scored and the one with the lowest keep_value is
evicted. The same network therefore works for 3 slots or 10,000.

Training labels come from hindsight: replaying a recorded trace tells us
when each object was actually requested next (Monte Carlo value targets).

Features use only information available at decision time (no future), so
the policy can be deployed. Hits are detected from entry.frequency, which the
simulator increments on every hit.
"""

from __future__ import annotations

import random
from math import exp, log1p
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smartcache.core.interfaces import CacheEntry

FEATURE_NAMES = [
    "log_cost",
    "log_age",
    "log_hits_in_cache",
    "log_time_in_cache",
    "log_mean_gap",
    "log_previous_misses",
    "log_time_since_first_seen",
    "hit_rate_in_cache",
    "log_total_requests",
    "long_term_rate",
    "log_recent_requests_short",
    "log_recent_requests_long",
]
# Time scales (in requests) for the recency-weighted popularity features.
DECAY_SHORT = 500.0
DECAY_LONG = 3000.0
FEATURE_COUNT = len(FEATURE_NAMES)


class ObjectFeatureTracker:
    """Keeps the per-object history needed to build features."""

    def __init__(self) -> None:
        self.insert_time: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self.misses: dict[str, int] = {}
        # Hits from earlier stays in the cache (added when an object leaves).
        self.past_hits: dict[str, int] = {}
        # Hits of the current stay already added to the decayed counters.
        self.counted_hits: dict[str, int] = {}
        # key -> [short value, long value, time of last update]
        self.decayed: dict[str, list[float]] = {}

    def _add_requests(self, key: str, count: int, when: int) -> None:
        """Add requests at time `when` to the exponentially decayed counters."""
        state = self.decayed.setdefault(key, [0.0, 0.0, float(when)])
        elapsed = max(0.0, when - state[2])
        state[0] = state[0] * exp(-elapsed / DECAY_SHORT) + count
        state[1] = state[1] * exp(-elapsed / DECAY_LONG) + count
        state[2] = max(state[2], float(when))

    def _decayed_now(self, key: str, now: int) -> tuple[float, float]:
        state = self.decayed.get(key)
        if state is None:
            return 0.0, 0.0
        elapsed = max(0.0, now - state[2])
        return (
            state[0] * exp(-elapsed / DECAY_SHORT),
            state[1] * exp(-elapsed / DECAY_LONG),
        )

    def observe(self, cache: Sequence[CacheEntry], incoming_key: str, timestamp: int) -> None:
        # Objects we have not seen yet were admitted during warm-up.
        for entry in cache:
            if entry.key not in self.insert_time:
                self.insert_time[entry.key] = entry.last_access - (entry.frequency - 1)
                self.first_seen.setdefault(entry.key, self.insert_time[entry.key])
                self.misses.setdefault(entry.key, 1)
                self.counted_hits[entry.key] = 0
                self._add_requests(entry.key, 1, self.insert_time[entry.key])
            # Hits are seen lazily: credit new ones at the time of the last hit.
            new_hits = entry.frequency - 1 - self.counted_hits.get(entry.key, 0)
            if new_hits > 0:
                self._add_requests(entry.key, new_hits, entry.last_access)
                self.counted_hits[entry.key] = entry.frequency - 1

    def admitted(self, key: str, timestamp: int) -> None:
        """Called for the incoming key: it missed and is inserted now."""
        self.insert_time[key] = timestamp
        self.first_seen.setdefault(key, timestamp)
        self.misses[key] = self.misses.get(key, 0) + 1
        self.counted_hits[key] = 0
        self._add_requests(key, 1, timestamp)

    def evicted(self, entry: CacheEntry) -> None:
        self.insert_time.pop(entry.key, None)
        self.counted_hits.pop(entry.key, None)
        self.past_hits[entry.key] = self.past_hits.get(entry.key, 0) + entry.frequency - 1

    def features(self, cache: Sequence[CacheEntry], timestamp: int) -> np.ndarray:
        rows = np.empty((len(cache), FEATURE_COUNT), dtype=np.float32)
        for i, entry in enumerate(cache):
            inserted = self.insert_time.get(entry.key, entry.last_access)
            hits = entry.frequency - 1
            age = max(0, timestamp - entry.last_access)
            in_cache = max(1, timestamp - inserted)
            total = self.misses.get(entry.key, 1) + self.past_hits.get(entry.key, 0) + hits
            known_for = max(1, timestamp - self.first_seen.get(entry.key, inserted))
            short, long = self._decayed_now(entry.key, timestamp)
            if hits > 0:
                mean_gap = max(1.0, (entry.last_access - inserted) / hits)
            else:
                mean_gap = float(in_cache)
            rows[i] = (
                log1p(entry.retrieval_cost_ms) / 10.0,
                log1p(age) / 10.0,
                log1p(hits) / 5.0,
                log1p(in_cache) / 10.0,
                log1p(mean_gap) / 10.0,
                log1p(self.misses.get(entry.key, 1)) / 5.0,
                log1p(known_for) / 10.0,
                hits / in_cache,
                log1p(total) / 5.0,
                min(1.0, 100.0 * total / known_for),
                log1p(short) / 3.0,
                log1p(long) / 3.0,
            )
        return rows


class NumpyMLP:
    """Tiny multilayer perceptron evaluated with NumPy (fast, no PyTorch)."""

    def __init__(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]):
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted reuse in [0, 1] for each row."""
        hidden = features
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.maximum(hidden @ weight + bias, 0.0)
        logits = hidden @ self.weights[-1] + self.biases[-1]
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        arrays = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        arrays["metadata"] = np.array(repr(dict(metadata or {})))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> tuple["NumpyMLP", str]:
        data = np.load(path, allow_pickle=False)
        layers = len([name for name in data.files if name.startswith("w")])
        model = cls(
            [data[f"w{i}"] for i in range(layers)],
            [data[f"b{i}"] for i in range(layers)],
        )
        if model.weights[0].shape[0] != FEATURE_COUNT:
            raise ValueError("Model feature count does not match this policy")
        return model, str(data["metadata"])


class LearnedValuePolicy:
    """Evict the cached object with the lowest cost * predicted reuse."""

    name = "LearnedValue"

    def __init__(self, model: NumpyMLP | str | Path):
        if isinstance(model, (str, Path)):
            model, self.metadata = NumpyMLP.load(model)
        else:
            self.metadata = ""
        self.model = model
        self.tracker = ObjectFeatureTracker()

    def keep_values(self, cache: Sequence[CacheEntry], timestamp: int) -> np.ndarray:
        features = self.tracker.features(cache, timestamp)
        costs = np.array([entry.retrieval_cost_ms for entry in cache], dtype=np.float32)
        return costs * self.model.predict(features)

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        incoming_key = str(incoming["key"])
        self.tracker.observe(cache, incoming_key, timestamp)
        victim = cache[int(np.argmin(self.keep_values(cache, timestamp)))]
        self.tracker.evicted(victim)
        self.tracker.admitted(incoming_key, timestamp)
        return victim.key

    def record_decision(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        victim_key: str,
    ) -> None:
        """Keep the feature history up to date when another policy decided."""
        self.tracker.observe(cache, str(incoming["key"]), timestamp)
        victim = next(entry for entry in cache if entry.key == victim_key)
        self.tracker.evicted(victim)
        self.tracker.admitted(str(incoming["key"]), timestamp)


class TrainingCollector:
    """Wraps a behaviour policy and records (features, reuse label) pairs.

    The behaviour policy decides what is evicted. At each eviction a random
    sample of cached objects is recorded together with its hindsight label
    gamma ** (requests until next use), read from the full trace.
    """

    def __init__(
        self,
        behaviour: Any,
        requests: Sequence[Mapping[str, Any]],
        gamma: float,
        samples_per_eviction: int = 8,
        seed: int = 0,
    ):
        from smartcache.policies.oracle import _FutureIndex

        self.behaviour = behaviour
        self.name = f"Collect({getattr(behaviour, 'name', 'policy')})"
        self.future = _FutureIndex(requests)
        self.gamma = gamma
        self.samples = samples_per_eviction
        self.rng = random.Random(seed)
        self.tracker = ObjectFeatureTracker()
        self.features: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

    def choose_victim(self, cache, incoming, timestamp, context=None) -> str:
        incoming_key = str(incoming["key"])
        self.tracker.observe(cache, incoming_key, timestamp)
        chosen = self.rng.sample(range(len(cache)), min(self.samples, len(cache)))
        subset = [cache[i] for i in chosen]
        self.features.append(self.tracker.features(subset, timestamp))
        distances = np.array(
            [self.future.distance(entry.key, timestamp) for entry in subset]
        )
        self.labels.append(np.power(self.gamma, distances).astype(np.float32))

        victim = self.behaviour.choose_victim(cache, incoming, timestamp, context)
        self.tracker.evicted(next(entry for entry in cache if entry.key == victim))
        self.tracker.admitted(incoming_key, timestamp)
        return victim

    def dataset(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.features), np.concatenate(self.labels)