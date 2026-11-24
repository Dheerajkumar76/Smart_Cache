from pathlib import Path

import numpy as np
import pytest

from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.policies.guarded import GuardedPolicy
from smartcache.policies.learned_value import (
    FEATURE_COUNT,
    LearnedValuePolicy,
    NumpyMLP,
    TrainingCollector,
)
from smartcache.policies.lru import LRUPolicy

MODEL = Path("artifacts/learned_value.npz")
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="trained model missing")


def tiny_model(seed: int = 0) -> NumpyMLP:
    rng = np.random.default_rng(seed)
    return NumpyMLP(
        [rng.normal(size=(FEATURE_COUNT, 4)), rng.normal(size=(4, 1))],
        [np.zeros(4), np.zeros(1)],
    )


class EvictMostExpensive:
    name = "EvictMostExpensive"

    def choose_victim(self, cache, incoming, timestamp, context=None):
        return max(cache, key=lambda entry: entry.retrieval_cost_ms).key


class AlwaysFails:
    name = "AlwaysFails"

    def choose_victim(self, cache, incoming, timestamp, context=None):
        raise RuntimeError("model unavailable")


def test_collector_records_finite_features_and_labels_in_range():
    trace = generate_chatbot_workload(3000, seed=1, drift=True)
    collector = TrainingCollector(LRUPolicy(), trace, gamma=0.99, samples_per_eviction=4)
    CacheSimulator(capacity=30).run(trace, collector)
    features, labels = collector.dataset()
    assert features.shape[1] == FEATURE_COUNT
    assert len(features) == len(labels) > 0
    assert np.isfinite(features).all()
    assert ((labels >= 0) & (labels <= 1)).all()


def test_model_save_and_load_round_trip(tmp_path):
    model = tiny_model()
    path = tmp_path / "model.npz"
    model.save(path, {"note": "test"})
    loaded, metadata = NumpyMLP.load(path)
    x = np.random.default_rng(1).normal(size=(5, FEATURE_COUNT)).astype(np.float32)
    assert np.allclose(model.predict(x), loaded.predict(x))
    assert "test" in metadata


def test_learned_policy_works_for_any_cache_size():
    trace = generate_chatbot_workload(2000, seed=2)
    for capacity in (3, 25, 150):
        result = CacheSimulator(capacity=capacity).run(trace, LearnedValuePolicy(tiny_model()))
        assert result["requests"] == 2000


@needs_model
def test_trained_model_beats_lru_on_unseen_traces():
    simulator = CacheSimulator(capacity=50)
    for seed, drift in ((39001, False), (39002, True)):
        trace = generate_chatbot_workload(6000, seed=seed, drift=drift)
        learned = simulator.run(trace, LearnedValuePolicy(MODEL))
        lru = simulator.run(trace, LRUPolicy())
        assert learned["total_miss_cost_ms"] < lru["total_miss_cost_ms"]


def test_simulator_reports_every_request_to_observing_policies():
    seen = []

    class Observer(LRUPolicy):
        def observe_request(self, key, cost, hit, timestamp):
            seen.append((key, hit, timestamp))

    trace = generate_chatbot_workload(500, seed=3)
    result = CacheSimulator(capacity=20).run(trace, Observer())
    assert len(seen) == 500
    assert sum(hit for _, hit, _ in seen) == result["hits"]


def test_guard_switches_away_from_a_bad_policy():
    trace = generate_chatbot_workload(12000, seed=4)
    simulator = CacheSimulator(capacity=50)
    bad = simulator.run(trace, EvictMostExpensive())
    guard = GuardedPolicy(EvictMostExpensive(), capacity=50)
    guarded = simulator.run(trace, guard)
    assert any(event["event"] == "switch_to_fallback" for event in guard.events)
    assert guarded["total_miss_cost_ms"] < bad["total_miss_cost_ms"]


@needs_model
def test_guard_leaves_a_good_policy_alone():
    trace = generate_chatbot_workload(12000, seed=5, drift=True)
    simulator = CacheSimulator(capacity=50)
    plain = simulator.run(trace, LearnedValuePolicy(MODEL))
    guard = GuardedPolicy(LearnedValuePolicy(MODEL), capacity=50)
    guarded = simulator.run(trace, guard)
    assert guard.events == []
    assert guarded["total_miss_cost_ms"] == plain["total_miss_cost_ms"]


def test_guard_uses_fallback_when_primary_errors():
    trace = generate_chatbot_workload(800, seed=6)
    guard = GuardedPolicy(AlwaysFails(), capacity=20)
    guarded = CacheSimulator(capacity=20).run(trace, guard)
    lru = CacheSimulator(capacity=20).run(trace, LRUPolicy())
    assert guard.errors > 0
    assert guarded["total_miss_cost_ms"] == lru["total_miss_cost_ms"]