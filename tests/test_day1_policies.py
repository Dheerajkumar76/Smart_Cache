from smartcache.core.cache import CacheSimulator
from smartcache.core.interfaces import CacheEntry
from smartcache.environment.workload import generate_workload
from smartcache.policies.adaptive_gdsf import AdaptiveGDSFPolicy
from smartcache.policies.contextual_bandit import LinUCBPolicy
from smartcache.policies.evict_cheapest import EvictCheapestPolicy
from smartcache.policies.fifo import FIFOPolicy
from smartcache.policies.gdsf import GDSFPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.oracle import optimal_miss_cost


def request(key, cost, size=1024):
    return {"key": key, "cost": cost, "size": size}


def test_evict_cheapest_picks_lowest_cost_then_oldest():
    cache = [
        CacheEntry("a", 50.0, 1024, 1, last_access=5),
        CacheEntry("b", 2.0, 1024, 1, last_access=9),
        CacheEntry("c", 2.0, 1024, 1, last_access=3),
    ]
    assert EvictCheapestPolicy().choose_victim(cache, request("d", 1), 10) == "c"


def test_optimum_matches_hand_computed_example():
    # Capacity 1: keeping "x" (cost 100) through the cheap "y" requests is best.
    trace = [request("x", 100), request("y", 1), request("x", 100), request("y", 1)]
    # Forced admission means every miss is admitted, so the best possible is
    # x(100) + y(1) + x(100) + y(1) = 202 with capacity 1, and
    # x(100) + y(1) = 101 with capacity 2.
    assert optimal_miss_cost(trace, capacity=1) == 202
    assert optimal_miss_cost(trace, capacity=2) == 101


def test_no_policy_beats_the_optimum():
    simulator = CacheSimulator(capacity=3)
    trace = generate_workload(600, seed=5, drift=True)
    best = optimal_miss_cost(trace, capacity=3)
    for policy in (
        LRUPolicy(),
        FIFOPolicy(),
        EvictCheapestPolicy(),
        GDSFPolicy(),
        AdaptiveGDSFPolicy(),
        LinUCBPolicy(),
    ):
        assert simulator.run(trace, policy)["total_miss_cost_ms"] >= best - 1e-9


def test_fresh_stateful_policies_give_identical_results():
    simulator = CacheSimulator(capacity=3)
    trace = generate_workload(800, seed=11, drift=True)
    for policy_class in (GDSFPolicy, AdaptiveGDSFPolicy, LinUCBPolicy):
        first = simulator.run(trace, policy_class())
        # Run a different trace in between: a fresh instance must not care.
        simulator.run(generate_workload(800, seed=12), policy_class())
        second = simulator.run(trace, policy_class())
        assert first["total_miss_cost_ms"] == second["total_miss_cost_ms"]


def test_gdsf_ages_out_objects_that_stop_being_used():
    # "old" is valuable and hit often at first, then never requested again.
    # With proper aging it must eventually be evicted instead of staying forever.
    trace = [request("old", 200) for _ in range(20)]
    trace += [request(f"new{i % 6}", 50) for i in range(400)]
    result = CacheSimulator(capacity=3).run(trace, GDSFPolicy())
    assert "old" not in result["final_cache"]


def test_adaptive_gdsf_sees_hits_between_evictions():
    policy = AdaptiveGDSFPolicy(window_size=50)
    trace = [request(k, 10) for k in "abc"] + [request("a", 10)] * 10
    trace += [request("d", 10)]
    CacheSimulator(capacity=3).run(trace, policy)
    # 3 warm-up misses (a, b, c), 10 hits on "a", then the miss for "d".
    assert list(policy.window) == [0] * 3 + [1] * 10 + [0]