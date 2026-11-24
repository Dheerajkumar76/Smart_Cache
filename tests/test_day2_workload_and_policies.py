from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import (
    ChatbotWorkloadConfig,
    build_catalog,
    generate_chatbot_workload,
)
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.oracle import BeladyPolicy, CostBeladyPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy
from smartcache.policies.sieve import SievePolicy


def request(key, cost=100, size=1024):
    return {"key": key, "cost": cost, "size": size}


def test_chatbot_workload_is_reproducible_and_consistent():
    first = generate_chatbot_workload(3000, seed=7, drift=True)
    second = generate_chatbot_workload(3000, seed=7, drift=True)
    assert first == second
    assert len(first) == 3000
    # The simulator requires each key to keep the same cost and size.
    CacheSimulator(capacity=50).run(first, LRUPolicy())


def test_chatbot_costs_follow_the_llm_latency_model():
    config = ChatbotWorkloadConfig(questions=200)
    for item in build_catalog(config, seed=1):
        expected = config.time_to_first_token_ms + item["tokens"] * config.ms_per_token
        assert item["cost"] == round(expected, 1)
        assert config.min_answer_tokens <= item["tokens"] <= config.max_answer_tokens


def test_chatbot_popularity_is_skewed():
    trace = generate_chatbot_workload(20000, seed=3)
    counts: dict[str, int] = {}
    for item in trace:
        counts[item["key"]] = counts.get(item["key"], 0) + 1
    top_100 = sum(sorted(counts.values(), reverse=True)[:100])
    # With Zipf popularity the 100 most asked questions (2% of the catalogue)
    # account for a large share of all traffic.
    assert top_100 / len(trace) > 0.3


def test_drift_changes_the_most_popular_questions():
    trace = generate_chatbot_workload(30000, seed=4, drift=True)
    phase = len(trace) // 3

    def top(requests):
        counts: dict[str, int] = {}
        for item in requests:
            counts[item["key"]] = counts.get(item["key"], 0) + 1
        return set(sorted(counts, key=counts.get, reverse=True)[:20])

    assert len(top(trace[:phase]) & top(trace[-phase:])) < 10


def test_sieve_clears_visited_bit_and_evicts_first_unvisited():
    # a, b, c fill the cache; "a" is hit; "d" must evict "b" (a is spared once).
    trace = [request(k) for k in ["a", "b", "c", "a", "d"]]
    result = CacheSimulator(capacity=3).run(trace, SievePolicy())
    assert sorted(result["final_cache"]) == ["a", "c", "d"]
    # Next miss: the hand moved past "b" to "c", which is unvisited.
    result = CacheSimulator(capacity=3).run(trace + [request("e")], SievePolicy())
    assert sorted(result["final_cache"]) == ["a", "d", "e"]


def test_s3fifo_protects_repeated_questions_from_one_hit_wonders():
    hot = [f"h{i}" for i in range(9)]
    trace = [request(k) for k in hot] * 2
    trace += [request(f"once{i}") for i in range(50)]
    s3 = CacheSimulator(capacity=10).run(trace, S3FIFOPolicy())
    lru = CacheSimulator(capacity=10).run(trace, LRUPolicy())
    assert set(hot) <= set(s3["final_cache"])
    assert not set(hot) & set(lru["final_cache"])


def test_references_beat_lru_on_the_chatbot_workload():
    trace = generate_chatbot_workload(4000, seed=9, drift=True)
    simulator = CacheSimulator(capacity=50)
    lru = simulator.run(trace, LRUPolicy())
    belady = simulator.run(trace, BeladyPolicy(trace))
    cost_belady = simulator.run(trace, CostBeladyPolicy(trace))
    assert belady["hit_ratio"] >= lru["hit_ratio"]
    assert cost_belady["total_miss_cost_ms"] < lru["total_miss_cost_ms"]