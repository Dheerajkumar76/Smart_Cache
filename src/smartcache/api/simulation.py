from __future__ import annotations

from collections import Counter
from time import perf_counter_ns
from typing import Any

from smartcache.core.interfaces import CacheEntry
from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.environment.workload import generate_workload
from smartcache.policies.adaptive_gdsf import AdaptiveGDSFPolicy
from smartcache.policies.contextual_bandit import LinUCBPolicy
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_policy import DQNPolicy
from smartcache.policies.dueling_dqn_policy import DuelingDQNPolicy
from smartcache.policies.evict_cheapest import EvictCheapestPolicy
from smartcache.policies.fifo import FIFOPolicy
from smartcache.policies.gdsf import GDSFPolicy
from smartcache.policies.learned_value import LearnedValuePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.ppo_policy import PPOPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy
from smartcache.policies.sieve import SievePolicy

# Display family for each policy name shown in the Simulation Explorer.
POLICY_FAMILIES = {
    "LRU": "Classical",
    "LFU": "Classical",
    "FIFO": "Classical",
    "SIEVE": "Modern baseline",
    "S3-FIFO": "Modern baseline",
    "CostAware": "Cost-aware heuristic",
    "EvictCheapest": "Cost-aware heuristic",
    "GDSF": "Cost-aware heuristic",
    "AdaptiveGDSF": "Cost-aware heuristic",
    "LinUCB_Bandit": "Contextual bandit",
    "DQN": "Deep RL",
    "DuelingDQN": "Deep RL",
    "PPO": "Deep RL",
    "LearnedValue": "Learned (ours)",
}

# Largest replay the explorer accepts per workload (every step is returned).
MAX_REQUESTS = {"demo": 200, "chatbot": 1000}

_CHECKPOINT_ERRORS = (FileNotFoundError, ValueError, KeyError, RuntimeError)


def _policy_set(
    checkpoint: str | None,
    capacity: int,
    dueling_checkpoint: str | None = None,
    ppo_checkpoint: str | None = None,
    learned_model: str | None = None,
) -> list[Any]:
    # Fresh instances on every call: GDSF, AdaptiveGDSF, and LinUCB keep
    # internal state that must not leak between replays.
    policies: list[Any] = [
        LRUPolicy(),
        LFUPolicy(),
        FIFOPolicy(),
        SievePolicy(),
        S3FIFOPolicy(),
        CostAwarePolicy(),
        EvictCheapestPolicy(),
        GDSFPolicy(),
        AdaptiveGDSFPolicy(),
        LinUCBPolicy(),
    ]
    # The learned value model works for any cache size.
    if learned_model:
        try:
            policies.append(LearnedValuePolicy(learned_model))
        except (FileNotFoundError, ValueError, KeyError, OSError):
            pass
    # Learned fixed-slot models are only valid for the capacity they were
    # trained on; incompatible or missing checkpoints are skipped.
    for policy_class, path in (
        (DQNPolicy, checkpoint),
        (DuelingDQNPolicy, dueling_checkpoint),
        (PPOPolicy, ppo_checkpoint),
    ):
        if not path:
            continue
        try:
            policies.append(policy_class(path, capacity=capacity))
        except _CHECKPOINT_ERRORS:
            pass
    return policies


def _run_policy(requests: list[dict[str, Any]], policy: Any, capacity: int):
    cache: list[CacheEntry] = []
    frequency: Counter[str] = Counter()
    last_access: dict[str, int] = {}
    history: list[str] = []
    catalog = {
        request["key"]: {
            "cost": float(request["cost"]),
            "size": float(request["size"]),
        }
        for request in requests
    }
    trace = []
    hits = 0
    total_cost = 0.0
    total_policy_time = 0.0

    for timestamp, request in enumerate(requests, start=1):
        key = request["key"]
        entry = next((item for item in cache if item.key == key), None)
        evicted = None
        policy_time = 0.0

        if entry is not None:
            result = "HIT"
            hits += 1
            entry.frequency += 1
            entry.last_access = timestamp
            explanation = "The object was already resident, so the backend was skipped."
        else:
            result = "MISS"
            total_cost += float(request["cost"])
            if len(cache) >= capacity:
                context = {
                    "catalog": catalog,
                    "last_access": last_access,
                    "frequency_counts": frequency,
                    "frequency_window": 100,
                    "recency_window": 1000,
                    "max_cost": max(item["cost"] for item in catalog.values()),
                    "max_size": max(item["size"] for item in catalog.values()),
                }
                started = perf_counter_ns()
                evicted = policy.choose_victim(
                    cache, request, timestamp, context=context
                )
                policy_time = (perf_counter_ns() - started) / 1_000_000
                cache.remove(next(item for item in cache if item.key == evicted))
                explanation = (
                    f"{policy.name} evicted {evicted} to admit {key}; "
                    f"the miss cost was {request['cost']:.1f} ms."
                )
            else:
                explanation = (
                    f"The cache had a free slot, so {key} was admitted without eviction."
                )
            cache.append(
                CacheEntry(
                    key=key,
                    retrieval_cost_ms=request["cost"],
                    size_bytes=request["size"],
                    frequency=1,
                    last_access=timestamp,
                )
            )

        last_access[key] = timestamp
        frequency[key] += 1
        history.append(key)
        if len(history) > 100:
            expired = history.pop(0)
            frequency[expired] -= 1
            if frequency[expired] <= 0:
                del frequency[expired]
        total_policy_time += policy_time
        trace.append(
            {
                "request_index": timestamp - 1,
                "request": request,
                "result": result,
                "after_cache": [item.key for item in cache],
                "evicted": evicted,
                "reward": 0.0 if result == "HIT" else -float(request["cost"]),
                "policy_time_ms": policy_time,
                "explanation": explanation,
                "cumulative_miss_cost_ms": total_cost,
            }
        )

    return {
        "policy": policy.name,
        "family": POLICY_FAMILIES.get(policy.name, "Other"),
        "steps": trace,
        "summary": {
            "requests": len(requests),
            "hits": hits,
            "misses": len(requests) - hits,
            "hit_ratio": hits / len(requests),
            "total_miss_cost_ms": total_cost,
            "policy_time_ms": total_policy_time,
            "net_cost_ms": total_cost + total_policy_time,
        },
    }


def build_simulation(
    *,
    scenario: str,
    seed: int,
    requests_count: int,
    capacity: int,
    checkpoint: str | None,
    dueling_checkpoint: str | None = None,
    ppo_checkpoint: str | None = None,
    learned_model: str | None = None,
    workload: str = "demo",
) -> dict[str, Any]:
    if scenario not in {"stationary", "drifting"}:
        raise ValueError("scenario must be stationary or drifting")
    if workload not in MAX_REQUESTS:
        raise ValueError("workload must be demo or chatbot")
    if requests_count < 1 or capacity < 1:
        raise ValueError("requests_count and capacity must be positive")
    if requests_count > MAX_REQUESTS[workload]:
        raise ValueError(
            f"At most {MAX_REQUESTS[workload]} requests for the {workload} workload"
        )
    generate = generate_chatbot_workload if workload == "chatbot" else generate_workload
    requests = generate(requests_count, seed=seed, drift=scenario == "drifting")
    results = [
        _run_policy(requests, policy, capacity)
        for policy in _policy_set(
            checkpoint, capacity, dueling_checkpoint, ppo_checkpoint, learned_model
        )
    ]
    return {
        "workload": workload,
        "scenario": scenario,
        "seed": seed,
        "capacity": capacity,
        "requests": requests,
        "policies": results,
    }