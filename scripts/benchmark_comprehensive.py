"""Benchmark every fixed-slot policy on the same traces.

Each (seed, policy) run gets a brand-new policy instance, so stateful policies
(GDSF, AdaptiveGDSF, LinUCB) cannot carry learned state between seeds.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from smartcache.core.cache import CacheSimulator
from smartcache.environment.workload import generate_workload
from smartcache.policies.adaptive_gdsf import AdaptiveGDSFPolicy
from smartcache.policies.contextual_bandit import LinUCBPolicy
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_policy import DQNPolicy
from smartcache.policies.dueling_dqn_policy import DuelingDQNPolicy
from smartcache.policies.evict_cheapest import EvictCheapestPolicy
from smartcache.policies.fifo import FIFOPolicy
from smartcache.policies.gdsf import GDSFPolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.oracle import optimal_miss_cost
from smartcache.policies.ppo_policy import PPOPolicy

# A factory receives the request trace and returns a fresh policy.
PolicyFactory = Callable[[list[dict[str, Any]]], Any]

OPTIMUM = "Optimum"


def build_factories(capacity: int, device: str) -> dict[str, PolicyFactory]:
    factories: dict[str, PolicyFactory] = {
        "LRU": lambda trace: LRUPolicy(),
        "LFU": lambda trace: LFUPolicy(),
        "FIFO": lambda trace: FIFOPolicy(),
        "CostAware": lambda trace: CostAwarePolicy(recency_tau=20.0),
        "EvictCheapest": lambda trace: EvictCheapestPolicy(),
        "GDSF": lambda trace: GDSFPolicy(),
        "AdaptiveGDSF": lambda trace: AdaptiveGDSFPolicy(),
        "LinUCB_Bandit": lambda trace: LinUCBPolicy(alpha=0.5),
    }
    learned = {
        "DQN": (DQNPolicy, Path("artifacts/dqn_best.pt")),
        "DuelingDQN": (DuelingDQNPolicy, Path("artifacts/dueling_dqn_best.pt")),
        "PPO": (PPOPolicy, Path("artifacts/ppo_best.pt")),
    }
    for name, (policy_class, path) in learned.items():
        if path.is_file():
            # Learned models are loaded once; their decisions do not change
            # between seeds because they never update during evaluation.
            model = policy_class(checkpoint_path=path, capacity=capacity, device=device)
            factories[name] = lambda trace, model=model: model
    return factories


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-slot multi-policy benchmark")
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", default="artifacts/comprehensive_multi_algo_benchmark.csv"
    )
    args = parser.parse_args()

    simulator = CacheSimulator(
        capacity=args.capacity, frequency_window=100, recency_window=1000
    )
    factories = build_factories(args.capacity, args.device)
    names = list(factories)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    print(f"Capacity {args.capacity} slots | {args.requests} requests | seeds {seeds[0]}-{seeds[-1]}")
    print(f"Policies: {', '.join(names)}")

    records: list[dict[str, Any]] = []
    for scenario, drift in (("stationary", False), ("drifting", True)):
        costs: dict[str, list[float]] = {n: [] for n in [*names, OPTIMUM]}
        hits: dict[str, list[float]] = {n: [] for n in names}
        per_decision_us: dict[str, list[float]] = {n: [] for n in names}
        wins_lru = {n: 0 for n in names}
        wins_ca = {n: 0 for n in names}

        for seed in seeds:
            trace = generate_workload(args.requests, seed=seed, drift=drift)
            results = {n: simulator.run(trace, factories[n](trace)) for n in names}
            costs[OPTIMUM].append(optimal_miss_cost(trace, args.capacity))
            lru_cost = results["LRU"]["total_miss_cost_ms"]
            ca_cost = results["CostAware"]["total_miss_cost_ms"]
            for n, r in results.items():
                cost = float(r["total_miss_cost_ms"])
                decisions = max(1, int(r["evictions"]))
                us = float(r["policy_time_ms"]) * 1000 / decisions
                costs[n].append(cost)
                hits[n].append(float(r["hit_ratio"]))
                per_decision_us[n].append(us)
                wins_lru[n] += cost < lru_cost
                wins_ca[n] += cost < ca_cost
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": n,
                        "requests": args.requests,
                        "hits": r["hits"],
                        "misses": r["misses"],
                        "hit_ratio": r["hit_ratio"],
                        "evictions": r["evictions"],
                        "miss_cost_ms": cost,
                        "policy_time_ms": r["policy_time_ms"],
                        "per_decision_us": us,
                        "net_cost_ms": r["estimated_net_cost_ms"],
                    }
                )

        lru = mean(costs["LRU"])
        ref = mean(costs[OPTIMUM])
        print(f"\n=== {scenario.upper()} (mean ± std over {len(seeds)} seeds) ===")
        print(
            f"{'Policy':<15} {'Miss cost (ms)':>22} {'Hit %':>7} {'µs/decision':>12} "
            f"{'vs LRU':>8} {'Gap closed':>11} {'Wins LRU|CA':>12}"
        )
        print(f"{'Optimum (offline)':<15} {ref:>12.1f} ± {stdev(costs[OPTIMUM]):>7.1f}   (best any policy could do)")
        for n in sorted(names, key=lambda n: mean(costs[n])):
            m = mean(costs[n])
            s = stdev(costs[n]) if len(costs[n]) > 1 else 0.0
            vs_lru = (lru - m) / lru * 100
            # 0% = as good as LRU, 100% = as good as the offline optimum.
            gap = (lru - m) / (lru - ref) * 100 if lru != ref else 0.0
            print(
                f"{n:<15} {m:>12.1f} ± {s:>7.1f} {mean(hits[n]) * 100:>6.2f}% "
                f"{mean(per_decision_us[n]):>12.2f} {vs_lru:>+7.2f}% {gap:>10.1f}% "
                f"{wins_lru[n]:>5}|{wins_ca[n]:<5}"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} rows to {out}")


if __name__ == "__main__":
    main()