"""Benchmark eviction policies on the AI-chatbot answer-cache workload.

Every (seed, policy) run uses a brand-new policy instance. Results are shown
against LRU (0%) and the best future-knowing reference (100%); the exact
optimum cannot be computed for thousands of objects.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.policies.adaptive_gdsf import AdaptiveGDSFPolicy
from smartcache.policies.contextual_bandit import LinUCBPolicy
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.evict_cheapest import EvictCheapestPolicy
from smartcache.policies.fifo import FIFOPolicy
from smartcache.policies.gdsf import GDSFPolicy
from smartcache.policies.learned_value import LearnedValuePolicy, NumpyMLP
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.oracle import BeladyPolicy, CostBeladyPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy
from smartcache.policies.sieve import SievePolicy

PolicyFactory = Callable[[list[dict[str, Any]]], Any]

# Future-knowing yardsticks: shown for reference, never deployable.
REFERENCES = ("Belady", "CostBelady")
LEARNED_MODEL = Path("artifacts/learned_value.npz")


def build_factories(include_linucb: bool) -> dict[str, PolicyFactory]:
    factories: dict[str, PolicyFactory] = {
        "LRU": lambda trace: LRUPolicy(),
        "LFU": lambda trace: LFUPolicy(),
        "FIFO": lambda trace: FIFOPolicy(),
        "SIEVE": lambda trace: SievePolicy(),
        "S3-FIFO": lambda trace: S3FIFOPolicy(),
        "CostAware": lambda trace: CostAwarePolicy(recency_tau=20.0),
        "EvictCheapest": lambda trace: EvictCheapestPolicy(),
        "GDSF": lambda trace: GDSFPolicy(),
        "AdaptiveGDSF": lambda trace: AdaptiveGDSFPolicy(),
    }
    if include_linucb:
        factories["LinUCB_Bandit"] = lambda trace: LinUCBPolicy(alpha=0.5)

    if LEARNED_MODEL.is_file():
        # Load the weights once; each run still gets a fresh policy (and a
        # fresh feature history) built around the same trained model.
        model, _ = NumpyMLP.load(LEARNED_MODEL)
        factories["LearnedValue"] = lambda trace: LearnedValuePolicy(model)
        
    factories["Belady"] = lambda trace: BeladyPolicy(trace)
    factories["CostBelady"] = lambda trace: CostBeladyPolicy(trace)
    return factories


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot answer-cache benchmark")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument(
        "--skip-linucb",
        action="store_true",
        help="LinUCB is slow at large capacities; skip it for quick runs",
    )
    parser.add_argument("--output", default="artifacts/chatbot_benchmark.csv")
    args = parser.parse_args()

    simulator = CacheSimulator(capacity=args.capacity)
    factories = build_factories(include_linucb=not args.skip_linucb)
    names = list(factories)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    print(
        f"Chatbot workload | capacity {args.capacity} answers | "
        f"{args.requests} questions per trace | seeds {seeds[0]}-{seeds[-1]}"
    )

    records: list[dict[str, Any]] = []
    for scenario, drift in (("stationary", False), ("drifting", True)):
        costs: dict[str, list[float]] = {n: [] for n in names}
        hits: dict[str, list[float]] = {n: [] for n in names}
        per_decision_us: dict[str, list[float]] = {n: [] for n in names}
        for seed in seeds:
            trace = generate_chatbot_workload(args.requests, seed=seed, drift=drift)
            for n in names:
                r = simulator.run(trace, factories[n](trace))
                us = float(r["policy_time_ms"]) * 1000 / max(1, int(r["evictions"]))
                costs[n].append(float(r["total_miss_cost_ms"]))
                hits[n].append(float(r["hit_ratio"]))
                per_decision_us[n].append(us)
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": n,
                        "capacity": args.capacity,
                        "requests": args.requests,
                        "hits": r["hits"],
                        "misses": r["misses"],
                        "hit_ratio": r["hit_ratio"],
                        "miss_cost_ms": r["total_miss_cost_ms"],
                        "per_decision_us": us,
                    }
                )

        lru = mean(costs["LRU"])
        best_ref = min(mean(costs[n]) for n in REFERENCES)
        print(f"\n=== {scenario.upper()} (mean ± std over {len(seeds)} seeds) ===")
        print(
            f"{'Policy':<15} {'LLM time on misses (s)':>24} {'Hit %':>7} "
            f"{'µs/decision':>12} {'vs LRU':>8} {'Gap closed':>11}"
        )
        for n in sorted(names, key=lambda n: mean(costs[n])):
            m = mean(costs[n])
            s = stdev(costs[n]) if len(costs[n]) > 1 else 0.0
            vs_lru = (lru - m) / lru * 100
            # 0% = LRU, 100% = best future-knowing reference.
            gap = (lru - m) / (lru - best_ref) * 100 if lru != best_ref else 0.0
            label = n + (" (ref)" if n in REFERENCES else "")
            print(
                f"{label:<15} {m / 1000:>14.1f} ± {s / 1000:>7.1f} "
                f"{mean(hits[n]) * 100:>6.2f}% {mean(per_decision_us[n]):>12.1f} "
                f"{vs_lru:>+7.2f}% {gap:>10.1f}%"
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