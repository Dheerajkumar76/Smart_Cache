"""Fast side-by-side comparison for live demos (about 10-60 seconds)."""

from __future__ import annotations

import argparse
from statistics import mean

from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import (
    ChatbotWorkloadConfig,
    generate_chatbot_workload,
)
from smartcache.policies.learned_value import LearnedValuePolicy, NumpyMLP
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.oracle import CostBeladyPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick LRU vs S3-FIFO vs LearnedValue")
    parser.add_argument("--model", default="artifacts/learned_value.npz")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=2, help="number of test traces")
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--drift", action="store_true", help="use drifting traffic")
    parser.add_argument("--zipf", type=float, default=0.9, help="popularity skew")
    parser.add_argument("--questions", type=int, default=5000)
    parser.add_argument("--ms-per-token", type=float, default=12.0)
    args = parser.parse_args()

    config = ChatbotWorkloadConfig(
        questions=args.questions,
        zipf_exponent=args.zipf,
        ms_per_token=args.ms_per_token,
    )
    model, _ = NumpyMLP.load(args.model)
    simulator = CacheSimulator(capacity=args.capacity)
    names = ["LRU", "S3-FIFO", "LearnedValue", "CostBelady (ref)"]
    costs = {n: [] for n in names}
    hits = {n: [] for n in names}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        trace = generate_chatbot_workload(args.requests, seed=seed, drift=args.drift, config=config)
        policies = [LRUPolicy(), S3FIFOPolicy(), LearnedValuePolicy(model), CostBeladyPolicy(trace)]
        for name, policy in zip(names, policies):
            result = simulator.run(trace, policy)
            costs[name].append(result["total_miss_cost_ms"] / 1000)
            hits[name].append(result["hit_ratio"] * 100)

    lru = mean(costs["LRU"])
    ref = mean(costs["CostBelady (ref)"])
    print(f"capacity={args.capacity} drift={args.drift} zipf={args.zipf} "
          f"questions={args.questions} model={args.model} traces={args.seeds}")
    print(f"{'Policy':<18}{'AI time (s)':>12}{'Hit %':>8}{'vs LRU':>9}{'Gap closed':>12}")
    for name in names:
        m = mean(costs[name])
        gap = (lru - m) / (lru - ref) * 100 if lru != ref else 0.0
        print(f"{name:<18}{m:>12,.0f}{mean(hits[name]):>7.1f}%{(lru - m) / lru * 100:>+8.1f}%{gap:>11.1f}%")


if __name__ == "__main__":
    main()