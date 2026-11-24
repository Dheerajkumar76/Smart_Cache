"""Two experiments that show HOW the policies behave over time.

1. Drift timeline: one drifting chatbot trace (the popular topics change
   twice). Records, for every request, whether each policy hit or missed and
   what the miss cost. Shows how quickly each policy adapts to new topics.

2. Safety guard: a deliberately bad policy (evicts the most expensive
   answer first), the same bad policy wrapped in the GuardedPolicy, and LRU.
   Records cumulative AI time and the moments the guard switches.

Outputs (used by scripts/make_figures.py and scripts/results_summary.py):
  artifacts/drift_timeline.csv
  artifacts/guard_timeline.csv
  artifacts/guard_events.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.policies.guarded import GuardedPolicy
from smartcache.policies.learned_value import LearnedValuePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy


class EvictMostExpensive:
    """Deliberately bad policy used to test the safety guard."""

    name = "EvictMostExpensive"

    def choose_victim(self, cache, incoming, timestamp, context=None):
        return max(cache, key=lambda entry: entry.retrieval_cost_ms).key


class Recorder:
    """Wraps a policy and records the miss cost of every request."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.name = policy.name
        self.miss_cost: list[float] = []

    def observe_request(self, key: str, cost: float, hit: bool, timestamp: int) -> None:
        self.miss_cost.append(0.0 if hit else cost)
        inner = getattr(self.policy, "observe_request", None)
        if inner is not None:
            inner(key, cost, hit, timestamp)

    def choose_victim(self, cache, incoming, timestamp, context=None):
        return self.policy.choose_victim(cache, incoming, timestamp, context)


def write_columns(path: Path, columns: dict[str, list[float]]) -> None:
    names = list(columns)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["request", *names])
        for i, row in enumerate(zip(*(columns[n] for n in names)), start=1):
            writer.writerow([i, *row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift and safety-guard timelines")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--requests", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--model", default="artifacts/learned_value.npz")
    args = parser.parse_args()

    out = Path("artifacts")
    simulator = CacheSimulator(capacity=args.capacity)

    # 1. Drift timeline
    trace = generate_chatbot_workload(args.requests, seed=args.seed, drift=True)
    recorders = [
        Recorder(LRUPolicy()),
        Recorder(LFUPolicy()),
        Recorder(S3FIFOPolicy()),
        Recorder(LearnedValuePolicy(args.model)),
    ]
    for recorder in recorders:
        simulator.run(trace, recorder)
        print(f"drift  {recorder.name:<14} AI time on misses: {sum(recorder.miss_cost) / 1000:,.0f} s")
    write_columns(out / "drift_timeline.csv", {r.name: r.miss_cost for r in recorders})

    # 2. Safety guard
    trace = generate_chatbot_workload(args.requests, seed=args.seed + 1)
    guard = GuardedPolicy(EvictMostExpensive(), capacity=args.capacity)
    recorders = [
        Recorder(LRUPolicy()),
        Recorder(EvictMostExpensive()),
        Recorder(guard),
    ]
    recorders[2].name = "Guarded(EvictMostExpensive)"
    for recorder in recorders:
        simulator.run(trace, recorder)
        print(f"guard  {recorder.name:<28} AI time on misses: {sum(recorder.miss_cost) / 1000:,.0f} s")
    write_columns(out / "guard_timeline.csv", {r.name: r.miss_cost for r in recorders})
    (out / "guard_events.json").write_text(json.dumps(guard.events, indent=2))
    print(f"guard  events: {[(e['event'], e['timestamp']) for e in guard.events]}")
    print("Saved artifacts/drift_timeline.csv, guard_timeline.csv, guard_events.json")


if __name__ == "__main__":
    main()