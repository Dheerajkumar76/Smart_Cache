"""Train the LearnedValue eviction policy on chatbot traces.

Approximate policy iteration:
  round 0: collect data while LRU and S3-FIFO make the decisions;
  round k: collect more data while the round k-1 learned policy decides
           (so the model also learns from the situations it creates itself);
  each round: fit the value network on all data collected so far and keep
              the model with the lowest miss cost on validation traces.

Seeds used: training 1000+, validation 2000+. Benchmarks use 30000+, so test
traces are never seen during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn

from smartcache.core.cache import CacheSimulator
from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.policies.learned_value import (
    FEATURE_COUNT,
    LearnedValuePolicy,
    NumpyMLP,
    TrainingCollector,
)
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.s3fifo import S3FIFOPolicy


def make_traces(seed_start: int, count: int, requests: int) -> list[list[dict]]:
    # Alternate stationary and drifting traces so both are learned.
    return [
        generate_chatbot_workload(requests, seed=seed_start + i, drift=i % 2 == 1)
        for i in range(count)
    ]


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    hidden: int,
    epochs: int,
    seed: int,
) -> NumpyMLP:
    torch.manual_seed(seed)
    network = nn.Sequential(
        nn.Linear(FEATURE_COUNT, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=2e-3)
    # Binary cross-entropy works well for targets between 0 and 1.
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels).unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), 1024):
            batch = order[start : start + 1024]
            loss = loss_fn(network(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        print(f"    epoch {epoch + 1}/{epochs} loss={total / len(x):.4f}")
    linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
    return NumpyMLP(
        [layer.weight.detach().numpy().T.copy() for layer in linear_layers],
        [layer.bias.detach().numpy().copy() for layer in linear_layers],
    )


def evaluate(model: NumpyMLP, traces: list[list[dict]], capacity: int) -> float:
    simulator = CacheSimulator(capacity=capacity)
    return mean(
        simulator.run(trace, LearnedValuePolicy(model))["total_miss_cost_ms"]
        for trace in traces
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LearnedValue policy")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--train-traces", type=int, default=6)
    parser.add_argument("--val-traces", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/learned_value.npz")
    args = parser.parse_args()

    train = make_traces(1000, args.train_traces, args.requests)
    validation = make_traces(2000, args.val_traces, args.requests)
    simulator = CacheSimulator(capacity=args.capacity)

    lru_cost = mean(
        simulator.run(trace, LRUPolicy())["total_miss_cost_ms"] for trace in validation
    )
    s3_cost = mean(
        simulator.run(trace, S3FIFOPolicy())["total_miss_cost_ms"] for trace in validation
    )
    print(f"Validation baselines: LRU {lru_cost / 1000:.1f} s, S3-FIFO {s3_cost / 1000:.1f} s")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    best_model: NumpyMLP | None = None
    best_cost = float("inf")

    for round_index in range(args.rounds):
        print(f"\nRound {round_index}: collecting data")
        for i, trace in enumerate(train):
            if best_model is None:
                behaviour = LRUPolicy() if i % 2 == 0 else S3FIFOPolicy()
            else:
                behaviour = LearnedValuePolicy(best_model)
            collector = TrainingCollector(
                behaviour, trace, gamma=args.gamma, seed=args.seed + 100 * round_index + i
            )
            simulator.run(trace, collector)
            x, y = collector.dataset()
            all_x.append(x)
            all_y.append(y)
        features = np.concatenate(all_x)
        labels = np.concatenate(all_y)
        print(f"  {len(features):,} training examples; fitting value network")
        model = fit(features, labels, args.hidden, args.epochs, args.seed + round_index)
        cost = evaluate(model, validation, args.capacity)
        print(
            f"  validation miss cost {cost / 1000:.1f} s "
            f"(LRU {lru_cost / 1000:.1f}, S3-FIFO {s3_cost / 1000:.1f})"
        )
        if cost < best_cost:
            best_cost, best_model = cost, model

    assert best_model is not None
    best_model.save(
        args.output,
        {
            "capacity": args.capacity,
            "gamma": args.gamma,
            "hidden": args.hidden,
            "rounds": args.rounds,
            "validation_miss_cost_ms": round(best_cost, 1),
        },
    )
    print(f"\nSaved best model ({best_cost / 1000:.1f} s on validation) to {args.output}")


if __name__ == "__main__":
    main()