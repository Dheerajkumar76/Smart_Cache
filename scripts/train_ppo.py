from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import numpy as np
import torch
from torch.distributions import Categorical

from smartcache.environment.cache_env import CacheEnv
from smartcache.environment.workload import generate_workload
from smartcache.training.ppo_agent import PPOAgent


def needs_eviction(env: CacheEnv) -> bool:
    request = env.requests[env.position]
    return str(request["key"]) not in env.cache and None not in env.cache


def evaluate(
    agent: PPOAgent,
    capacity: int,
    requests_per_trace: int,
    seeds: list[int],
    frequency_window: int,
    recency_window: int,
) -> dict[str, float]:
    costs: list[float] = []
    hit_ratios: list[float] = []
    for seed in seeds:
        requests = generate_workload(requests_per_trace, seed=seed, drift=True)
        env = CacheEnv(
            requests, capacity=capacity, frequency_window=frequency_window
        )
        env.recency_window = recency_window
        state, _ = env.reset(seed=seed)
        misses = 0
        miss_cost = 0.0
        steps = 0
        while True:
            if needs_eviction(env):
                action = agent.act(
                    state,
                    action_mask=env.action_mask(),
                    deterministic=True,
                )
            else:
                action = env.noop_action
            state, reward, terminated, truncated, info = env.step(action)
            misses += int(not info["hit"])
            miss_cost += info["miss_cost_ms"]
            steps += 1
            if terminated or truncated:
                break
        costs.append(miss_cost)
        hit_ratios.append(1.0 - misses / steps)
    return {
        "miss_cost_ms": mean(costs),
        "hit_ratio": mean(hit_ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--requests", type=int, default=800)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frequency_window = 100
    recency_window = 1000
    state_dim = args.capacity * 5 + 3
    initial_requests = generate_workload(1, seed=args.seed, drift=True)
    initial_env = CacheEnv(
        initial_requests,
        capacity=args.capacity,
        frequency_window=frequency_window,
    )
    agent = PPOAgent(
        state_dim,
        initial_env.action_space.n,
        device=args.device,
    )
    best_cost = float("inf")
    artifacts = Path("artifacts")

    for episode in range(args.episodes):
        requests = generate_workload(
            args.requests, seed=args.seed + episode, drift=True
        )
        env = CacheEnv(
            requests, capacity=args.capacity, frequency_window=frequency_window
        )
        env.recency_window = recency_window
        state, _ = env.reset(seed=args.seed + episode)
        rollouts: list[dict[str, Any]] = []
        total_cost = 0.0

        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []

        while True:
            current_mask = env.action_mask()
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                logits, val = agent.net(state_t)
                valid_t = torch.tensor(current_mask, dtype=torch.bool, device=agent.device)
                logits = logits.squeeze(0).masked_fill(~valid_t, -1e9)
                dist = Categorical(logits=logits)
                if current_mask[: env.capacity].any():
                    action = dist.sample()
                else:
                    action = torch.tensor(env.noop_action, device=agent.device)
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated

            states.append(state)
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            # Reward scaled
            rewards.append(float(reward) / agent.reward_scale_ms)
            values.append(float(val.item()))
            masks.append(current_mask)

            state = next_state
            total_cost += info["miss_cost_ms"]
            if done:
                break

        # Compute Generalized Advantage Estimation / Discounted Returns
        returns = []
        discounted_sum = 0.0
        for r in reversed(rewards):
            discounted_sum = r + agent.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        advantages = [ret - v for ret, v in zip(returns, values)]

        for s, a, lp, ret, adv, m in zip(states, actions, log_probs, returns, advantages, masks):
            rollouts.append({
                "state": s,
                "action": a,
                "log_prob": lp,
                "return": ret,
                "advantage": adv,
                "mask": m
            })

        loss = agent.update(rollouts, epochs=4, batch_size=64)

        metadata = {
            "capacity": args.capacity,
            "frequency_window": frequency_window,
            "recency_window": recency_window,
            "episode": episode + 1,
        }
        agent.save_checkpoint(artifacts / "ppo_latest.pt", metadata)
        validation = evaluate(
            agent,
            args.capacity,
            args.requests,
            [args.seed + 10_000, args.seed + 20_000],
            frequency_window,
            recency_window,
        )
        if validation["miss_cost_ms"] < best_cost:
            best_cost = validation["miss_cost_ms"]
            agent.save_checkpoint(artifacts / "ppo_best.pt", metadata | validation)
        if (episode + 1) % 10 == 0:
            print(
                f"PPO episode={episode + 1} cost_ms={total_cost:.1f} "
                f"loss={loss:.5f} "
                f"validation_cost_ms={validation['miss_cost_ms']:.1f}"
            )

    print(f"\nPPO training complete! Best validation cost: {best_cost:.1f} ms")


if __name__ == "__main__":
    main()

