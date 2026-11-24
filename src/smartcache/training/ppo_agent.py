from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import AdamW


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature(state)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.01,
        reward_scale_ms: float = 250.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.reward_scale_ms = reward_scale_ms
        self.device = torch.device(device)
        self.net = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = AdamW(self.net.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, state, epsilon: float = 0.0, action_mask=None, deterministic: bool = False) -> int:
        if action_mask is None:
            valid = np.ones(self.action_dim, dtype=np.bool_)
        else:
            valid = np.asarray(action_mask, dtype=np.bool_)
        valid_actions = np.flatnonzero(valid)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits, _ = self.net(state_tensor)
        logits = logits.squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(~valid_tensor, -1e9)

        if deterministic:
            return int(logits.argmax().item())

        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item())

    def update(self, rollouts: list[dict[str, Any]], epochs: int = 4, batch_size: int = 64) -> float:
        if not rollouts:
            return 0.0

        states = torch.tensor(np.array([r["state"] for r in rollouts]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([r["action"] for r in rollouts], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([r["log_prob"] for r in rollouts], dtype=torch.float32, device=self.device)
        returns = torch.tensor([r["return"] for r in rollouts], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([r["advantage"] for r in rollouts], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([r["mask"] for r in rollouts]), dtype=torch.bool, device=self.device)

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n_batches = 0
        indices = np.arange(len(rollouts))

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(rollouts), batch_size):
                b_idx = indices[start : start + batch_size]
                b_states = states[b_idx]
                b_actions = actions[b_idx]
                b_old_lp = old_log_probs[b_idx]
                b_returns = returns[b_idx]
                b_adv = advantages[b_idx]
                b_masks = masks[b_idx]

                logits, values = self.net(b_states)
                logits = logits.masked_fill(~b_masks, -1e9)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * nn.functional.mse_loss(values.squeeze(-1), b_returns)

                loss = policy_loss + value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    def save_checkpoint(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "net_state_dict": self.net.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple[PPOAgent, dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            device=device,
        )
        agent.net.load_state_dict(checkpoint["net_state_dict"])
        return agent, checkpoint.get("metadata", {})

