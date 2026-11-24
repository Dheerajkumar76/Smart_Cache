from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.feature(state)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        return val + (adv - adv.mean(dim=-1, keepdim=True))


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 250,
        reward_scale_ms: float = 250.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.reward_scale_ms = reward_scale_ms
        self.device = torch.device(device)
        self.online = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.online_net = self.online
        self.target_net = self.target
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = AdamW(self.online.parameters(), lr=learning_rate)
        self.update_count = 0

    @torch.no_grad()
    def act(self, state, epsilon: float = 0.0, action_mask=None) -> int:
        if action_mask is None:
            valid = np.ones(self.action_dim, dtype=np.bool_)
        else:
            valid = np.asarray(action_mask, dtype=np.bool_)
        valid_actions = np.flatnonzero(valid)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.online_net(state_tensor).squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        q_values = q_values.masked_fill(~valid_tensor, float("-inf"))
        return int(q_values.argmax().item())

    def learn(self, batch, reward_scale_ms: float = 250.0) -> float:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_action_masks,
        ) = batch
        current = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            next_online_q = self.online_net(next_states)
            next_online_q = next_online_q.masked_fill(
                ~next_action_masks, float("-inf")
            )
            best_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states).gather(
                1, best_actions
            ).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * next_target_q

        loss = nn.functional.smooth_l1_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save_checkpoint(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "gamma": self.gamma,
                "reward_scale_ms": self.reward_scale_ms,
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> tuple[DuelingDQNAgent, dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            gamma=checkpoint["gamma"],
            reward_scale_ms=checkpoint["reward_scale_ms"],
            device=device,
        )
        agent.online.load_state_dict(checkpoint["online_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        return agent, checkpoint.get("metadata", {})

