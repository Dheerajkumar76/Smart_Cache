from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from smartcache.core.interfaces import CacheEntry


class LinUCBPolicy:
    """
    LinUCB Contextual Bandit Policy for Cache Eviction.
    
    Treats eviction as a contextual bandit selection over resident cache objects.
    Predicts object retention value and evicts the item with minimum estimated utility.
    """
    name = "LinUCB_Bandit"

    def __init__(self, feature_dim: int = 5, alpha: float = 0.5, l2_reg: float = 1.0) -> None:
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.A = np.eye(feature_dim) * l2_reg
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros(feature_dim)
        self.theta = np.zeros(feature_dim)

    def _extract_features(
        self,
        entry: CacheEntry,
        timestamp: int,
        context: Mapping[str, Any] | None,
    ) -> np.ndarray:
        max_cost = float(context["max_cost"]) if context else 250.0
        max_size = float(context["max_size"]) if context else 32768.0
        recency_window = float(context["recency_window"]) if context else 1000.0

        age = max(0, timestamp - entry.last_access)
        recency_val = float(np.exp(-age / recency_window))
        freq_val = float(np.log1p(entry.frequency) / np.log1p(100.0))
        cost_val = float(np.log1p(entry.retrieval_cost_ms) / np.log1p(max_cost))
        size_val = float(entry.size_bytes / max_size)
        interaction = freq_val * cost_val

        return np.array([recency_val, freq_val, cost_val, size_val, interaction], dtype=np.float32)

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")

        self.theta = self.A_inv @ self.b
        best_score = float("inf")
        victim_key = cache[0].key
        victim_features = None
        victim_cost = cache[0].retrieval_cost_ms

        for entry in cache:
            x = self._extract_features(entry, timestamp, context)
            mean_val = float(np.dot(self.theta, x))
            uncertainty = self.alpha * float(np.sqrt(max(0.0, np.dot(x, self.A_inv @ x))))
            # Lower confidence bound for retention value
            score = mean_val - uncertainty

            if score < best_score:
                best_score = score
                victim_key = entry.key
                victim_features = x
                victim_cost = entry.retrieval_cost_ms

        # Online Sherman-Morrison update on eviction
        if victim_features is not None:
            x_col = victim_features.reshape(-1, 1)
            cost_norm = float(np.log1p(victim_cost) / np.log1p(250.0))
            self.A += x_col @ x_col.T
            self.b += cost_norm * victim_features
            denom = float(1.0 + (x_col.T @ self.A_inv @ x_col).item())
            self.A_inv -= (self.A_inv @ x_col @ x_col.T @ self.A_inv) / denom

        return victim_key

