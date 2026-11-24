"""Synthetic request trace for an AI public-service chatbot answer cache.

Scenario: a government help chatbot (farmers, students, patients) sends each
user question to a large language model (LLM). Identical questions can be
answered from a cache instead of calling the LLM again.

* Each unique question has a fixed answer length (tokens). The miss cost is
  the LLM's response time for that answer:
      cost_ms = time_to_first_token_ms + tokens * ms_per_token
* Popularity follows a Zipf law: a few questions are asked very often and
  most are rare, as in real query logs.
* Popularity is independent of answer length, so "keep the expensive
  answers" is no longer enough: a good policy must weigh cost AND demand.
* With drift, the request stream is split into phases (for example exam
  season, then sowing season). Each phase promotes a different group of
  "hot topic" questions to the top of the popularity ranking.

All defaults are illustrative assumptions, not measurements of a real system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatbotWorkloadConfig:
    questions: int = 5000
    zipf_exponent: float = 0.9
    mean_answer_tokens: float = 250.0
    min_answer_tokens: int = 40
    max_answer_tokens: int = 600
    time_to_first_token_ms: float = 300.0
    ms_per_token: float = 12.0
    bytes_per_token: int = 4
    response_overhead_bytes: int = 200
    phases: int = 3
    hot_topic_fraction: float = 0.05


def _answer_tokens(rng: random.Random, config: ChatbotWorkloadConfig) -> int:
    # Log-normal lengths: most answers are short, a few are long.
    sigma = 0.6
    mu = math.log(config.mean_answer_tokens) - sigma**2 / 2
    tokens = int(rng.lognormvariate(mu, sigma))
    return max(config.min_answer_tokens, min(config.max_answer_tokens, tokens))


def build_catalog(
    config: ChatbotWorkloadConfig, seed: int
) -> list[dict[str, float | int | str]]:
    """Create the question catalogue: key, LLM cost (ms) and answer size."""
    rng = random.Random(seed)
    catalog = []
    for index in range(config.questions):
        tokens = _answer_tokens(rng, config)
        catalog.append(
            {
                "key": f"q{index:05d}",
                "cost": round(
                    config.time_to_first_token_ms + tokens * config.ms_per_token, 1
                ),
                "size": tokens * config.bytes_per_token
                + config.response_overhead_bytes,
                "tokens": tokens,
            }
        )
    return catalog


def _zipf_weights(count: int, exponent: float) -> list[float]:
    return [1.0 / (rank**exponent) for rank in range(1, count + 1)]


def generate_chatbot_workload(
    number_of_requests: int = 20000,
    seed: int = 42,
    drift: bool = False,
    config: ChatbotWorkloadConfig | None = None,
) -> list[dict]:
    """Return a list of requests: {"key", "cost", "size"} per request."""
    if number_of_requests < 1:
        raise ValueError("number_of_requests must be positive")
    config = config or ChatbotWorkloadConfig()
    rng = random.Random(seed)
    catalog = build_catalog(config, seed)
    weights = _zipf_weights(len(catalog), config.zipf_exponent)

    # Base popularity ranking: a random order of questions.
    base_order = list(range(len(catalog)))
    rng.shuffle(base_order)

    phase_count = config.phases if drift else 1
    hot_size = max(1, int(len(catalog) * config.hot_topic_fraction))
    requests: list[dict] = []
    for phase in range(phase_count):
        start = number_of_requests * phase // phase_count
        end = number_of_requests * (phase + 1) // phase_count
        if phase == 0:
            order = base_order
        else:
            # A new hot topic: a random group jumps to the top of the ranking.
            hot = rng.sample(base_order, hot_size)
            hot_set = set(hot)
            order = hot + [index for index in base_order if index not in hot_set]
        chosen = rng.choices(order, weights=weights, k=end - start)
        requests.extend(
            {
                "key": catalog[index]["key"],
                "cost": catalog[index]["cost"],
                "size": catalog[index]["size"],
            }
            for index in chosen
        )
    return requests