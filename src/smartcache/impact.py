"""Turn cache results into human terms: rupees, energy, CO2 and waiting time.

Every cache miss in the chatbot scenario is one call to a large language
model (LLM). Its cost in the simulation is the LLM response time:

    cost_ms = time_to_first_token_ms + output_tokens * ms_per_token

so the number of generated tokens can be recovered exactly from the total
miss cost and the number of misses. Money, energy and CO2 are then computed
per generated token.

The default figures are ASSUMPTIONS for illustration. Edit them for a real
deployment:

* price_inr_per_1k_output_tokens: what your LLM provider charges. Prices
  differ by more than 10x between providers and models.
* energy_wh_per_1k_output_tokens: Epoch AI (2025) estimates about 0.3 Wh for
  a typical ChatGPT (GPT-4o) query. Assuming a typical answer of about 500
  tokens gives about 0.6 Wh per 1,000 output tokens.
* grid_kg_co2_per_kwh: India's grid weighted-average emission factor is
  0.727 kg CO2 per kWh (Central Electricity Authority, CO2 Baseline Database
  version 20.0). This assumes the model is served from the Indian grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImpactAssumptions:
    price_inr_per_1k_output_tokens: float = 0.50
    energy_wh_per_1k_output_tokens: float = 0.6
    grid_kg_co2_per_kwh: float = 0.727
    time_to_first_token_ms: float = 300.0
    ms_per_token: float = 12.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def llm_impact(
    llm_time_ms: float,
    llm_calls: float,
    assumptions: ImpactAssumptions = ImpactAssumptions(),
) -> dict[str, float]:
    """Impact of `llm_calls` LLM calls that together took `llm_time_ms`."""
    tokens = max(
        0.0,
        (llm_time_ms - llm_calls * assumptions.time_to_first_token_ms)
        / assumptions.ms_per_token,
    )
    kwh = tokens / 1000 * assumptions.energy_wh_per_1k_output_tokens / 1000
    return {
        "llm_calls": llm_calls,
        "output_tokens": tokens,
        "cost_inr": tokens / 1000 * assumptions.price_inr_per_1k_output_tokens,
        "energy_kwh": kwh,
        "co2_kg": kwh * assumptions.grid_kg_co2_per_kwh,
        "user_wait_hours": llm_time_ms / 3_600_000,
    }


def scale(values: dict[str, float], factor: float) -> dict[str, float]:
    return {key: value * factor for key, value in values.items()}


def savings(baseline: dict[str, float], policy: dict[str, float]) -> dict[str, float]:
    """How much the policy saves compared with the baseline (positive = saved)."""
    return {key: baseline[key] - policy[key] for key in baseline}