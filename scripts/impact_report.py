"""Impact report: what each cache policy saves per 1 million chatbot questions.

Reads the results of scripts/benchmark_chatbot.py, converts them into LLM
calls, rupees, energy, CO2 and user waiting time, and compares each policy
with having no cache at all and with LRU. Writes artifacts/impact_summary.json
for the dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from smartcache.environment.chatbot_workload import generate_chatbot_workload
from smartcache.impact import ImpactAssumptions, llm_impact, savings, scale

REFERENCES = {"Belady", "CostBelady"}
PER = 1_000_000  # report everything per 1 million questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot cache impact report")
    parser.add_argument("--input", default="artifacts/chatbot_benchmark.csv")
    parser.add_argument("--output", default="artifacts/impact_summary.json")
    parser.add_argument("--price-inr-per-1k-tokens", type=float, default=None)
    args = parser.parse_args()

    assumptions = ImpactAssumptions()
    if args.price_inr_per_1k_tokens is not None:
        assumptions = ImpactAssumptions(
            price_inr_per_1k_output_tokens=args.price_inr_per_1k_tokens
        )

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary: dict[str, object] = {
        "assumptions": assumptions.as_dict(),
        "per_questions": PER,
        "capacity": int(rows[0]["capacity"]),
        "scenarios": {},
    }

    for scenario in ("stationary", "drifting"):
        scenario_rows = [r for r in rows if r["scenario"] == scenario]
        requests = int(scenario_rows[0]["requests"])
        factor = PER / requests
        seeds = sorted({int(r["seed"]) for r in scenario_rows})

        # No cache: every question is sent to the LLM.
        demand = mean(
            sum(item["cost"] for item in generate_chatbot_workload(
                requests, seed=seed, drift=scenario == "drifting"))
            for seed in seeds
        )
        no_cache = scale(llm_impact(demand, requests, assumptions), factor)

        per_policy: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"miss_cost": [], "misses": []}
        )
        for r in scenario_rows:
            per_policy[r["policy"]]["miss_cost"].append(float(r["miss_cost_ms"]))
            per_policy[r["policy"]]["misses"].append(float(r["misses"]))

        policies = {}
        for name, values in per_policy.items():
            impact = scale(
                llm_impact(mean(values["miss_cost"]), mean(values["misses"]), assumptions),
                factor,
            )
            policies[name] = {"impact": impact, "reference": name in REFERENCES}
        lru = policies["LRU"]["impact"]
        for data in policies.values():
            data["saved_vs_no_cache"] = savings(no_cache, data["impact"])
            data["saved_vs_lru"] = savings(lru, data["impact"])

        summary["scenarios"][scenario] = {"no_cache": no_cache, "policies": policies}

        print(f"\n=== {scenario.upper()}: per {PER:,} questions, cache of "
              f"{summary['capacity']} answers ===")
        print(f"{'Policy':<15} {'LLM calls':>10} {'Cost (₹)':>11} {'Energy kWh':>11} "
              f"{'CO2 kg':>8} {'User wait h':>12} {'₹ saved vs LRU':>15} {'CO2 kg saved vs LRU':>20}")
        print(f"{'No cache':<15} {no_cache['llm_calls']:>10,.0f} {no_cache['cost_inr']:>11,.0f} "
              f"{no_cache['energy_kwh']:>11,.1f} {no_cache['co2_kg']:>8,.1f} "
              f"{no_cache['user_wait_hours']:>12,.0f}")
        for name, data in sorted(policies.items(), key=lambda kv: kv[1]["impact"]["cost_inr"]):
            i, s = data["impact"], data["saved_vs_lru"]
            label = name + (" (ref)" if data["reference"] else "")
            print(f"{label:<15} {i['llm_calls']:>10,.0f} {i['cost_inr']:>11,.0f} "
                  f"{i['energy_kwh']:>11,.1f} {i['co2_kg']:>8,.1f} {i['user_wait_hours']:>12,.0f} "
                  f"{s['cost_inr']:>15,.0f} {s['co2_kg']:>20,.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\nAssumptions: {assumptions.as_dict()}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()