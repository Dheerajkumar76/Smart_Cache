"""Collect every number the report needs into artifacts/results_summary.md.

Reads the CSV/JSON outputs of the benchmark, impact and behaviour scripts.
Missing inputs are skipped with a note, so it can run at any stage.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from smartcache.environment.workload import generate_workload
from smartcache.policies.oracle import optimal_miss_cost

ART = Path("artifacts")
REFERENCES = {"Belady", "CostBelady"}


def read_csv(name: str) -> list[dict[str, str]] | None:
    path = ART / name
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def per_policy(rows, scenario, field="miss_cost_ms"):
    values = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario:
            values[r["policy"]].append(float(r[field]))
    return values


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def toy_section() -> str:
    rows = read_csv("comprehensive_multi_algo_benchmark.csv")
    if rows is None:
        return "_Original workload results not found._"
    out = ["## 1. Original 7-object workload (capacity 3, 10 seeds)\n",
           "Gap closed: 0% = LRU, 100% = exact offline optimum.\n"]
    for scenario in ("stationary", "drifting"):
        costs = per_policy(rows, scenario)
        seeds = sorted({int(r["seed"]) for r in rows if r["scenario"] == scenario})
        requests = int(rows[0]["requests"])
        optimum = mean(
            optimal_miss_cost(generate_workload(requests, seed=s, drift=scenario == "drifting"), 3)
            for s in seeds
        )
        lru = mean(costs["LRU"])
        body = [["**Optimum (offline)**", f"{optimum:,.1f}", "100.0%"]]
        for name in sorted(costs, key=lambda n: mean(costs[n])):
            m = mean(costs[name])
            body.append([name, f"{m:,.1f} ± {stdev(costs[name]):,.1f}",
                         f"{(lru - m) / (lru - optimum) * 100:.1f}%"])
        out += [f"### {scenario.capitalize()}\n", table(["Policy", "Miss cost (ms)", "Gap closed"], body), ""]
    return "\n".join(out)


def chatbot_section() -> str:
    files = {50: "chatbot_benchmark_c50.csv", 100: "chatbot_benchmark.csv",
             200: "chatbot_benchmark_c200.csv"}
    out = ["## 2. AI chatbot workload (5,000 questions, 20,000 requests per trace, "
           "5 unseen test seeds 30000-30004)\n",
           "Gap closed: 0% = LRU, 100% = best future-knowing reference (Belady or CostBelady).\n"]
    main = read_csv(files[100])
    if main is None:
        return "_Chatbot benchmark results not found._"
    for scenario in ("stationary", "drifting"):
        costs = per_policy(main, scenario)
        hits = per_policy(main, scenario, "hit_ratio")
        lru = mean(costs["LRU"])
        ref = min(mean(costs[n]) for n in REFERENCES if n in costs)
        body = []
        for name in sorted(costs, key=lambda n: mean(costs[n])):
            m = mean(costs[name])
            label = f"{name} (ref)" if name in REFERENCES else (f"**{name}**" if name == "LearnedValue" else name)
            body.append([label, f"{m / 1000:,.1f} ± {stdev(costs[name]) / 1000:,.1f}",
                         f"{mean(hits[name]) * 100:.2f}%", f"{(lru - m) / lru * 100:+.2f}%",
                         f"{(lru - m) / (lru - ref) * 100:.1f}%"])
        out += [f"### Cache of 100 answers, {scenario}\n",
                table(["Policy", "AI time on misses (s)", "Hit ratio", "vs LRU", "Gap closed"], body), ""]

    out.append("### Cache size sweep (same model, trained at size 100)\n")
    body = []
    for size, name in files.items():
        rows = read_csv(name)
        if rows is None:
            continue
        for scenario in ("stationary", "drifting"):
            costs = per_policy(rows, scenario)
            lru = mean(costs["LRU"])
            ref = min(mean(costs[n]) for n in REFERENCES if n in costs)
            gap = {n: (lru - mean(v)) / (lru - ref) * 100 for n, v in costs.items()}
            by_seed = defaultdict(dict)
            for r in rows:
                if r["scenario"] == scenario:
                    by_seed[r["seed"]][r["policy"]] = float(r["miss_cost_ms"])
            wins = sum(s["LearnedValue"] < s["S3-FIFO"] for s in by_seed.values())
            body.append([str(size), scenario, f"{gap['LearnedValue']:.1f}%", f"{gap['S3-FIFO']:.1f}%",
                         f"{gap['SIEVE']:.1f}%", f"{wins}/{len(by_seed)}"])
    out.append(table(["Cache size", "Traffic", "LearnedValue", "S3-FIFO", "SIEVE",
                      "LearnedValue beats S3-FIFO (seeds)"], body))
    return "\n".join(out)


def impact_section() -> str:
    path = ART / "impact_summary.json"
    if not path.is_file():
        return "_Impact summary not found._"
    data = json.loads(path.read_text())
    a = data["assumptions"]
    out = ["## 3. Impact per 1 million questions (cache of 100 answers)\n"]
    for scenario, s in data["scenarios"].items():
        p = s["policies"]
        body = []
        for name, label in (("LearnedValue", "vs LRU"), ("S3-FIFO", "vs LRU")):
            v = p[name]["saved_vs_lru"]
            body.append([f"{name} {label}", f"₹{v['cost_inr']:,.0f}", f"{v['energy_kwh']:,.1f}",
                         f"{v['co2_kg']:,.1f}", f"{v['user_wait_hours']:,.0f}", f"{v['llm_calls']:,.0f}"])
        v = p["LearnedValue"]["saved_vs_no_cache"]
        body.append(["LearnedValue vs no cache", f"₹{v['cost_inr']:,.0f}", f"{v['energy_kwh']:,.1f}",
                     f"{v['co2_kg']:,.1f}", f"{v['user_wait_hours']:,.0f}", f"{v['llm_calls']:,.0f}"])
        out += [f"### {scenario.capitalize()}\n",
                table(["Saving", "LLM spend", "Energy (kWh)", "CO₂ (kg)", "User wait (h)", "LLM calls"], body), ""]
    stationary = data["scenarios"]["stationary"]["policies"]["LearnedValue"]["saved_vs_lru"]
    out.append(
        f"**Headline (1 million questions/day, stationary, vs LRU, per year):** "
        f"₹{stationary['cost_inr'] * 365 / 1e5:,.2f} lakh, {stationary['energy_kwh'] * 365:,.0f} kWh, "
        f"{stationary['co2_kg'] * 365 / 1000:,.1f} tonnes CO₂, "
        f"{stationary['user_wait_hours'] * 365:,.0f} hours of user waiting.\n")
    out.append(
        f"Assumptions: ₹{a['price_inr_per_1k_output_tokens']} per 1,000 output tokens (editable); "
        f"{a['energy_wh_per_1k_output_tokens']} Wh per 1,000 output tokens (from Epoch AI's ~0.3 Wh "
        f"per typical ChatGPT query, assuming ~500-token answers); {a['grid_kg_co2_per_kwh']} kg CO₂/kWh "
        f"(CEA CO₂ Baseline Database v20, Indian grid); LLM time = {a['time_to_first_token_ms']} ms + "
        f"{a['ms_per_token']} ms per token.")
    return "\n".join(out)


def behaviour_section() -> str:
    out = ["## 4. Behaviour over time\n"]
    drift = read_csv("drift_timeline.csv")
    if drift is not None:
        policies = [c for c in drift[0] if c != "request"]
        third = len(drift) // 3
        body = []
        for p in policies:
            values = [float(r[p]) for r in drift]
            phases = [sum(values[i * third:(i + 1) * third]) / 1000 for i in range(3)]
            body.append([p, f"{sum(values) / 1000:,.0f}"] + [f"{x:,.0f}" for x in phases])
        out += [f"### Drift timeline ({len(drift):,} requests, topics change twice)\n",
                table(["Policy", "Total AI time (s)", "Phase 1", "Phase 2", "Phase 3"], body), ""]
    guard = read_csv("guard_timeline.csv")
    events_path = ART / "guard_events.json"
    if guard is not None and events_path.is_file():
        events = json.loads(events_path.read_text())
        policies = [c for c in guard[0] if c != "request"]
        body = [[p, f"{sum(float(r[p]) for r in guard) / 1000:,.0f}"] for p in policies]
        first = next((e["timestamp"] for e in events if e["event"] == "switch_to_fallback"), None)
        out += [f"### Safety guard ({len(guard):,} requests)\n",
                table(["Policy", "Total AI time (s)"], body), "",
                f"First switch to LRU at request {first:,}; "
                f"{sum(e['event'] == 'switch_to_fallback' for e in events)} switches in total "
                f"(the guard retries the primary policy after each 5,000-request cooldown)."]
    return "\n".join(out)


def main() -> None:
    sections = ["# SmartCache results summary\n",
                "Generated by scripts/results_summary.py from the files in artifacts/.\n",
                toy_section(), chatbot_section(), impact_section(), behaviour_section()]
    path = ART / "results_summary.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()