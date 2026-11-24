"""Create the report figures (PNG, 200 dpi) in artifacts/figures/.

Run after the benchmarks, scripts/impact_report.py and
scripts/behaviour_experiments.py. Figures:

  fig1_toy_workload.png    old 7-object workload: a trivial rule is near-optimal
  fig2_chatbot_cost.png    chatbot workload: AI time spent on misses per policy
  fig3_cache_sizes.png     gap closed at cache sizes 50 / 100 / 200
  fig4_drift_timeline.png  how policies react when popular topics change
  fig5_safety_guard.png    the guard catching a deliberately bad policy
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from smartcache.environment.workload import generate_workload  # noqa: E402
from smartcache.policies.oracle import optimal_miss_cost  # noqa: E402

ART = Path("artifacts")
OUT = ART / "figures"

# Colours: one accent for our policy, grey for context, and a fixed
# categorical order when several policies are compared as lines.
OURS = "#2a78d6"
CONTEXT = "#9aa3ae"
SERIES = {"LearnedValue": "#2a78d6", "S3-FIFO": "#eb6834", "LRU": "#1baf7a",
          "LFU": "#eda100", "SIEVE": "#e87ba4"}
TEXT = "#1f2328"
MUTED = "#5f6670"
REFERENCES = {"Belady", "CostBelady"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_by_policy(rows, scenario, field="miss_cost_ms"):
    values = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario:
            values[r["policy"]].append(float(r[field]))
    return {k: mean(v) for k, v in values.items()}


def label_line_ends(ax, items, min_gap_fraction=0.07):
    """Write a label at the end of each line, nudged apart so none overlap.

    items: (x, y, text) tuples. Labels are spread vertically so that
    neighbours are at least `min_gap_fraction` of the y-range apart.
    """
    low, high = ax.get_ylim()
    gap = (high - low) * min_gap_fraction
    ordered = sorted(items, key=lambda item: item[1])
    placed: list[float] = []
    for _, y, _ in ordered:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    # Re-centre the stack on the lines it labels, so labels stay close to them.
    shift = mean(y for _, y, _ in ordered) - mean(placed)
    placed = [y + shift for y in placed]
    if placed[0] < low:
        placed = [y + (low - placed[0]) for y in placed]
    # If the top label was pushed out of the plot, shift the whole stack down.
    overflow = placed[-1] - (high - gap / 2)
    if overflow > 0:
        placed = [max(low, y - overflow) for y in placed]
    for (x, y, text), y_label in zip(ordered, placed):
        ax.annotate(text, (x, y), xytext=(x, y_label), textcoords="data",
                    va="center", fontsize=9, color=TEXT,
                    xycoords="data", annotation_clip=False,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def gap_closed(costs, reference):
    lru = costs["LRU"]
    return {k: (lru - v) / (lru - reference) * 100 for k, v in costs.items()}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT / name}")


def fig1_toy_workload():
    rows = read_csv(ART / "comprehensive_multi_algo_benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, scenario in zip(axes, ("stationary", "drifting")):
        costs = mean_by_policy(rows, scenario)
        seeds = sorted({int(r["seed"]) for r in rows if r["scenario"] == scenario})
        requests = int(rows[0]["requests"])
        optimum = mean(
            optimal_miss_cost(generate_workload(requests, seed=s, drift=scenario == "drifting"), 3)
            for s in seeds
        )
        gaps = sorted(gap_closed(costs, optimum).items(), key=lambda kv: kv[1])
        names = [k for k, _ in gaps]
        values = [v for _, v in gaps]
        colors = [OURS if n == "EvictCheapest" else CONTEXT for n in names]
        ax.axvline(100, color=MUTED, lw=1, ls="--", zorder=0)
        ax.barh(names, values, color=colors, height=0.7)
        for y, v in enumerate(values):
            # Negative bars get their label inside the bar's right end (next to 0).
            x, ha = (v + 2, "left") if v >= 0 else (2, "left")
            ax.text(x, y, f"{v:.0f}%", va="center", ha=ha, fontsize=8, color=TEXT,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xlim(min(-20, min(values) - 5), 118)
        ax.text(100, len(values) - 0.4, "optimum", fontsize=8, color=MUTED, ha="center")
        ax.axvline(0, color="#c9ced6", lw=1)
        ax.set_title(f"{scenario.capitalize()} traffic")
        ax.set_xlabel("Gap closed from LRU (0%) to the offline optimum (100%)")
    for ax in axes:
        ax.set_xlim(min(a.get_xlim()[0] for a in axes), 118)
    fig.suptitle("Original 7-object workload: the one-line rule \"evict the cheapest\" (blue) "
                 "is already near-optimal", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig1_toy_workload.png")


def fig2_chatbot_cost():
    rows = read_csv(ART / "chatbot_benchmark.csv")
    capacity = rows[0]["capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, scenario in zip(axes, ("stationary", "drifting")):
        costs = mean_by_policy(rows, scenario)
        order = sorted(costs, key=costs.get, reverse=True)
        hours = [costs[n] / 3_600_000 for n in order]
        for y, (name, value) in enumerate(zip(order, hours)):
            if name in REFERENCES:
                ax.barh(y, value, height=0.7, facecolor="white", edgecolor=CONTEXT,
                        hatch="///", linewidth=1)
            else:
                ax.barh(y, value, height=0.7, color=OURS if name == "LearnedValue" else CONTEXT)
            ax.text(value + 0.2, y, f"{value:.1f} h", va="center", fontsize=8, color=TEXT)
        ax.set_yticks(range(len(order)),
                      [n + (" (ref)" if n in REFERENCES else "") for n in order])
        ax.set_title(f"{scenario.capitalize()} traffic")
        ax.set_xlabel("AI hours on misses per 20,000 questions\n(lower is better)")
    fig.suptitle(f"Chatbot workload, cache of {capacity} answers, mean of 5 unseen test traces. "
                 "Blue: LearnedValue (ours); hatched: future-knowing references",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig2_chatbot_cost.png")


def fig3_cache_sizes():
    files = {50: "chatbot_benchmark_c50.csv", 100: "chatbot_benchmark.csv",
             200: "chatbot_benchmark_c200.csv"}
    shown = ["LearnedValue", "S3-FIFO", "SIEVE", "LFU"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, scenario in zip(axes, ("stationary", "drifting")):
        series = defaultdict(list)
        sizes = []
        for size, name in files.items():
            path = ART / name
            if not path.is_file():
                continue
            costs = mean_by_policy(read_csv(path), scenario)
            reference = min(costs[n] for n in REFERENCES)
            gaps = gap_closed(costs, reference)
            sizes.append(size)
            for policy in shown:
                series[policy].append(gaps[policy])
        for policy in shown:
            ax.plot(sizes, series[policy], marker="o", ms=6, lw=2,
                    color=SERIES[policy], label=policy)
        ax.axhline(0, color="#c9ced6", lw=1)
        ax.set_ylim(-45, 75)
        label_line_ends(ax, [(sizes[-1] + 8, series[p][-1], p) for p in shown])
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.set_xlim(sizes[0] - 15, sizes[-1] + 60)
        ax.set_title(f"{scenario.capitalize()} traffic")
        ax.set_xlabel("Cache size (answers)")
        ax.grid(axis="y", color="#eceff3")
    axes[0].set_ylabel("Gap closed from LRU to best reference (%)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("The same LearnedValue model (trained at size 100) leads at every cache size",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig3_cache_sizes.png")


def rolling(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def fig4_drift_timeline():
    rows = read_csv(ART / "drift_timeline.csv")
    policies = [c for c in rows[0] if c != "request"]
    window = 1000
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(window, len(rows) + 1)
    ends = []
    for policy in policies:
        values = np.array([float(r[policy]) for r in rows]) / 1000  # seconds
        smooth = rolling(values, window) * 1000 / 60  # minutes per 1,000 questions
        ax.plot(x, smooth, lw=2 if policy == "LearnedValue" else 1.5,
                color=SERIES.get(policy, CONTEXT), label=policy)
        ends.append((x[-1] + 300, smooth[-1], policy))
    total = len(rows)
    label_line_ends(ax, ends)
    for boundary in (total // 3, 2 * total // 3):
        ax.axvline(boundary, color=MUTED, lw=1, ls="--")
        ax.text(boundary + 150, ax.get_ylim()[0], "new hot topic ", fontsize=8,
                color=MUTED, va="bottom")
    ax.set_xlim(0, total * 1.1)
    ax.set_xlabel("Request number")
    ax.set_ylabel("AI minutes per 1,000 questions\n(rolling, lower is better)")
    ax.grid(axis="y", color="#eceff3")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=9)
    fig.suptitle("Drifting traffic: LFU keeps stale questions after each topic change; "
                 "LearnedValue adapts and stays lowest", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig4_drift_timeline.png")


def fig5_safety_guard():
    rows = read_csv(ART / "guard_timeline.csv")
    events = json.loads((ART / "guard_events.json").read_text())
    policies = [c for c in rows[0] if c != "request"]
    colors = {"LRU": "#1baf7a", "EvictMostExpensive": "#e34948",
              "Guarded(EvictMostExpensive)": OURS}
    labels = {"EvictMostExpensive": "Bad policy alone",
              "Guarded(EvictMostExpensive)": "Bad policy + safety guard", "LRU": "LRU"}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(1, len(rows) + 1)
    ends = []
    for policy in policies:
        hours = np.cumsum([float(r[policy]) for r in rows]) / 3_600_000
        ax.plot(x, hours, lw=2, color=colors.get(policy, CONTEXT), label=labels.get(policy, policy))
        ends.append((x[-1] + 400, hours[-1], f"{labels.get(policy, policy)}: {hours[-1]:.1f} h"))
    for event in events:
        if event["event"] == "switch_to_fallback":
            ax.axvline(event["timestamp"], color=OURS, lw=0.8, ls=":")
    ax.set_xlim(0, len(rows) * 1.32)
    label_line_ends(ax, ends)
    ax.set_xlabel("Request number (dotted lines: guard switches to LRU)")
    ax.set_ylabel("Cumulative AI time on misses (hours)")
    ax.grid(axis="y", color="#eceff3")
    fig.suptitle("Safety guard: a deliberately bad policy is detected after 2,000 requests "
                 "and kept close to LRU", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_safety_guard.png")


def main() -> None:
    fig1_toy_workload()
    fig2_chatbot_cost()
    fig3_cache_sizes()
    fig4_drift_timeline()
    fig5_safety_guard()


if __name__ == "__main__":
    main()