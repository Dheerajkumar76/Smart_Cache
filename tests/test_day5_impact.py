import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smartcache.api.app import create_app
from smartcache.api.simulation import build_simulation
from smartcache.impact import ImpactAssumptions, llm_impact, savings

MODEL = Path("artifacts/learned_value.npz")


def test_llm_impact_recovers_tokens_and_converts_units():
    a = ImpactAssumptions()
    # Two calls of 100 tokens each: 2 * (300 + 100 * 12) = 3000 ms.
    impact = llm_impact(3000.0, 2, a)
    assert impact["output_tokens"] == pytest.approx(200)
    assert impact["cost_inr"] == pytest.approx(200 / 1000 * a.price_inr_per_1k_output_tokens)
    kwh = 200 / 1000 * a.energy_wh_per_1k_output_tokens / 1000
    assert impact["energy_kwh"] == pytest.approx(kwh)
    assert impact["co2_kg"] == pytest.approx(kwh * a.grid_kg_co2_per_kwh)
    assert impact["user_wait_hours"] == pytest.approx(3000 / 3_600_000)


def test_savings_are_positive_when_policy_is_cheaper():
    base = llm_impact(10_000.0, 4)
    better = llm_impact(4_000.0, 2)
    saved = savings(base, better)
    assert all(value > 0 for value in saved.values())


def test_explorer_replays_the_chatbot_workload():
    result = build_simulation(
        workload="chatbot",
        scenario="drifting",
        seed=3,
        requests_count=300,
        capacity=20,
        checkpoint=None,
        learned_model=str(MODEL) if MODEL.is_file() else None,
    )
    assert result["workload"] == "chatbot"
    assert len(result["requests"]) == 300
    assert result["requests"][0]["key"].startswith("q")
    names = [policy["policy"] for policy in result["policies"]]
    assert "S3-FIFO" in names and "SIEVE" in names
    if MODEL.is_file():
        assert "LearnedValue" in names


def test_explorer_limits_replay_length():
    with pytest.raises(ValueError):
        build_simulation(
            workload="chatbot", scenario="stationary", seed=1,
            requests_count=5000, capacity=20, checkpoint=None,
        )
    with pytest.raises(ValueError):
        build_simulation(
            workload="unknown", scenario="stationary", seed=1,
            requests_count=10, capacity=3, checkpoint=None,
        )


def test_impact_endpoints(tmp_path, monkeypatch):
    summary = {"assumptions": ImpactAssumptions().as_dict(), "per_questions": 1_000_000,
               "capacity": 100, "scenarios": {}}
    path = tmp_path / "impact.json"
    path.write_text(json.dumps(summary))
    monkeypatch.setenv("IMPACT_SUMMARY", str(path))
    with TestClient(create_app(cache_backend="memory")) as client:
        assert client.get("/impact-data").json()["capacity"] == 100
        page = client.get("/impact")
        assert page.status_code == 200
        assert "Impact report" in page.text

    monkeypatch.setenv("IMPACT_SUMMARY", str(tmp_path / "missing.json"))
    with TestClient(create_app(cache_backend="memory")) as client:
        assert client.get("/impact-data").status_code == 404