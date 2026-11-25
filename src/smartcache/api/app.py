from __future__ import annotations

import os
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from smartcache.api.demo_origin import ORIGIN_SPECS, OriginSpec, load_origin_payload
from smartcache.api.sqlite_origin import SQLiteOrigin
from smartcache.policies.cost_aware import CostAwarePolicy
from smartcache.policies.dqn_byte_policy import DQNBytePolicy
from smartcache.policies.lfu import LFUPolicy
from smartcache.policies.lru import LRUPolicy
from smartcache.policies.safe_policy import SafePolicy
from smartcache.runtime.cache_service import ByteCacheRuntime
from smartcache.runtime.redis_cache_service import RedisByteCacheRuntime
from smartcache.runtime.redis_store import RedisSharedCache
from smartcache.api.simulation import build_simulation


def create_app(
    *,
    policy_factory: Callable[[], object] | None = None,
    origin_specs: Mapping[str, OriginSpec] = ORIGIN_SPECS,
    origin_loader: Callable[[str], bytes] = load_origin_payload,
    capacity_bytes: int | None = None,
    max_objects: int | None = None,
    frequency_window: int | None = None,
    recency_window: int | None = None,
    cost_scale_ms: float | None = None,
    size_scale_bytes: float | None = None,
    origin_backend: str | None = None,
    cache_backend: str | None = None,
) -> FastAPI:
    use_sqlite = (
        (origin_backend or os.getenv("ORIGIN_BACKEND", "demo")).lower() == "sqlite"
        and origin_specs is ORIGIN_SPECS
        and origin_loader is load_origin_payload
    )
    sqlite_origin = (
        SQLiteOrigin(os.getenv("SQLITE_DATABASE", "data/origin.sqlite"))
        if use_sqlite
        else None
    )
    effective_origin_specs = (
        sqlite_origin.specs_from_profile(os.getenv("ORIGIN_PROFILE"))
        if sqlite_origin is not None
        else origin_specs
    )
    effective_origin_loader = (
        sqlite_origin.loader() if sqlite_origin is not None else origin_loader
    )
    settings = {
        "capacity_bytes": capacity_bytes or int(os.getenv("CACHE_CAPACITY_BYTES", "24576")),
        "max_objects": max_objects or int(os.getenv("CACHE_MAX_OBJECTS", "16")),
        "frequency_window": frequency_window or 100,
        "recency_window": recency_window or 1000,
        "cost_scale_ms": cost_scale_ms or 250.0,
        "size_scale_bytes": size_scale_bytes or 32768.0,
    }
    cache_backend = (cache_backend or os.getenv("CACHE_BACKEND", "memory")).lower()
    if cache_backend not in {"memory", "redis"}:
        raise ValueError("CACHE_BACKEND must be memory or redis")

    def build_policy():
        if policy_factory is not None:
            return policy_factory()
        policy_name = os.getenv("CACHE_POLICY", "dqn-byte").lower()
        if policy_name == "lru":
            return SafePolicy(LRUPolicy(), LRUPolicy())
        if policy_name == "lfu":
            return SafePolicy(LFUPolicy(), LRUPolicy())
        if policy_name == "cost-aware":
            return SafePolicy(CostAwarePolicy(), LRUPolicy())
        if policy_name == "dqn-byte":
            checkpoint = os.getenv("CACHE_CHECKPOINT")
            if not checkpoint:
                manifest_path = Path("artifacts/model_registry/active.json")
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    active = manifest.get("active", {}).get("byte")
                    if active:
                        checkpoint = active["path"]
                        expected_hash = active["sha256"]
                        actual_hash = hashlib.sha256(
                            Path(checkpoint).read_bytes()
                        ).hexdigest()
                        if actual_hash != expected_hash:
                            raise ValueError("Active byte checkpoint hash mismatch")
            checkpoint = checkpoint or "artifacts/byte_dqn_best.pt"
            try:
                primary = DQNBytePolicy(
                    checkpoint_path=checkpoint,
                    capacity_bytes=settings["capacity_bytes"],
                    max_objects=settings["max_objects"],
                    frequency_window=settings["frequency_window"],
                    recency_window=settings["recency_window"],
                    device=os.getenv("CACHE_DEVICE", "cpu"),
                )
            except (FileNotFoundError, ValueError, RuntimeError):
                fallback = os.getenv("CACHE_FALLBACK_POLICY")
                if fallback == "lru":
                    return SafePolicy(LRUPolicy(), LRUPolicy())
                if fallback == "cost-aware":
                    return SafePolicy(CostAwarePolicy(), LRUPolicy())
                raise
            return SafePolicy(primary, LRUPolicy())
        raise ValueError(f"Unknown CACHE_POLICY: {policy_name}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        policy = build_policy()
        if cache_backend == "redis":
            store = RedisSharedCache(
                url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
                namespace=os.getenv("CACHE_REDIS_NAMESPACE", "smartcache:byte:v2"),
                capacity_bytes=settings["capacity_bytes"],
                max_objects=settings["max_objects"],
            )
            store.client.ping()
            app.state.cache = RedisByteCacheRuntime(
                client=store,
                policy=policy,
                origin_specs=effective_origin_specs,
                origin_loader=effective_origin_loader,
                **settings,
            )
        else:
            app.state.cache = ByteCacheRuntime(
                policy=policy,
                origin_specs=effective_origin_specs,
                origin_loader=effective_origin_loader,
                **settings,
            )
        yield

    app = FastAPI(title="SmartCache RL", lifespan=lifespan)

    pages_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=pages_dir / "static"), name="static")

    def page(name: str) -> str:
        """Return one of the dashboard's HTML pages from this folder."""
        return (pages_dir / name).read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return page("home.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/objects")
    def objects():
        return {"objects": sorted(effective_origin_specs)}

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog():
        return page("catalog.html")

    @app.get("/catalog-data")
    def catalog_data():
        return {
            "origin_backend": "sqlite" if use_sqlite else "demo",
            "objects": [
                {
                    "key": key,
                    "retrieval_cost_ms": spec.retrieval_cost_ms,
                    "size_bytes": spec.size_bytes,
                }
                for key, spec in sorted(effective_origin_specs.items())
            ],
        }

    @app.get("/runtime-info")
    def runtime_info():
        return {
            "origin_backend": "sqlite" if use_sqlite else "demo",
            "cache_backend": cache_backend,
            "policy": os.getenv("CACHE_POLICY", "dqn-byte"),
            "capacity_bytes": settings["capacity_bytes"],
            "max_objects": settings["max_objects"],
            "objects": sorted(effective_origin_specs),
        }

    @app.get("/objects/{key}")
    def get_object(key: str):
        if key not in effective_origin_specs:
            raise HTTPException(status_code=404, detail=f"Unknown object: {key}")
        result = app.state.cache.get(key)
        headers = {
            "X-Cache": "HIT" if result.hit else "MISS",
            "X-Backend-Latency-Ms": f"{result.backend_latency_ms:.3f}",
            "X-Policy-Time-Ms": f"{result.policy_time_ms:.3f}",
            "X-Cache-Used-Bytes": str(result.used_bytes),
            "X-Cache-Bypassed": str(result.bypassed).lower(),
        }
        return Response(content=result.payload, media_type="application/octet-stream", headers=headers)

    @app.get("/metrics")
    def metrics():
        return app.state.cache.snapshot_metrics()

    @app.get("/metrics-ui", response_class=HTMLResponse)
    def metrics_ui():
        return page("metrics.html")

    @app.get("/simulation")
    def simulation(
        scenario: str = "stationary",
        seed: int = 2026,
        requests: int = 30,
        capacity: int = 3,
        workload: str = "demo",
    ):
        # The explorer replays the fixed-slot model. The live runtime uses the
        # byte-aware model, whose state/action dimensions are incompatible with
        # this slot-based simulation.
        checkpoint = os.getenv(
            "CACHE_SIMULATION_CHECKPOINT", "artifacts/dqn_best.pt"
        )
        dueling_checkpoint = os.getenv(
            "CACHE_SIMULATION_DUELING_CHECKPOINT", "artifacts/dueling_dqn_best.pt"
        )
        ppo_checkpoint = os.getenv(
            "CACHE_SIMULATION_PPO_CHECKPOINT", "artifacts/ppo_best.pt"
        )
        learned_model = os.getenv(
            "CACHE_SIMULATION_LEARNED_MODEL", "artifacts/learned_value.npz"
        )
        try:
            return build_simulation(
                scenario=scenario,
                seed=seed,
                requests_count=requests,
                capacity=capacity,
                checkpoint=checkpoint,
                dueling_checkpoint=dueling_checkpoint,
                ppo_checkpoint=ppo_checkpoint,
                learned_model=learned_model,
                workload=workload,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/impact-data")
    def impact_data():
        path = Path(os.getenv("IMPACT_SUMMARY", "artifacts/impact_summary.json"))
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Run scripts/benchmark_chatbot.py and scripts/impact_report.py first",
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/impact", response_class=HTMLResponse)
    def impact_page():
        return page("impact.html")

    @app.get("/explorer", response_class=HTMLResponse)
    def explorer():
        return page("simulation.html")

    return app


app = create_app()
