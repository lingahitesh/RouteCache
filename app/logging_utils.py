"""
Structured logging for RouteCache.

Two output channels:
  1. SQLite request_logs table — for offline analysis and metrics
  2. JSON lines file — for real-time debugging and log aggregation

Every log entry carries a request_id (UUID) so you can trace a single
request through every component.

Traceability: FR-6.4, NFR-9
Build Plan: Step 5
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from app.db.connection import get_db

logger = logging.getLogger("routecache")


def hash_query(query: str) -> str:
    """SHA-256 hash of the query text (for privacy + deduplication)."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def log_request(
    request_id: str,
    query_text: str,
    cache_zone: str | None = None,
    similarity: float | None = None,
    risk_features: dict | None = None,
    risk_score: float | None = None,
    thresholds: dict | None = None,
    verifier_verdict: str | None = None,
    verifier_score: float | None = None,
    route: str | None = None,
    router_score: float | None = None,
    served_entry_id: int | None = None,
    model_used: str | None = None,
    cost_usd: float = 0.0,
    baseline_cost_usd: float = 0.0,
    latency_embed_ms: float | None = None,
    latency_faiss_ms: float | None = None,
    latency_risk_ms: float | None = None,
    latency_verify_ms: float | None = None,
    latency_route_ms: float | None = None,
    latency_model_ms: float | None = None,
    latency_total_ms: float = 0.0,
) -> None:
    """
    Log a completed request to both SQLite and the JSON log file.

    Called once per request, after the response is assembled but
    before it's returned to the client.
    """
    query_hash = hash_query(query_text)

    # ── SQLite insert ──
    try:
        db = get_db()
        db.execute("""
            INSERT INTO request_logs (
                request_id, query_text, query_hash,
                cache_zone, similarity,
                risk_features_json, risk_score, thresholds_json,
                verifier_verdict, verifier_score,
                route, router_score, served_entry_id, model_used,
                cost_usd, baseline_cost_usd,
                latency_embed_ms, latency_faiss_ms, latency_risk_ms,
                latency_verify_ms, latency_route_ms, latency_model_ms,
                latency_total_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id, query_text, query_hash,
            cache_zone, similarity,
            json.dumps(risk_features) if risk_features else None,
            risk_score,
            json.dumps(thresholds) if thresholds else None,
            verifier_verdict, verifier_score,
            route, router_score, served_entry_id, model_used,
            cost_usd, baseline_cost_usd,
            latency_embed_ms, latency_faiss_ms, latency_risk_ms,
            latency_verify_ms, latency_route_ms, latency_model_ms,
            latency_total_ms,
        ))
        db.commit()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to write to SQLite: {e}")

    # ── JSON log line ──
    try:
        config_module = __import__("config.config", fromlist=["load_config"])
        cfg = config_module.load_config()
        log_path = Path(cfg.logging.json_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "query_hash": query_hash,
            "cache_zone": cache_zone,
            "similarity": similarity,
            "risk_features": risk_features,
            "risk_score": risk_score,
            "verifier": {"verdict": verifier_verdict, "score": verifier_score},
            "route": route,
            "router_score": router_score,
            "cost_usd": cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "latency_ms": {
                "embed": latency_embed_ms, "faiss": latency_faiss_ms,
                "risk": latency_risk_ms, "verify": latency_verify_ms,
                "route": latency_route_ms, "model": latency_model_ms,
                "total": latency_total_ms,
            },
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"[{request_id}] Failed to write JSON log: {e}")