"""
Metrics collection and aggregation.

Collects per-request metrics in memory and exposes them via GET /metrics.
Also writes to SQLite (via log_request, Step 5) for offline analysis.

Traceability: FR-6.3
Build Plan: Step 13
"""

import threading
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AggregateMetrics:
    """Running aggregate metrics since last reset."""
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_baseline_cost_usd: float = 0.0

    cache_hits: int = 0
    cache_verified_hits: int = 0
    cache_misses: int = 0
    verify_zone_entries: int = 0
    verifier_passes: int = 0
    verifier_rejects: int = 0

    routes_to_small: int = 0
    routes_to_large: int = 0

    failures: dict = field(default_factory=lambda: defaultdict(int))
    fallbacks: int = 0

    latency_totals: list = field(default_factory=list)
    latency_embeds: list = field(default_factory=list)
    latency_verifies: list = field(default_factory=list)
    latency_models: list = field(default_factory=list)


class MetricsRecorder:
    """
    Thread-safe metrics recorder.

    Usage:
        recorder = MetricsRecorder()
        recorder.record_request(trace, response, latencies)
        snapshot = recorder.snapshot()  # for GET /metrics
    """

    def __init__(self):
        self._metrics = AggregateMetrics()
        self._lock = threading.Lock()

    def record_request(
        self,
        action: str,
        zone: str | None,
        cost_usd: float,
        baseline_cost_usd: float,
        verifier_verdict: str | None,
        latency_total_ms: float,
        latency_embed_ms: float | None = None,
        latency_verify_ms: float | None = None,
        latency_model_ms: float | None = None,
    ) -> None:
        with self._lock:
            m = self._metrics
            m.total_requests += 1
            m.total_cost_usd += cost_usd
            m.total_baseline_cost_usd += baseline_cost_usd

            if action == "serve" and zone == "SERVE":
                m.cache_hits += 1
            elif action == "serve" and zone == "VERIFY":
                m.cache_verified_hits += 1
            elif action in ("small", "large"):
                m.cache_misses += 1

            if zone == "VERIFY":
                m.verify_zone_entries += 1
                if verifier_verdict == "SAFE":
                    m.verifier_passes += 1
                elif verifier_verdict == "UNSAFE":
                    m.verifier_rejects += 1

            if action == "small":
                m.routes_to_small += 1
            elif action == "large":
                m.routes_to_large += 1

            m.latency_totals.append(latency_total_ms)
            if latency_embed_ms is not None:
                m.latency_embeds.append(latency_embed_ms)
            if latency_verify_ms is not None and latency_verify_ms > 0:
                m.latency_verifies.append(latency_verify_ms)
            if latency_model_ms is not None and latency_model_ms > 0:
                m.latency_models.append(latency_model_ms)

    def record_failure(self, component: str, fallback: bool = False) -> None:
        with self._lock:
            self._metrics.failures[component] += 1
            if fallback:
                self._metrics.fallbacks += 1

    def snapshot(self) -> dict:
        """
        Return a JSON-serializable snapshot of all metrics.
        This backs GET /metrics.
        """
        with self._lock:
            m = self._metrics
            total = max(m.total_requests, 1)  # avoid division by zero
            served = m.cache_hits + m.cache_verified_hits

            def percentile(data, p):
                if not data:
                    return 0.0
                import numpy as np
                return float(np.percentile(data, p))

            return {
                "total_requests": m.total_requests,

                # ── Cost ──
                "total_cost_usd": round(m.total_cost_usd, 6),
                "total_baseline_cost_usd": round(m.total_baseline_cost_usd, 6),
                "cost_saved_percent": round(
                    (1 - m.total_cost_usd / max(m.total_baseline_cost_usd, 1e-10)) * 100, 2
                ),

                # ── Cache ──
                "cache_hit_rate": round(served / total, 4),
                "cache_hits": m.cache_hits,
                "cache_verified_hits": m.cache_verified_hits,
                "cache_misses": m.cache_misses,

                # ── Verifier ──
                "verify_zone_rate": round(m.verify_zone_entries / total, 4),
                "verifier_pass_rate": round(
                    m.verifier_passes / max(m.verify_zone_entries, 1), 4
                ),
                "verifier_passes": m.verifier_passes,
                "verifier_rejects": m.verifier_rejects,

                # ── Routing ──
                "routes_to_small": m.routes_to_small,
                "routes_to_large": m.routes_to_large,
                "small_model_share": round(m.routes_to_small / total, 4),

                # ── Latency ──
                "latency_total_p50_ms": percentile(m.latency_totals, 50),
                "latency_total_p95_ms": percentile(m.latency_totals, 95),
                "latency_total_p99_ms": percentile(m.latency_totals, 99),
                "latency_embed_p50_ms": percentile(m.latency_embeds, 50),
                "latency_embed_p95_ms": percentile(m.latency_embeds, 95),
                "latency_verify_p50_ms": percentile(m.latency_verifies, 50),
                "latency_verify_p95_ms": percentile(m.latency_verifies, 95),
                "latency_model_p50_ms": percentile(m.latency_models, 50),
                "latency_model_p95_ms": percentile(m.latency_models, 95),

                # ── Failures ──
                "failures": dict(m.failures),
                "fallbacks": m.fallbacks,
            }

    def reset(self):
        """Reset all metrics. Used between evaluation runs."""
        with self._lock:
            self._metrics = AggregateMetrics()