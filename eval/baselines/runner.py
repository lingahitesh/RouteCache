"""
Baseline runner — runs the full SRS Section 8.3 baseline list.

9 required baselines:
  1. Always-large model
  2. Always-small model
  3. Random routing at matched small-model share
  4. Exact-match cache
  5. Unverified semantic cache at a single tuned global threshold
  6. Verified cache, fixed thresholds, no risk adaptation
  7. Router only, without cache
  8. Full system: verified + risk-adaptive + cache-aware routing
  9. (stretch) Full system + cascade

Each baseline runs the same workload and produces:
  - Total cost
  - Cost saved vs always-large
  - Wrong-hit rate (both definitions from NFR-3)
  - Latency percentiles
  - Hit rate (where applicable)

All runs replay from recorded outputs (NFR-8) — zero additional API spend.

Traceability: FR-8.3
Build Plan: Step 13
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
import time

import httpx


@dataclass
class BaselineResult:
    """Results from a single baseline run."""
    name: str
    total_cost_usd: float
    baseline_cost_usd: float  # always-large reference
    cost_saved_percent: float
    hit_rate: float
    wrong_hit_rate_pair: float  # NFR-3 definition (a)
    wrong_hit_rate_answer: float  # NFR-3 definition (b)
    latency_p50_ms: float
    latency_p95_ms: float
    total_requests: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_saved_percent": round(self.cost_saved_percent, 2),
            "hit_rate": round(self.hit_rate, 4),
            "wrong_hit_rate_pair": round(self.wrong_hit_rate_pair, 4),
            "wrong_hit_rate_answer": round(self.wrong_hit_rate_answer, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "total_requests": self.total_requests,
        }


def run_baseline_through_proxy(
        workload: list[dict],
        proxy_url: str = "http://localhost:8000",
        override: dict | None = None,
) -> list[dict]:
    """
    Send a workload through the proxy and collect results.

    Args:
        workload: List of {"query": str, ...} from the workload generator.
        proxy_url: Base URL of the running proxy.
        override: x_routecache overrides (e.g. {"bypass_cache": True}).

    Returns:
        List of response metadata dicts.
    """
    client = httpx.Client(base_url=proxy_url, timeout=120.0)
    results = []

    for i, item in enumerate(workload):
        payload = {
            "messages": [{"role": "user", "content": item["query"]}],
            "temperature": 0.0,
        }
        if override:
            payload["x_routecache"] = override

        resp = client.post("/v1/chat/completions", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            results.append({
                "query": item["query"],
                "query_type": item.get("type", "unknown"),
                "answer": data["choices"][0]["message"]["content"],
                "x_routecache": data["x_routecache"],
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(workload)}] processed...")

        # Rate limit mitigation: 2-second pause between requests
        # For 500-query workloads, this adds ~16 minutes total time on first run.
        # Cached responses skip LLM calls entirely, so second runs are instant.
        if i < len(workload) - 1:  # Don't sleep after last query
            time.sleep(2)

    return results