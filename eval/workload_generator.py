"""
Synthetic workload generator for evaluation.

Generates a stream of queries with controllable:
  - Repeat rate (same query appears multiple times, Zipf distribution)
  - Paraphrase rate (semantically equivalent but differently worded)
  - Query type distribution (factual, code, math, reasoning, etc.)

SRS FR-8.2: "The harness shall include a workload generator that
produces repeats and paraphrases at controllable rates."

Build Plan: Step 13
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    n_unique_queries: int = 100      # base query set size
    total_requests: int = 500        # total requests in the workload
    exact_repeat_rate: float = 0.3   # fraction that are exact repeats
    paraphrase_rate: float = 0.2     # fraction that are paraphrases
    zipf_exponent: float = 1.2       # Zipf skew (higher = more concentrated)
    seed: int = 42                   # reproducibility


# Paraphrase templates — simple text transformations
PARAPHRASE_TEMPLATES = [
    lambda q: f"Can you tell me: {q.lower().rstrip('?')}?",
    lambda q: f"{q.rstrip('?')} — explain please.",
    lambda q: f"I want to know: {q.lower()}",
    lambda q: q.replace("What is", "What's").replace("How do I", "How can I"),
    lambda q: q.replace("?", "? Please be concise."),
    lambda q: f"Quick question: {q.lower()}",
]


def generate_workload(
    base_queries: list[str],
    config: WorkloadConfig | None = None,
) -> list[dict]:
    """
    Generate a synthetic workload from a base query set.

    Args:
        base_queries: List of unique queries to build from.
        config: Workload parameters.

    Returns:
        List of {"query": str, "type": "original"|"repeat"|"paraphrase", "source_idx": int}
    """
    if config is None:
        config = WorkloadConfig()

    rng = random.Random(config.seed)
    queries = base_queries[:config.n_unique_queries]
    n = config.total_requests

    # Zipf distribution for query selection
    weights = [1.0 / (i + 1) ** config.zipf_exponent for i in range(len(queries))]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    workload = []

    for _ in range(n):
        # Pick a base query (Zipf-weighted)
        idx = rng.choices(range(len(queries)), weights=weights, k=1)[0]
        base = queries[idx]

        # Decide type
        roll = rng.random()
        if roll < config.exact_repeat_rate:
            # Exact repeat
            workload.append({
                "query": base,
                "type": "repeat",
                "source_idx": idx,
            })
        elif roll < config.exact_repeat_rate + config.paraphrase_rate:
            # Paraphrase
            template = rng.choice(PARAPHRASE_TEMPLATES)
            paraphrased = template(base)
            workload.append({
                "query": paraphrased,
                "type": "paraphrase",
                "source_idx": idx,
            })
        else:
            # Original (first occurrence or novel)
            workload.append({
                "query": base,
                "type": "original",
                "source_idx": idx,
            })

    return workload


def save_workload(workload: list[dict], path: str = "eval/workloads/default.json") -> None:
    """Save a generated workload to disk for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(workload, f, indent=2)


def load_workload(path: str = "eval/workloads/default.json") -> list[dict]:
    """Load a previously saved workload."""
    with open(path) as f:
        return json.load(f)