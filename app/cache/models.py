"""
Data structures for the semantic cache.

These are the typed objects that flow between CacheManager, the
decision engine, and the API layer. They are NOT Pydantic models
(no validation overhead on the hot path) — they're dataclasses.

Build Plan: Step 7
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Provenance:
    """
    Origin of a cached answer (FR-2.11).

    Passed into DecisionPolicy.decide() — the decision engine needs
    to know WHERE an answer came from (small model? large model?)
    but should never see the answer text itself.
    """
    source_model: Literal["small", "large"]
    route_decision: Literal["small", "large", "cache"]
    from_cache: bool
    created_at: datetime


@dataclass
class CacheCandidate:
    """
    A potential cache hit returned by CacheManager.lookup().

    This is what the decision engine evaluates. It includes the
    similarity score, the cached answer, and the provenance —
    everything needed to decide SERVE / VERIFY / MISS.
    """
    entry_id: int
    query: str                     # the ORIGINAL query that was cached
    answer: str                    # the cached answer
    similarity: float              # cosine similarity to the new query
    provenance: Provenance
    system_context_hash: str | None = None
    temperature: float | None = None


@dataclass
class RequestContext:
    """
    Context that affects cache key matching (FR-2.4).

    system_prompt and temperature are hashed into the lookup key,
    so two contextually different requests with identical query text
    never collide.
    """
    system_prompt: str | None = None
    conversation_history: list[dict] | None = None
    temperature: float = 0.0
    model_override: str | None = None
    bypass_cache: bool = False