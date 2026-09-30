"""
Request and response models for the /v1/chat/completions endpoint.

These follow the OpenAI chat format (FR-1.1) with a custom x_routecache
extension block for RouteCache-specific metadata (FR-1.4, NFR-9).

Traceability: FR-1.1, FR-1.2, FR-1.4
Build Plan: Step 4
"""

from pydantic import BaseModel, Field
from typing import Literal


# ─── Request Models ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class RouteCacheRequestOptions(BaseModel):
    """
    Optional per-request overrides (FR-1.4).
    Sent as x_routecache in the request body.
    """
    quality_target: float | None = None       # hint for FR-3.7
    bypass_cache: bool = False                 # forces MISS zone unconditionally
    force_model: str | None = None            # skip routing, use this model
    risk_level: str = "default"               # override risk policy


class ChatRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Example:
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "How do I reverse a list?"}],
            "temperature": 0.0,
            "x_routecache": {"bypass_cache": false}
        }
    """
    model: str = "auto"                        # "auto" = RouteCache-managed routing
    messages: list[ChatMessage]
    temperature: float = 0.0
    x_routecache: RouteCacheRequestOptions = RouteCacheRequestOptions()


# ─── Response Models ──────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    """
    Per-component latency in milliseconds.
    Fields are nullable because not every component runs on every request.
    """
    embed: float | None = None
    lookup: float | None = None
    risk: float | None = None
    verify: float | None = None
    route: float | None = None
    model: float | None = None
    total: float = 0.0


class VerifierInfo(BaseModel):
    """Verifier result metadata (populated only if verification ran)."""
    type: str | None = None                    # "cross_encoder" or "nli"
    verdict: str | None = None                 # "SAFE" or "UNSAFE"
    score: float | None = None


class EntryProvenance(BaseModel):
    """Origin of the cache entry that was served (if any)."""
    tier: str | None = None                    # "small" or "large"
    created_at: str | None = None              # ISO-8601


class RouteCacheResponseMeta(BaseModel):
    """
    The x_routecache block in every response (NFR-9).

    This is the single most important observability surface in the system.
    Every field here ends up in the request log (FR-6.4) and is available
    for error analysis (SRS Section 8.5).
    """
    cache: str = "disabled"                    # "hit", "verified_hit", "miss", "unavailable", "disabled"
    zone: str | None = None                    # "SERVE", "VERIFY", "MISS", None
    similarity: float | None = None
    risk_flags: list[str] = []                 # names of risk features that fired
    thresholds: dict[str, float] | None = None # {"t_low": 0.82, "t_high": 0.93}
    verifier: VerifierInfo = VerifierInfo()
    entry_provenance: EntryProvenance | None = None
    decision: str | None = None                # "serve", "serve_after_verify", "route_small", "route_large"
    route: str | None = None                   # "small-model", "large-model", None
    router_score: float | None = None
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0             # always-large equivalent (FR-6.5)
    latency_ms: LatencyBreakdown = LatencyBreakdown()


class ChatResponseChoice(BaseModel):
    """One choice in the response (we always return exactly one)."""
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """
    Full response matching OpenAI's chat completion format,
    plus the x_routecache metadata block.
    """
    id: str                                    # request ID (uuid)
    object: str = "chat.completion"
    choices: list[ChatResponseChoice]
    x_routecache: RouteCacheResponseMeta