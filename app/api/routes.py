"""
API route handlers for RouteCache.

Step 4 starts with a bare pass-through to the large model.
Steps 6-11 progressively add cache lookup, verification, risk
extraction, routing, and the joint decision engine.

Traceability: FR-1.1, FR-1.2
Build Plan: Step 4
"""

import uuid
import time
from fastapi import APIRouter, HTTPException
from datetime import datetime
from config.config import load_config
from app.adapters.adapters import GroqAdapter, RetryingAdapter
from app.cost import compute_cost_for_model, compute_baseline_cost
from app.api.models import (
    ChatRequest, ChatResponse, ChatResponseChoice, ChatMessage,
    RouteCacheResponseMeta, LatencyBreakdown,
)
from app.cache.models import RequestContext, Provenance
from app.cache.cache_manager import cache_zone, CacheManager
from app.verifier.nli import NLIVerifier
# Initialize cache manager singleton
cache_manager = CacheManager()
verifier = NLIVerifier()
router = APIRouter()

# ── Load config and build adapters once at import time ──
config = load_config()
large_adapter = RetryingAdapter(GroqAdapter(config.models["large"]))
small_adapter = RetryingAdapter(GroqAdapter(config.models["small"]))


@router.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completion endpoint.

    Step 4 behavior: always forwards to the large model (baseline 1).
    Later steps add cache, verification, risk, and routing here.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # ── Step 0: Validate ──
    if not request.messages:
        raise HTTPException(status_code=400, detail={
            "error": {"type": "invalid_request", "message": "messages list is empty"}
        })

    query_text = request.messages[-1].content  # last user message

    # ── Build context for cache key matching (FR-2.4) ──
    context = RequestContext(
        system_prompt=next(
            (m.content for m in request.messages if m.role == "system"), None
        ),
        temperature=request.temperature,
        model_override=request.x_routecache.force_model,
        bypass_cache=request.x_routecache.bypass_cache,
    )

    # ── Cache lookup (Step 7) ──
    candidate = None
    zone = None
    embed_ms = 0.0
    faiss_ms = 0.0

    if not context.bypass_cache:
        candidate, embed_ms, faiss_ms = cache_manager.lookup(query_text, context)
        if candidate:
            print(f"DEBUG: Found candidate for '{query_text[:40]}' - similarity={candidate.similarity:.4f}")

    if candidate:
        zone = cache_zone(
            candidate.similarity,
            config.cache.t_low,
            config.cache.t_high,
        )
        print(f"DEBUG: Zone={zone} (sim={candidate.similarity:.4f}, t_low={config.cache.t_low}, t_high={config.cache.t_high})")
    else:
        zone = "MISS"
        print(f"DEBUG: No candidate found (empty index or no match)")

        # In the route handler, after cache lookup determines zone == "VERIFY":

    if zone == "SERVE":
        # High similarity — serve directly, no verification needed
        answer_text = candidate.answer
        served_from_cache = True
        cache_status = "hit"

    elif zone == "VERIFY":
        # Borderline similarity — run verifier before serving
        verifier_result = verifier.check(
            new_query=query_text,
            cached_query=candidate.query,
        )
        verify_ms = verifier_result.latency_ms

        if verifier_result.verdict == "SAFE":
            # Verification passed — serve the cached answer
            answer_text = candidate.answer
            served_from_cache = True
            cache_status = "verified_hit"
            # Update entry's verification status in metadata store
            cache_manager.metadata.update_verification_status(
                candidate.entry_id, "safe"
            )
        else:
            # Verification FAILED — do NOT serve cached answer
            # Fall through to model call (same as MISS)
            zone = "VERIFY"  # keep zone for logging
            cache_status = "miss"
            # Mark entry as unsafe
            cache_manager.metadata.update_verification_status(
                candidate.entry_id, "unsafe"
            )
            # ... proceed to model call below ...

    elif zone == "MISS":
        # No cache candidate — call a model
        # ... (existing model call logic from Step 7) ...
        # ── MISS: call a model ──
        model_key = "large"  # Step 10 adds routing
        if context.model_override and context.model_override in config.models:
            model_key = context.model_override

        adapter = large_adapter if model_key == "large" else small_adapter

        messages_dicts = [m.model_dump() for m in request.messages]
        model_start = time.time()
        response = adapter.call(messages_dicts, temperature=request.temperature)
        model_latency_ms = (time.time() - model_start) * 1000

        answer_text = response.text
        cost_usd = response.cost_usd

        # ── Store in cache (FR-2.3) ──
        provenance = Provenance(
            source_model=model_key,
            route_decision=model_key,
            from_cache=False,
            created_at=datetime.utcnow(),
        )
        entry_id = cache_manager.store(
            query=query_text,
            answer=answer_text,
            context=context,
            provenance=provenance,
        )
        print(f"DEBUG: Stored cache entry {entry_id} for query: {query_text[:50]}")

    total_latency = (time.time() - start_time) * 1000
    
    # Set defaults for variables needed in response
    if not served_from_cache:
        baseline_cost = compute_baseline_cost(response.tokens_in, response.tokens_out, config)
    else:
        model_latency_ms = 0.0
        cost_usd = 0.0
        baseline_cost = 0.0
        model_key = "cache"

    # ── Build response ──
    return ChatResponse(
        id=request_id,
        choices=[
            ChatResponseChoice(
                message=ChatMessage(role="assistant", content=answer_text)
            )
        ],
        x_routecache=RouteCacheResponseMeta(
            cache=cache_status if served_from_cache else "miss",
            zone=zone,
            similarity=candidate.similarity if candidate else None,
            route=f"{model_key}-model" if not served_from_cache else "cache",
            cost_usd=cost_usd if not served_from_cache else 0.0,
            baseline_cost_usd=baseline_cost if not served_from_cache else compute_baseline_cost(0, 0, config),
            latency_ms=LatencyBreakdown(
                embed=embed_ms,
                faiss=faiss_ms,
                model=model_latency_ms if not served_from_cache else 0.0,
                total=total_latency,
            ),
        ),
    )


@router.get("/health")
def health():
    """Lightweight health check for container orchestrators."""
    return {"status": "ok", "cache": "disabled", "models": list(config.models.keys())}