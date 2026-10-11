"""
API route handlers for RouteCache.

Step 4 starts with a bare pass-through to the large model.
Steps 6-11 progressively add cache lookup, verification, risk
extraction, routing, and the joint decision engine.

Step 14.6: Full unified pipeline with all components wired together.

Traceability: FR-1.1, FR-1.2
Build Plan: Step 4, Step 14.6
"""

import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime
from config.config import load_config
from app.adapters.adapters import GroqAdapter, RetryingAdapter
from app.cost import compute_cost_for_model, compute_baseline_cost
from app.api.models import (
    ChatRequest, ChatResponse, ChatResponseChoice, ChatMessage,
    RouteCacheResponseMeta, LatencyBreakdown, VerifierInfo,
)
from app.cache.models import RequestContext, Provenance
from app.cache.cache_manager import cache_zone, CacheManager
from app.risk.risk import extract, compute_risk_score, resolve_thresholds
from app.router.features import extract_router_features
from app.decision import create_decision_policy
from app.decision.base import Action
from app.logging_utils import log_request

logger = logging.getLogger(__name__)

cache_manager = CacheManager()
decision_policy = create_decision_policy()
router = APIRouter()

# ── Load config and build adapters once at import time ──
config = load_config()
large_adapter = RetryingAdapter(GroqAdapter(config.models["large"]))
small_adapter = RetryingAdapter(GroqAdapter(config.models["small"]))


@router.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completion endpoint.

    Step 14.6: Full unified pipeline with all components wired together:
      1. Parse + validate (Step 4)
      2. Risk extraction (Step 9)
      3. Threshold resolution (Step 9, C2)
      4. Cache lookup (Step 7)
      5. Zone determination (Step 7)
      6. Router feature extraction (Step 10)
      7. Joint decision (Step 11, C3)
      8. Execute decision
      9. Store on miss (Step 7)
      10. Log everything (Step 5)
    """
    import sys
    print("="*80, file=sys.stderr, flush=True)
    print("REQUEST RECEIVED", file=sys.stderr, flush=True)
    print("="*80, file=sys.stderr, flush=True)
    
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

    # ── Step 9: Extract risk features ──
    risk_start = time.time()
    risk_features = extract(query_text)
    risk_score = compute_risk_score(risk_features)
    risk_ms = (time.time() - risk_start) * 1000

    # ── Step 9: Resolve thresholds (C2) ──
    effective_t_low, effective_t_high = resolve_thresholds(
        features=risk_features,
        risk_score=risk_score,
        base_t_low=config.cache.t_low,
        base_t_high=config.cache.t_high,
        policy=config.risk.policy,
        adaptive_slope=config.risk.adaptive_slope,
    )
    
    logger.warning("="*80)
    logger.warning(f"REQUEST: policy={config.risk.policy}, query={query_text[:50]}")
    
    print(f"DEBUG THRESHOLDS: policy={config.risk.policy}, risk_score={risk_score:.3f}, "
          f"base=({config.cache.t_low:.3f}, {config.cache.t_high:.3f}), "
          f"effective=({effective_t_low:.3f}, {effective_t_high:.3f}), "
          f"risk_flags={risk_features.active_flags}", flush=True)

    # ── Step 7: Cache lookup ──
    candidate = None
    embed_ms = 0.0
    faiss_ms = 0.0
    zone = "MISS"

    if not context.bypass_cache:
        candidate, embed_ms, faiss_ms = cache_manager.lookup(query_text, context)
        if candidate:
            # ── Step 7: Determine zone ──
            zone = cache_zone(candidate.similarity, effective_t_low, effective_t_high)
            print(f"DEBUG: Found candidate for '{query_text[:60]}' - similarity={candidate.similarity:.4f}, zone={zone}, cached_query='{candidate.query[:60]}'", flush=True)

    # ── Step 10: Extract router features ──
    router_features = extract_router_features(query_text, risk_features)

    # ── Step 11: Joint decision (C3) ──
    decision_start = time.time()
    trace = decision_policy.decide(
        candidate=candidate,
        risk_features=risk_features,
        router_features=router_features,
        zone=zone,
        risk_score=risk_score,
        effective_t_low=effective_t_low,
        effective_t_high=effective_t_high,
    )
    decision_ms = (time.time() - decision_start) * 1000

    # ── Execute the decision ──
    answer_text = None
    cost_usd = 0.0
    model_latency_ms = 0.0
    tokens_in = 0
    tokens_out = 0
    cache_status = "miss"
    model_key = None

    if trace.action == Action.SERVE:
        # Serve from cache
        answer_text = candidate.answer
        cache_status = "hit"
        model_key = "cache"
        print(f"DEBUG: SERVE from cache (entry {candidate.entry_id})")

    elif trace.action in (Action.SMALL, Action.LARGE):
        # Call the chosen model
        model_key = "small" if trace.action == Action.SMALL else "large"
        
        # Check for override
        if context.model_override and context.model_override in config.models:
            model_key = context.model_override

        adapter = small_adapter if model_key == "small" else large_adapter

        messages_dicts = [m.model_dump() for m in request.messages]
        model_start = time.time()
        response = adapter.call(messages_dicts, temperature=request.temperature)
        model_latency_ms = (time.time() - model_start) * 1000

        answer_text = response.text
        cost_usd = response.cost_usd
        tokens_in = response.tokens_in
        tokens_out = response.tokens_out

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
        print(f"DEBUG: Stored cache entry {entry_id} after {model_key} model call")

    total_latency = (time.time() - start_time) * 1000

    # ── Compute baseline cost for comparison ──
    if trace.action != Action.SERVE:
        baseline_cost = compute_baseline_cost(tokens_in, tokens_out, config)
    else:
        baseline_cost = compute_baseline_cost(0, 0, config)

    # ── Build response with full metadata ──
    resp = ChatResponse(
        id=request_id,
        choices=[
            ChatResponseChoice(
                message=ChatMessage(role="assistant", content=answer_text)
            )
        ],
        x_routecache=RouteCacheResponseMeta(
            cache=cache_status,
            zone=zone,
            similarity=candidate.similarity if candidate else None,
            risk_flags=risk_features.active_flags,
            thresholds={"t_low": effective_t_low, "t_high": effective_t_high} if effective_t_low else None,
            verifier=VerifierInfo(
                type=config.verifier.type,
                verdict=trace.verifier_result.verdict if trace.verifier_result else None,
                score=trace.verifier_result.score if trace.verifier_result else None,
            ),
            decision=trace.action.value,
            route=f"{model_key}-model" if model_key != "cache" else "cache",
            router_score=trace.router_score,
            cost_usd=cost_usd,
            baseline_cost_usd=baseline_cost,
            latency_ms=LatencyBreakdown(
                embed=embed_ms,
                faiss=faiss_ms,
                verify=trace.verifier_result.latency_ms if trace.verifier_result else None,
                route=decision_ms,
                model=model_latency_ms,
                total=total_latency,
            ),
        ),
    )
    
    # Add debug info for troubleshooting
    logger.warning(f"Response: zone={zone}, action={trace.action}, policy={config.risk.policy}, "
                   f"risk_score={risk_score:.3f}, t_low={effective_t_low:.3f}, t_high={effective_t_high:.3f}")
    
    # ── Log the request (Step 5, Section 8.6) ──
    log_request(
        request_id=request_id,
        query_text=query_text,
        cache_zone=zone,
        similarity=candidate.similarity if candidate else None,
        risk_features={"flags": risk_features.active_flags, "score": risk_score},
        risk_score=risk_score,
        thresholds={"t_low": effective_t_low, "t_high": effective_t_high},
        verifier_verdict=trace.verifier_result.verdict if trace.verifier_result else None,
        verifier_score=trace.verifier_result.score if trace.verifier_result else None,
        route=f"{model_key}-model" if model_key != "cache" else "cache",
        router_score=trace.router_score,
        served_entry_id=candidate.entry_id if (trace.action == Action.SERVE and candidate) else None,
        model_used=config.models[model_key].name if model_key in config.models else None,
        cost_usd=cost_usd,
        baseline_cost_usd=baseline_cost,
        latency_embed_ms=embed_ms,
        latency_faiss_ms=faiss_ms,
        latency_risk_ms=risk_ms,
        latency_verify_ms=trace.verifier_result.latency_ms if trace.verifier_result else None,
        latency_route_ms=decision_ms,
        latency_model_ms=model_latency_ms,
        latency_total_ms=total_latency,
    )
    
    return resp


@router.get("/health")
def health():
    """Lightweight health check for container orchestrators."""
    import sys
    print("HEALTH CHECK HIT", file=sys.stderr, flush=True)
    return {"status": "ok", "cache": "disabled", "models": list(config.models.keys())}