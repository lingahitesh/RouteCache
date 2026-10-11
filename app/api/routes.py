"""
API route handlers for RouteCache.

Step 4 starts with a bare pass-through to the large model.
Steps 6-11 progressively add cache lookup, verification, risk
extraction, routing, and the joint decision engine.
Step 12 adds graceful degradation: cache / verifier / router / provider
failures are caught and routed to a safe fallback instead of crashing
the request (NFR-5).

Step 14.6: Full unified pipeline with all components wired together.

Traceability: FR-1.1, FR-1.2, NFR-5
Build Plan: Step 4, Step 12, Step 14.6
"""

import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from config.config import load_config
from app.adapters.adapters import GroqAdapter, RetryingAdapter
from app.cost import compute_baseline_cost
from app.api.models import (
    ChatRequest, ChatResponse, ChatResponseChoice, ChatMessage,
    RouteCacheResponseMeta, LatencyBreakdown, VerifierInfo,
)
from app.cache.models import RequestContext, Provenance
from app.cache.cache_manager import cache_zone, CacheManager
from app.risk.risk import extract, compute_risk_score, resolve_thresholds
from app.router.features import extract_router_features
from app.decision import create_decision_policy
from app.decision.base import Action, DecisionTrace
from app.logging_utils import log_request
from app.errors import (
    VerifierUnavailableError, RouterUnavailableError, ProviderError,
)
from app.metrics.recorder import metrics_recorder

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

    Step 14.6 pipeline, Step 12 degradation wrapping each fallible stage:
      1. Parse + validate (Step 4)
      2. Risk extraction (Step 9) -- fast regex, not wrapped: a regex
         failure here is a code bug, not an external dependency failing,
         so NFR-5's "degrade gracefully" doesn't apply to it the way it
         does to the cache/verifier/router/provider calls below.
      3. Threshold resolution (Step 9, C2)
      4. Cache lookup (Step 7) -- WRAPPED: failure -> treat as MISS
      5. Zone determination (Step 7)
      6. Router feature extraction (Step 10)
      7. Joint decision (Step 11, C3) -- WRAPPED: verifier/router failure
         -> Action.LARGE (never serve an unverified candidate just
         because verification itself broke)
      8. Execute decision -- WRAPPED: provider failure -> try the other
         model before giving up
      9. Store on miss (Step 7) -- WRAPPED: a failed cache write must
         not lose the answer the user is about to receive
      10. Log everything (Step 5)
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # ── Step 0: Validate ──
    if not request.messages:
        raise HTTPException(status_code=400, detail={
            "error": {"type": "invalid_request", "message": "messages list is empty"}
        })

    # ── Step 0b: Validate x_routecache.force_model, if given (FR-1.2) ──
    # Only "small"/"large" are valid: those are the only two keys with a
    # constructed adapter in this module. "judge1"/"judge2" are real
    # config.models keys (used by scripts/generate_router_labels.py) but
    # have no adapter wired here — accepting them would previously have
    # silently called large_adapter while logging the judge's name, a
    # latent mismatch this validation also closes off.
    force_model = request.x_routecache.force_model
    if force_model and force_model not in ("small", "large"):
        raise HTTPException(status_code=400, detail={
            "error": {"type": "invalid_request",
                      "message": f"Unknown model '{force_model}' for "
                                 f"x_routecache.force_model. Valid values: "
                                 f"'small', 'large'."}
        })

    query_text = request.messages[-1].content  # last user message

    # ── Build context for cache key matching (FR-2.4) ──
    context = RequestContext(
        system_prompt=next(
            (m.content for m in request.messages if m.role == "system"), None
        ),
        temperature=request.temperature,
        model_override=force_model,
        bypass_cache=request.x_routecache.bypass_cache,
    )

    # Latency tracking
    embed_ms = faiss_ms = 0.0

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

    candidate = None
    zone = "MISS"
    decision_ms = 0.0

    if force_model:
        # FR-1.4 / Design Doc §6.1: force_model "skips routing entirely" —
        # no cache lookup, no verifier, no router. zone=None (not "MISS")
        # because no lookup was even attempted; "MISS" would misleadingly
        # imply one happened and found nothing.
        trace = DecisionTrace(
            action=Action.SMALL if force_model == "small" else Action.LARGE,
            zone=None, similarity=None, risk_score=risk_score,
            risk_flags=risk_features.active_flags,
        )
        zone = None

    else:
        # ── Step 7: Cache lookup — WRAPPED (NFR-5) ──
        if not context.bypass_cache:
            try:
                candidate, embed_ms, faiss_ms = cache_manager.lookup(query_text, context)
                if candidate:
                    zone = cache_zone(candidate.similarity, effective_t_low, effective_t_high)
            except Exception as e:
                # Any cache-layer failure (embedding, FAISS, SQLite) -> proceed
                # as if this were a MISS, same as the DDD §4.8 failure path.
                metrics_recorder.record_failure("cache", fallback=True)
                candidate = None
                zone = "MISS"
                logger.warning(f"[{request_id}] Cache unavailable: {e}. Proceeding as MISS.")

        # ── Step 10: Extract router features ──
        router_features = extract_router_features(query_text, risk_features)

        # ── Step 11: Joint decision (C3) — WRAPPED (NFR-5) ──
        decision_start = time.time()
        try:
            trace = decision_policy.decide(
                candidate=candidate,
                risk_features=risk_features,
                router_features=router_features,
                zone=zone,
                risk_score=risk_score,
                effective_t_low=effective_t_low,
                effective_t_high=effective_t_high,
            )
        except VerifierUnavailableError as e:
            metrics_recorder.record_failure("verifier", fallback=True)
            logger.warning(f"[{request_id}] Verifier unavailable: {e}. Falling back to LARGE.")
            trace = DecisionTrace(
                action=Action.LARGE, zone=zone,
                similarity=candidate.similarity if candidate else None,
                risk_score=risk_score, risk_flags=risk_features.active_flags,
            )
        except RouterUnavailableError as e:
            metrics_recorder.record_failure("router", fallback=True)
            logger.warning(f"[{request_id}] Router unavailable: {e}. Falling back to LARGE.")
            trace = DecisionTrace(
                action=Action.LARGE, zone=zone,
                similarity=candidate.similarity if candidate else None,
                risk_score=risk_score, risk_flags=risk_features.active_flags,
            )
        except Exception as e:
            # Unknown decision-engine failure — same safe default. Less
            # specific than the two branches above (metrics/logs just say
            # "decision" rather than naming the component), but still
            # correct: the candidate is never served on an error.
            metrics_recorder.record_failure("decision", fallback=True)
            logger.error(f"[{request_id}] Decision engine error: {e}. Falling back to LARGE.")
            trace = DecisionTrace(
                action=Action.LARGE, zone=zone,
                similarity=candidate.similarity if candidate else None,
                risk_score=risk_score, risk_flags=risk_features.active_flags,
            )
        decision_ms = (time.time() - decision_start) * 1000

    # ── Execute the decision ──
    answer_text = None
    cost_usd = 0.0
    model_latency_ms = 0.0
    tokens_in = 0
    tokens_out = 0
    model_used = None
    model_key = None

    if trace.action == Action.SERVE:
        # Serve from cache.
        answer_text = candidate.answer
        model_key = "cache"
        model_used = f"cache:{candidate.provenance.source_model}"

    elif trace.action in (Action.SMALL, Action.LARGE):
        # When force_model was set, trace.action already reflects it
        # (built directly above) — no separate override check needed
        # here anymore, and none would be safe: config.models also has
        # "judge1"/"judge2" keys with no adapter constructed in this
        # module, which the old override check didn't guard against.
        model_key = "small" if trace.action == Action.SMALL else "large"
        adapter = small_adapter if model_key == "small" else large_adapter
        messages_dicts = [m.model_dump() for m in request.messages]

        try:
            model_start = time.time()
            response = adapter.call(messages_dicts, temperature=request.temperature)
            model_latency_ms = (time.time() - model_start) * 1000

            answer_text = response.text
            cost_usd = response.cost_usd
            tokens_in = response.tokens_in
            tokens_out = response.tokens_out
            model_used = config.models[model_key].name

        except ProviderError as e:
            # Primary model failed after its own retries — try the other
            # model once before giving up (FR-5.3's "fallback to another
            # model"). Both Groq-backed adapters raise ProviderError on
            # terminal failure (see app/adapters/adapters.py).
            logger.warning(f"[{request_id}] {model_key} model failed: {e}. Trying fallback.")
            fallback_key = "large" if model_key == "small" else "small"
            fallback_adapter = large_adapter if fallback_key == "large" else small_adapter

            try:
                model_start = time.time()
                response = fallback_adapter.call(messages_dicts, temperature=request.temperature)
                model_latency_ms = (time.time() - model_start) * 1000

                answer_text = response.text
                cost_usd = response.cost_usd
                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                model_key = fallback_key
                model_used = config.models[fallback_key].name
                metrics_recorder.record_failure(f"provider_{model_key}", fallback=True)

            except ProviderError as fallback_err:
                logger.error(
                    f"[{request_id}] Both models failed. "
                    f"Primary: {e}. Fallback: {fallback_err}."
                )
                metrics_recorder.record_failure("provider_both", fallback=False)
                raise HTTPException(status_code=502, detail={
                    "error": {"type": "model_unavailable",
                              "message": "All model providers are unreachable."}
                })

        # ── Store in cache (FR-2.3) — WRAPPED ──
        # A failed cache write must not cost the user the answer they
        # already paid for; log and move on.
        try:
            provenance = Provenance(
                source_model=model_key,
                route_decision=model_key,
                from_cache=False,
                created_at=datetime.now(timezone.utc),
            )
            cache_manager.store(
                query=query_text,
                answer=answer_text,
                context=context,
                provenance=provenance,
                has_temporal_risk=risk_features.has_temporal,
            )
        except Exception as store_err:
            metrics_recorder.record_failure("cache_store", fallback=True)
            logger.warning(f"[{request_id}] Cache store failed: {store_err}")

    total_latency = (time.time() - start_time) * 1000

    # ── Baseline cost for comparison (NFR-2) ──
    # A real model call: use its actual token counts. A cache hit: we
    # don't yet store token counts on CacheEntry (see audit §2.1 / Open
    # Decision 2), so compute_baseline_cost(0, 0, ...) would report $0
    # savings for every hit, which contradicts the SRS §4.2 example.
    # Using the same 500/200 estimate already used as a rough constant
    # elsewhere (app/decision/expected_cost.py) is a stopgap, not the
    # final fix — it gives hits a non-zero, non-misleading baseline
    # without inventing a schema change inside this merge.
    if trace.action != Action.SERVE:
        baseline_cost = compute_baseline_cost(tokens_in, tokens_out, config)
    else:
        baseline_cost = compute_baseline_cost(500, 200, config)

    # ── cache status for the response: distinguish a direct SERVE-zone
    # hit from one that was verified first (SRS §4.2, Design Doc §6.2) ──
    if trace.action == Action.SERVE:
        cache_status = "verified_hit" if zone == "VERIFY" else "hit"
    else:
        cache_status = "miss"

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
            thresholds=(
                {"t_low": effective_t_low, "t_high": effective_t_high}
                if effective_t_low is not None else None
            ),
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

    # ── Log the request (Step 5, Section 8.6) ──
    log_request(
        request_id=request_id,
        query_text=query_text,
        cache_zone=zone,
        similarity=candidate.similarity if candidate else None,
        risk_features=risk_features.to_dict(),
        risk_score=risk_score,
        thresholds={"t_low": effective_t_low, "t_high": effective_t_high},
        verifier_verdict=trace.verifier_result.verdict if trace.verifier_result else None,
        verifier_score=trace.verifier_result.score if trace.verifier_result else None,
        route=f"{model_key}-model" if model_key != "cache" else "cache",
        router_score=trace.router_score,
        served_entry_id=candidate.entry_id if (trace.action == Action.SERVE and candidate) else None,
        model_used=model_used,
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
    return {"status": "ok", "cache": "disabled", "models": list(config.models.keys())}