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

from config.config import load_config
from app.adapters.adapters import GroqAdapter, RetryingAdapter
from app.cost import compute_cost_for_model, compute_baseline_cost
from app.api.models import (
    ChatRequest, ChatResponse, ChatResponseChoice, ChatMessage,
    RouteCacheResponseMeta, LatencyBreakdown,
)

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

    # ── Force model override (FR-1.4) ──
    if request.x_routecache.force_model:
        model_key = request.x_routecache.force_model
        if model_key not in config.models:
            raise HTTPException(status_code=400, detail={
                "error": {"type": "invalid_request",
                          "message": f"Unknown model: {model_key}. Available: {list(config.models.keys())}"}
            })
    else:
        model_key = "large"   # Step 4: always-large baseline

    # ── Select adapter ──
    if model_key == "small":
        adapter = small_adapter
    else:
        adapter = large_adapter

    # ── Call the model ──
    messages_dicts = [m.model_dump() for m in request.messages]
    model_start = time.time()
    response = adapter.call(messages_dicts, temperature=request.temperature)
    model_latency = (time.time() - model_start) * 1000

    # ── Compute costs ──
    cost_usd = response.cost_usd
    baseline_cost = compute_baseline_cost(
        response.tokens_in, response.tokens_out, config
    )

    total_latency = (time.time() - start_time) * 1000

    # ── Log the request (Step 5) ──
    from app.logging_utils import log_request
    log_request(
        request_id=request_id,
        query_text=request.messages[-1].content,  # last user message
        cache_zone=None,  # no cache yet
        route=f"{model_key}-model",
        model_used=config.models[model_key].name,
        cost_usd=cost_usd,
        baseline_cost_usd=baseline_cost,
        latency_model_ms=model_latency,
        latency_total_ms=total_latency,
    )

    # ── Build response ──
    return ChatResponse(
        id=request_id,
        choices=[
            ChatResponseChoice(
                message=ChatMessage(role="assistant", content=response.text)
            )
        ],
        x_routecache=RouteCacheResponseMeta(
            cache="disabled",                  # no cache yet (Step 7 enables it)
            zone=None,
            route=f"{model_key}-model",
            cost_usd=cost_usd,
            baseline_cost_usd=baseline_cost,
            latency_ms=LatencyBreakdown(
                model=model_latency,
                total=total_latency,
            ),
        ),
    )


@router.get("/health")
def health():
    """Lightweight health check for container orchestrators."""
    return {"status": "ok", "cache": "disabled", "models": list(config.models.keys())}