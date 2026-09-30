"""
RouteCache — FastAPI application entry point.

Run with: uvicorn app.main:app --reload --port 8000
Test with: curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

Traceability: FR-1.1
Build Plan: Step 4
"""

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="RouteCache",
    description="Verified semantic caching and risk-adaptive routing for LLM inference",
    version="0.1.0",
)

app.include_router(router)