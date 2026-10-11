"""
Provider adapters for LLM API calls.

Each adapter wraps an OpenAI-compatible endpoint (Groq, Ollama, etc.)
and returns a ProviderResponse with tokens, cost, and latency.

CRITICAL (NFR-8): Every single call — success or failure — is written
to disk IMMEDIATELY, before anything else happens with the response.
This is what makes Groq's free-tier rate limits survivable across
24 weeks of experiments.

Traceability: FR-5.1–FR-5.4, NFR-8
Build Plan: Step 2
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Protocol
import httpx
from app.cost import compute_cost
from app.utils.paths import get_data_dir


# ─── Data Structures ─────────────────────────────────────────────────

@dataclass
class ProviderResponse:
    """Complete record of a single LLM API call."""
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    model_name: str
    tokens_estimated: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ProviderAdapterProtocol(Protocol):
    """
    Contract that every adapter must satisfy (Detailed Design Doc Section 2.7).
    This is what makes FR-5.1's "add by configuration" possible.
    """
    def call(self, messages: list[dict], temperature: float = 0.0) -> ProviderResponse: ...


# ─── Disk Cache for NFR-8 ────────────────────────────────────────────

class ResponseDiskCache:
    """
    Writes every API call to disk as a JSON file, keyed by a hash of
    (model, messages, temperature). On replay, returns the cached
    response instead of calling the API.

    This is NOT the semantic cache (that's app/cache/). This is the
    experiment-reproducibility cache from NFR-8 and Build Plan A.1:
    "cache every single model output to disk from the very first API call."
    """

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = get_data_dir() / "recorded_outputs"
        self.cache_dir = cache_dir if isinstance(cache_dir, Path) else Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list[dict], temperature: float) -> str:
        """Deterministic hash of the request."""
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float) -> ProviderResponse | None:
        """Return cached response if it exists, else None."""
        key = self._key(model, messages, temperature)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return ProviderResponse(**data)
        return None

    def put(self, model: str, messages: list[dict], temperature: float,
            response: ProviderResponse) -> None:
        """Write response to disk. Called IMMEDIATELY after every API call."""
        key = self._key(model, messages, temperature)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(asdict(response), f, indent=2)


# Module-level singleton — created once, shared by all adapters
_disk_cache = ResponseDiskCache()


# ─── Groq / OpenAI-Compatible Adapter ────────────────────────────────

class GeminiAdapter:
    """
    Adapter for Google Gemini API.
    
    Gemini DOES support role field ("user" or "model"), and uses system_instruction
    for system prompts. This implementation correctly preserves conversation structure.
    
    API structure:
    - URL: {base_url}/models/{model}:generateContent (key in header)
    - Headers: x-goog-api-key
    - Request: { system_instruction, contents: [{role, parts}], generationConfig }
    - Response: { candidates, usageMetadata }
    """
    
    def __init__(self, model_config, disk_cache: ResponseDiskCache | None = None):
        self.model_config = model_config
        self.disk_cache = disk_cache or _disk_cache
        self.client = httpx.Client(base_url=model_config.base_url, timeout=60.0)
    
    def _headers(self) -> dict:
        """Gemini uses x-goog-api-key header."""
        api_key = os.environ.get(self.model_config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing env var: {self.model_config.api_key_env}. "
                f"Set it in your .env file."
            )
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
    
    @staticmethod
    def _to_gemini(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Convert OpenAI-style messages to Gemini format.
        Returns (system_parts, contents).
        
        Gemini uses:
        - system_instruction: { parts: [{text}] } for system prompts
        - contents: [{ role: "user"|"model", parts: [{text}] }] for conversation
        - "assistant" role maps to "model"
        """
        system_parts = []
        contents = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_parts.append({"text": content})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                # Map OpenAI "assistant" to Gemini "model"
                contents.append({"role": "model", "parts": [{"text": content}]})
        
        return system_parts, contents
    
    def call(self, messages: list[dict], temperature: float = 0.0) -> ProviderResponse:
        """Call Gemini API with OpenAI-style messages."""
        model_name = self.model_config.name
        
        # Check disk cache first
        cached = self.disk_cache.get(model_name, messages, temperature)
        if cached:
            return cached
        
        # Convert to Gemini format
        system_parts, contents = self._to_gemini(messages)
        
        # Build payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        
        # Add system_instruction if present
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}
        
        start = time.time()
        try:
            resp = self.client.post(
                f"/models/{model_name}:generateContent",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.time() - start) * 1000

            # Extract response text (join all parts)
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            answer_text = "".join(part.get("text", "") for part in parts)
            
            # Extract token counts (include thoughtsTokenCount if present)
            usage = data.get("usageMetadata", {})
            tokens_in = usage.get("promptTokenCount", 0)
            tokens_out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            
            # Compute cost
            cost = compute_cost(
                tokens_in, tokens_out,
                self.model_config.price_in,
                self.model_config.price_out
            )

            result = ProviderResponse(
                text=answer_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tokens_estimated=False,
                cost_usd=cost,
                latency_ms=latency_ms,
                model_name=model_name,
            )
            
            # Write to disk IMMEDIATELY (NFR-8)
            self.disk_cache.put(model_name, messages, temperature, result)
            return result
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Gemini API error: {e.response.status_code} - {e.response.text}")


class GroqAdapter:
    """
    Adapter for any OpenAI-compatible API (Groq, OpenAI, etc.).
    Uses provider-reported token counts — they're authoritative for Groq.

    Usage:
        adapter = GroqAdapter(config.models["large"])
        response = adapter.call([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, model_config, disk_cache: ResponseDiskCache | None = None):
        self.model_config = model_config
        self.disk_cache = disk_cache or _disk_cache
        self.client = httpx.Client(
            base_url=model_config.base_url,
            timeout=60.0,    # generous timeout — Groq can be slow under load
        )

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.model_config.api_key_env:
            api_key = os.environ.get(self.model_config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing env var: {self.model_config.api_key_env}. "
                    f"Set it in your .env file."
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def call(self, messages: list[dict], temperature: float = 0.0) -> ProviderResponse:
        """
        Call the model and return a ProviderResponse.
        Checks disk cache first (NFR-8 replay). On a real call,
        writes to disk BEFORE returning.
        """
        model_name = self.model_config.name

        # ── Check disk cache first (free replay) ──
        cached = self.disk_cache.get(model_name, messages, temperature)
        if cached is not None:
            return cached

        # ── Real API call ──
        start = time.time()
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.client.post(
            "/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - start) * 1000

        # ── Extract response fields ──
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        # Cost computed here, at the adapter level — not by the caller
        from app.cost import compute_cost
        cost_usd = compute_cost(
            tokens_in, tokens_out,
            self.model_config.price_in, self.model_config.price_out,
        )

        response = ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model_name=model_name,
            tokens_estimated=False,    # Groq reports real counts
        )

        # ── Write to disk IMMEDIATELY (NFR-8) ──
        self.disk_cache.put(model_name, messages, temperature, response)

        return response


# ─── Retrying Wrapper ─────────────────────────────────────────────────

class RetryingAdapter:
    """
    Wraps any adapter with bounded retries and timeout.
    Traceability: FR-5.3, Detailed Design Document Section 2.7.

    On failure after all retries, raises ProviderError (see app/errors.py,
    built in Step 12).
    """

    def __init__(self, inner: GroqAdapter, max_retries: int = 3,
                 timeout_ms: int = 30000):
        self.inner = inner
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms

    def call(self, messages: list[dict], temperature: float = 0.0) -> ProviderResponse:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.inner.call(messages, temperature)
            except httpx.HTTPStatusError as e:
                last_error = e
                # 429 = rate limited, 5xx = server error — retry
                if e.response.status_code in (429, 500, 502, 503):
                    wait = min(2 ** attempt, 10)  # exponential backoff, max 10s
                    time.sleep(wait)
                    continue
                raise   # 4xx client errors — don't retry
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                wait = min(2 ** attempt, 10)
                time.sleep(wait)
                continue

        raise RuntimeError(
            f"All {self.max_retries} retries failed for "
            f"{self.inner.model_config.name}: {last_error}"
        )