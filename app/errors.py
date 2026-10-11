"""
RouteCache exception hierarchy.

Every subclass of RouteCacheError except InvalidRequestError is caught
at the API layer and converted into a graceful-degradation path — never
propagated as a 5xx to the client except internal_error.

Safety rule: on verifier or router failure, the fallback is ALWAYS
Action.LARGE, never Action.SERVE. An unverifiable cache candidate is
never served just because verification broke.

Traceability: NFR-5, Detailed Design Doc Section 9
Build Plan: Step 12
"""


class RouteCacheError(Exception):
    """Base exception for all RouteCache errors."""
    component: str = "unknown"


class CacheUnavailableError(RouteCacheError):
    """FAISS index or SQLite metadata store is unreachable."""
    component = "cache"


class VerifierUnavailableError(RouteCacheError):
    """Verifier model failed to load, crashed, or timed out."""
    component = "verifier"


class RouterUnavailableError(RouteCacheError):
    """Router model failed to load or predict."""
    component = "router"


class ProviderError(RouteCacheError):
    """LLM provider API call failed."""
    component = "provider"


class ProviderTimeoutError(ProviderError):
    """LLM provider call timed out after all retries."""
    component = "provider"


class ProviderRateLimitError(ProviderError):
    """LLM provider returned 429 after all retries."""
    component = "provider"


class InvalidRequestError(RouteCacheError):
    """
    Client sent a malformed request.
    This is the ONLY exception that becomes a 4xx — all others
    trigger graceful degradation, not client-visible errors.
    """
    component = "api"