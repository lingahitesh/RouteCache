"""
Verifier protocol — the contract both implementations must satisfy.

Both CrossEncoderVerifier and NLIVerifier implement this protocol,
which is what makes the FR-2.6 comparison a one-line config change
(verifier.type in config.yaml) rather than two separate code paths.

Traceability: FR-2.6 (C1)
Build Plan: Step 8
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class VerifierResult:
    """
    The output of a verification check.

    verdict:  "SAFE" means the cached answer is safe to serve for the new query.
              "UNSAFE" means it's not — fall through to model call.
    score:    The raw similarity/entailment score (for logging and threshold tuning).
    detail:   Additional scores for debugging. Cross-encoder has one score;
              NLI has forward and backward entailment scores.
    latency_ms: How long verification took (for NFR-1a reporting).
    """
    verdict: Literal["SAFE", "UNSAFE"]
    score: float
    detail: dict[str, float]
    latency_ms: float = 0.0


class Verifier(Protocol):
    """
    Protocol that every verifier must implement.

    The system never calls a concrete verifier class — it calls this
    protocol. Swap implementations by changing config.verifier.type.
    """
    def check(self, new_query: str, cached_query: str) -> VerifierResult:
        """
        Check whether a cached query is semantically equivalent to a new query.

        Args:
            new_query:    The incoming user query.
            cached_query: The query text of the candidate cache entry.

        Returns:
            VerifierResult with verdict, score, and latency.
        """
        ...