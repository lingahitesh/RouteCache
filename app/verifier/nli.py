"""
NLI (Natural Language Inference) verifier — STUB.

This is built AFTER the cross-encoder path is fully working end-to-end
(Build Plan Part C: "Add NLI once the cross-encoder path is fully
working, so the comparison is apples-to-apples").

The NLI verifier uses bidirectional entailment:
  - Forward:  entailment(new_query -> cached_query)
  - Backward: entailment(cached_query -> new_query)
  - Combined: min(forward, backward)
  - SAFE only if BOTH directions hold (Lesson 4, Section 8)

Using min(fwd, bwd) catches the asymmetric case:
  "What is a car?" -> "What is a vehicle?" has high forward entailment
  but low backward — the cached answer about vehicles may not fully
  answer a question specifically about cars.

KEY RISK (SRS Section 9): NLI models are trained on declarative
statement pairs, not question pairs. Transfer to this task is an
empirical question — this is exactly what RQ2 tests.

Traceability: FR-2.6 (C1), RQ2
"""

from app.verifier.base import VerifierResult


class NLIVerifier:
    """
    NLI-based verifier using bidirectional entailment.

    NOT YET IMPLEMENTED — placeholder for Step 8's cross-encoder-first
    approach. Build this once the cross-encoder path passes its
    definition of done.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"):
        raise NotImplementedError(
            "NLI verifier is deferred per Build Plan Part C. "
            "Build the cross-encoder path end-to-end first, then add NLI "
            "so the comparison (FR-2.6) is apples-to-apples."
        )

    def check(self, new_query: str, cached_query: str) -> VerifierResult:
        """
        When implemented, this will:
          1. Run entailment(premise=new_query, hypothesis=cached_query) -> fwd
          2. Run entailment(premise=cached_query, hypothesis=new_query) -> bwd
          3. combined = min(fwd, bwd)   # both directions must hold
          4. verdict = SAFE if combined >= pass_threshold
        """
        raise NotImplementedError