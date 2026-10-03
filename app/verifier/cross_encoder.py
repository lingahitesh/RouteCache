"""
Cross-encoder verifier for borderline cache hits.

Uses a sentence-transformers CrossEncoder model fine-tuned on
paraphrase data (QQP/STS-B). Takes a (query_a, query_b) pair and
produces a single similarity score.

This is the primary verifier — NLI is the comparison variant.

How it works:
  1. Concatenate [new_query, cached_query] as a single input
  2. The cross-encoder processes both simultaneously (unlike bi-encoder
     which processes them separately)
  3. Output: a single score between 0 and 1
  4. If score >= pass_threshold -> SAFE (serve the cached answer)
  5. If score < pass_threshold -> UNSAFE (don't serve, call a model)

Why cross-encoder and not bi-encoder:
  The bi-encoder (Step 6's embedder) already did the fast, approximate
  matching. The cross-encoder is the slow, precise second opinion.
  This two-stage design (fast recall, then precise verification) is the
  "bi-encoder for recall, cross-encoder for precision" pattern from
  Sentence-BERT (Reimers & Gurevych, 2019).

Traceability: FR-2.6 (C1)
Build Plan: Step 8
"""

import time
from sentence_transformers import CrossEncoder as STCrossEncoder

from config.config import load_config
from app.verifier.base import VerifierResult


class CrossEncoderVerifier:
    """
    Cross-encoder paraphrase verifier.

    Usage:
        verifier = CrossEncoderVerifier()
        result = verifier.check("How do I reverse a list?", "How can I reverse a Python list?")
        # result.verdict = "SAFE", result.score = 0.94
    """

    def __init__(self, model_name: str | None = None, pass_threshold: float | None = None):
        """
        Args:
            model_name:     HuggingFace model ID. Defaults to config value.
            pass_threshold: Score above which a pair is considered SAFE.
                            Defaults to config value.

        The model downloads on first use (~50-100MB depending on model).
        Subsequent calls use the cached model.
        """
        config = load_config()
        self.model_name = model_name or config.verifier.model_name
        self.pass_threshold = pass_threshold or config.verifier.pass_threshold

        # Load the cross-encoder model
        # max_length=512 covers most query pairs; longer truncates
        self.model = STCrossEncoder(self.model_name, max_length=512)

    def check(self, new_query: str, cached_query: str) -> VerifierResult:
        """
        Check whether two queries are semantically equivalent.

        The cross-encoder sees both queries at once (concatenated with [SEP]),
        which is why it's more accurate than the bi-encoder — it can attend
        across both queries simultaneously.

        Args:
            new_query:    The incoming user query.
            cached_query: The query from the cache entry.

        Returns:
            VerifierResult:
              - verdict: "SAFE" if score >= pass_threshold, "UNSAFE" otherwise
              - score: raw cross-encoder similarity score
              - detail: {"paraphrase_score": <score>}
              - latency_ms: time taken for the cross-encoder forward pass
        """
        start = time.time()

        # Cross-encoder predict: takes a list of (sentence_a, sentence_b) pairs
        # Returns a list of scores
        scores = self.model.predict([(new_query, cached_query)])
        score = float(scores[0])

        latency_ms = (time.time() - start) * 1000

        verdict = "SAFE" if score >= self.pass_threshold else "UNSAFE"

        return VerifierResult(
            verdict=verdict,
            score=score,
            detail={"paraphrase_score": score},
            latency_ms=latency_ms,
        )