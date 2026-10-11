# scripts/test_verifier.py
"""
Step 8 definition of done:
  Run a product-123-vs-product-124 style adversarial pair through
  the pipeline and confirm the verifier actually REJECTS it.

  These are the kinds of queries where surface similarity is HIGH
  (same words, same structure) but the correct answer DIFFERS.
  A cache without verification serves these wrong — silently.
"""

from app.verifier import create_verifier

verifier = create_verifier()
print(f"Verifier: {verifier.model_name}, threshold: {verifier.pass_threshold}")

# ─── Test pairs ───────────────────────────────────────────────────────
# (query_a, query_b, expected_verdict)
# "should_match" pairs: same meaning, verifier should say SAFE
# "should_reject" pairs: different meaning despite surface similarity

test_pairs = [
    # ── Should match (true paraphrases) ──
    (
        "How do I reverse a list in Python?",
        "How can I reverse a Python list?",
        "SAFE",
        "true paraphrase"
    ),
    (
        "What is the capital of France?",
        "What's the capital city of France?",
        "SAFE",
        "true paraphrase"
    ),
    (
        "Explain machine learning simply.",
        "What is machine learning, explained in simple terms?",
        "SAFE",
        "true paraphrase"
    ),

    # ── Should REJECT (adversarial near-duplicates) ──
    (
        "What is the return policy for order 123?",
        "What is the return policy for order 124?",
        "UNSAFE",
        "entity swap (order number)"
    ),
    (
        "What is the population of Tokyo in 2020?",
        "What is the population of Tokyo in 2021?",
        "UNSAFE",
        "number swap (year)"
    ),
    (
        "Is Python 3.11 faster than 3.10?",
        "Is Python 3.10 faster than 3.11?",
        "UNSAFE",
        "order swap (comparison direction)"
    ),
    (
        "Can I return this item?",
        "Can I not return this item?",
        "UNSAFE",
        "negation insertion"
    ),
    (
        "What is the GDP of Japan?",
        "What is the GDP of China?",
        "UNSAFE",
        "entity swap (country)"
    ),
]

passed = 0
failed = 0

for query_a, query_b, expected, description in test_pairs:
    result = verifier.check(query_a, query_b)
    status = "PASS" if result.verdict == expected else "FAIL"

    if status == "PASS":
        passed += 1
    else:
        failed += 1

    print(f"  [{status}] {description}")
    print(f"         \"{query_a[:50]}\"")
    print(f"      vs \"{query_b[:50]}\"")
    print(f"         Score: {result.score:.4f}  Verdict: {result.verdict}  "
          f"(expected: {expected})  Latency: {result.latency_ms:.0f}ms")
    print()

print(f"Results: {passed} passed, {failed} failed out of {len(test_pairs)}")
print()

if failed > 0:
    print("WARNING: Some pairs did not match expected verdicts.")
    print("This may mean:")
    print("  1. The pass_threshold needs tuning (most likely)")
    print("  2. The cross-encoder model isn't strong enough for these cases")
    print("  3. The expected verdict was wrong (check manually)")
    print()
    print("ACTION: Examine the scores for the failed pairs and adjust")
    print("verifier.pass_threshold in config.yaml accordingly.")
else:
    print("All pairs matched expected verdicts. Step 8: PASSED")