# scripts/test_risk_features.py
"""
Step 9 definition of done:
  Construct 10 adversarial pairs covering all six risk categories.
  Confirm each correctly triggers its intended feature flag.
  Then compare fixed vs adaptive thresholds.
"""

from app.risk.risk import extract, compute_risk_score, resolve_thresholds

# ─── 10 adversarial queries covering all 6 risk categories ───────────
test_queries = [
    # (query, expected_features)
    ("What is the return policy for order 4471?",
     ["numbers", "entities"]),

    ("The population was not 8 million in 2020.",
     ["numbers", "negation", "temporal"]),

    ("Write a Python function def reverse(lst): return lst[::-1]",
     ["code"]),

    ("Calculate the integral of x^2 from 0 to 5.",
     ["math", "numbers"]),

    ("Who is the current CEO of Tesla as of today?",
     ["entities", "temporal"]),

    ("Can I NOT cancel my subscription for account #12345?",
     ["negation", "numbers", "entities"]),

    ("What was the GDP of Japan in 2021 vs 2022?",
     ["numbers", "entities", "temporal"]),

    ("Solve: 3x + 7 = 22",
     ["math", "numbers"]),

    ("```python\nprint('hello')\n``` Explain this code.",
     ["code"]),

    ("Yesterday the stock price was never above $150.",
     ["temporal", "negation", "numbers"]),
]

print("=" * 70)
print("Risk Feature Extraction Tests")
print("=" * 70)

all_passed = True
for query, expected in test_queries:
    features = extract(query)
    active = features.active_flags

    # Check each expected feature is present
    missing = [f for f in expected if f not in active]
    extra = [f for f in active if f not in expected]

    status = "PASS" if not missing else "FAIL"
    if missing:
        all_passed = False

    print(f"\n  [{status}] \"{query[:60]}...\"")
    print(f"         Expected: {expected}")
    print(f"         Got:      {active}")
    if missing:
        print(f"         MISSING:  {missing}")
    if extra:
        print(f"         Extra:    {extra}  (not a failure, just unexpected)")
    print(f"         Latency:  {features.latency_ms:.3f}ms")

# ─── Fixed vs Adaptive threshold comparison ──────────────────────────
print(f"\n{'=' * 70}")
print("Fixed vs Adaptive Threshold Comparison")
print("=" * 70)

base_low, base_high = 0.82, 0.93

for query, expected in test_queries:
    features = extract(query)
    risk_score = compute_risk_score(features)

    fixed_low, fixed_high = resolve_thresholds(
        features, risk_score, base_low, base_high, "fixed"
    )
    adaptive_low, adaptive_high = resolve_thresholds(
        features, risk_score, base_low, base_high, "adaptive", adaptive_slope=0.05
    )

    differs = "DIFFERS" if adaptive_low != fixed_low else "SAME"
    print(f"\n  [{differs}] risk={risk_score:.2f}  \"{query[:50]}...\"")
    print(f"         Fixed:    t_low={fixed_low:.3f}  t_high={fixed_high:.3f}")
    print(f"         Adaptive: t_low={adaptive_low:.3f}  t_high={adaptive_high:.3f}")

if all_passed:
    print(f"\n\nAll feature extraction tests PASSED.")
else:
    print(f"\n\nSome tests FAILED — review patterns above.")