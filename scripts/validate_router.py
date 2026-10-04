# scripts/validate_router.py
"""Step 10 definition of done: eyeball check that P(large) makes sense."""

from app.router import create_router
from app.router.features import extract_router_features
from app.risk.risk import extract as extract_risk

router = create_router()

# Easy queries — should get LOW P(large)
easy = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "What color is the sky?",
    "How do I print in Python?",
    "What is HTML?",
]

# Hard queries — should get HIGH P(large)
hard = [
    "Explain the mathematical proof of Godel's incompleteness theorem.",
    "Write a distributed consensus algorithm in Python with fault tolerance.",
    "Compare and contrast transformer and LSTM architectures for NLP.",
    "Design a database schema for a multi-tenant SaaS application.",
    "Analyze the computational complexity of the knapsack problem.",
]

print("Easy queries (expect LOW P(large)):")
for q in easy:
    risk = extract_risk(q)
    features = extract_router_features(q, risk)
    p = router.predict(features)
    print(f"  P(large)={p:.3f}  {q[:60]}")

print("\nHard queries (expect HIGH P(large)):")
for q in hard:
    risk = extract_risk(q)
    features = extract_router_features(q, risk)
    p = router.predict(features)
    print(f"  P(large)={p:.3f}  {q[:60]}")