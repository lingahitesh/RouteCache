# scripts/test_cache.py
"""
Step 7 definition of done:
  - Replay a small workload with deliberate repeats (5-10 paraphrases)
  - Confirm hit rate is nonzero
  - Confirm cached answers are actually returned instead of calling the model
"""

import httpx

client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)

# Queries with deliberate paraphrases
workload = [
    # Original
    "How do I reverse a list in Python?",
    "What is the capital of France?",
    "Explain machine learning simply.",

    # Paraphrases — should hit cache
    "How can I reverse a Python list?",
    "What's the capital of France?",
    "Explain ML in simple terms.",

    # More paraphrases
    "Reverse a list in Python how?",
    "France's capital is what?",
    "What is machine learning, explained simply?",

    # New query — should miss
    "What is the Pythagorean theorem?",
]

hits = 0
misses = 0

for query in workload:
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": query}],
    })
    
    # Debug: check response status and content
    if resp.status_code != 200:
        print(f"ERROR: Status {resp.status_code}")
        print(f"Response text: {resp.text}")
        resp.raise_for_status()
    
    try:
        data = resp.json()
    except Exception as e:
        print(f"ERROR parsing JSON for query: {query}")
        print(f"Response status: {resp.status_code}")
        print(f"Response headers: {resp.headers}")
        print(f"Response text: {resp.text[:500]}")
        raise
    
    meta = data["x_routecache"]
    cache_status = meta["cache"]
    sim = meta.get("similarity")

    if cache_status in ("hit", "verified_hit"):
        hits += 1
        sim_str = f"sim={sim:.4f}" if sim is not None else "sim=N/A"
        print(f"  HIT  ({sim_str}) {query[:50]}")
    else:
        misses += 1
        print(f"  MISS              {query[:50]}")

print(f"\nHits: {hits}, Misses: {misses}, Hit rate: {hits/(hits+misses)*100:.0f}%")
assert hits > 0, "FAIL: cache never hit — paraphrases should produce hits!"
print("Step 7 definition of done: PASSED")