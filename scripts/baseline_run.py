# scripts/baseline_run.py
"""Run 30 queries through the pass-through proxy and save results."""

import httpx
import json
import time
from pathlib import Path

QUERIES = [
    "What is the capital of France?",
    "How do I reverse a list in Python?",
    "Explain quantum computing in simple terms.",
    "What is 15 * 37?",
    "Write a haiku about programming.",
    "What is the difference between let and const in JavaScript?",
    "Who painted the Mona Lisa?",
    "How does photosynthesis work?",
    "What is the refund window for order 4471?",
    "Convert 100 Fahrenheit to Celsius.",
    "What is machine learning?",
    "Write a SQL query to find duplicate rows.",
    "What is the population of Tokyo?",
    "Explain the difference between TCP and UDP.",
    "How do I handle errors in Python?",
    "What is the meaning of life?",
    "Write a function to check if a string is a palindrome.",
    "What is the capital of Japan?",
    "How does a neural network learn?",
    "What is 2 to the power of 10?",
    "Explain REST APIs to a beginner.",
    "What is the Pythagorean theorem?",
    "How do I create a virtual environment in Python?",
    "What is the GDP of the United States?",
    "Write a regex to match email addresses.",
    "What is the boiling point of water?",
    "Explain the concept of recursion.",
    "What are the SOLID principles?",
    "How does HTTPS work?",
    "What is the time complexity of binary search?",
]

client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)
results = []

for i, query in enumerate(QUERIES, 1):
    print(f"[{i:2d}/30] {query[:50]}...")
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.0,
    })
    resp.raise_for_status()
    data = resp.json()
    results.append({
        "query": query,
        "answer": data["choices"][0]["message"]["content"],
        "x_routecache": data["x_routecache"],
    })
    meta = data["x_routecache"]
    print(f"       cost=${meta['cost_usd']:.8f}  latency={meta['latency_ms']['total']:.0f}ms")

    # Rate limit mitigation: 2-second pause between requests
    # Groq free tier: ~30 req/min token limit. This ensures we stay well under.
    # Disk cache (Step 2) means cached responses return instantly without this delay.
    if i < len(QUERIES):  # Don't sleep after the last query
        time.sleep(3)

# Save results
output_path = Path("data/baseline_30_queries.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

total_cost = sum(r["x_routecache"]["cost_usd"] for r in results)
avg_latency = sum(r["x_routecache"]["latency_ms"]["total"] for r in results) / len(results)
print(f"\n{'=' * 50}")
print(f"Total cost:    ${total_cost:.6f}")
print(f"Avg latency:   {avg_latency:.0f} ms")
print(f"Results saved: {output_path}")