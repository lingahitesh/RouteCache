"""
Generate routing labels by running both small and large models on a
query set and comparing quality with the judge.

This is the most expensive script in the project (API calls to 3 models
per query). Run it ONCE, cache everything (NFR-8), replay for free.

Label logic (Detailed Design Doc Section 4.5, Lesson 7, Section 7):
  - Run small model on query -> answer_small
  - Run large model on query -> answer_large
  - Judge compares both (pairwise, both orderings)
  - If small is "good enough" (tie or win): label = 0 (route to small)
  - If large wins clearly: label = 1 (route to large)

Build Plan: Step 10
"""

import json
import time
from pathlib import Path
import numpy as np
from config.config import load_config
from app.adapters.adapters import GroqAdapter, GeminiAdapter, RetryingAdapter
from app.router.features import extract_router_features
from app.risk.risk import extract as extract_risk
from app.router.logistic import LogisticRouter
from app.utils.paths import get_data_dir
from app.adapters.adapters import ResponseDiskCache

config = load_config()
disk_cache = ResponseDiskCache()

small_adapter = RetryingAdapter(GroqAdapter(config.models["small"], disk_cache))
large_adapter = RetryingAdapter(GroqAdapter(config.models["large"], disk_cache))
judge1_adapter = RetryingAdapter(GeminiAdapter(config.models["judge1"], disk_cache))
judge2_adapter = RetryingAdapter(GeminiAdapter(config.models["judge2"], disk_cache))

# Load training queries (you need to prepare this file)
queries_path = get_data_dir() / "router_training_queries.json"
if not queries_path.exists():
    print("ERROR: Create data/router_training_queries.json first.")
    print("It should be a JSON list of query strings.")
    print('Example: ["How do I reverse a list?", "Explain quantum computing.", ...]')
    exit(1)

with open(queries_path) as f:
    queries = json.load(f)

print(f"Generating labels for {len(queries)} queries...")
print(f"This calls 3 models per query. Total API calls: ~{len(queries) * 3}")
print(f"Estimated time with rate limiting: ~{len(queries) * 2 / 60:.1f} minutes")

# Judge prompt (SRS Appendix C)
JUDGE_PROMPT = """You are comparing two answers to the same question.

QUESTION: {question}
ANSWER A: {answer_a}
ANSWER B: {answer_b}

Which answer is more correct and helpful? Respond with exactly one of:
A, B, TIE.
Do not favour longer answers."""

features_list = []
labels = []
results = []

for i, query in enumerate(queries):
    print(f"[{i + 1}/{len(queries)}] {query[:50]}...")

    # Get both answers
    messages = [{"role": "user", "content": query}]
    r_small = small_adapter.call(messages)
    r_large = large_adapter.call(messages)
    # Judge: order A=small, B=large (use judge1)
    judge_msg_1 = [{"role": "user", "content": JUDGE_PROMPT.format(
        question=query, answer_a=r_small.text, answer_b=r_large.text
    )}]
    j1 = judge1_adapter.call(judge_msg_1, temperature=0.0)
    time.sleep(2)
    
    # Judge: order A=large, B=small (use judge2 for second opinion)
    judge_msg_2 = [{"role": "user", "content": JUDGE_PROMPT.format(
        question=query, answer_a=r_large.text, answer_b=r_small.text
    )}]
    j2 = judge2_adapter.call(judge_msg_2, temperature=0.0)
    time.sleep(2)
    # Parse verdicts
    v1 = j1.text.strip().upper()
    v2 = j2.text.strip().upper()

    # Determine label
    # v1: A=small, B=large. v2: A=large, B=small
    small_wins = (v1 == "A" and v2 == "B")  # small won in both orderings
    tie = (v1 == "TIE" and v2 == "TIE")
    large_wins = (v1 == "B" and v2 == "A")  # large won in both orderings

    if small_wins or tie:
        label = 0  # small is good enough
    elif large_wins:
        label = 1  # large needed
    else:
        label = 1  # disagreement — default to large (conservative)

    # Extract features
    risk = extract_risk(query)
    rf = extract_router_features(query, risk)
    features_list.append(rf.to_vector())
    labels.append(label)

    results.append({
        "query": query, "label": label,
        "judge_v1": v1, "judge_v2": v2,
        "small_answer": r_small.text[:100], "large_answer": r_large.text[:100],
    })
    print(f"       label={label} ({'large needed' if label else 'small ok'})")

    # Rate limit mitigation: 2-second pause between query batches
    # Each query makes 3 API calls (small, large, judge×2).
    # 2-second pause keeps us under Groq's ~30 req/min limit.
    # Disk cache means already-seen queries skip the LLM calls entirely.
    if i < len(queries) - 1:  # Don't sleep after last query
        time.sleep(2)

# Save raw results
output_path = get_data_dir() / "router_labels.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nLabels saved to {output_path}")

# Train the router
X = np.array(features_list)
y = np.array(labels)

router = LogisticRouter()
metrics = router.train(X, y)
router.save()

print(f"\nRouter trained:")
print(f"  Accuracy:   {metrics['accuracy']:.2%}")
print(f"  Samples:    {metrics['n_samples']}")
print(f"  Features:   {metrics['n_features']}")
print(f"  Model saved to: {config.router.model_path}")