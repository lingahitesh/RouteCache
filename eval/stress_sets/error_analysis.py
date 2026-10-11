"""
Category-level error analysis on the adversarial set.

Runs each adversarial pair through the system and reports:
  - Per-category: how many pairs did the system get right/wrong?
  - Which risk features fired (or didn't fire) on wrong predictions?
  - Verifier effectiveness: did the verifier catch the cases it should have?

This is the core of C4's evaluation: not "does the system work on average"
but "where exactly does it fail."

SRS Section 8.5: "False cache hits by cause (near-paraphrase with
different answer, negation, changed numbers, swapped entities,
code-mixed near-miss)"

Build Plan: Step 14
"""

import json
from collections import defaultdict
from pathlib import Path

import httpx


def run_adversarial_evaluation(
    pairs_path: str = "eval/stress_sets/adversarial_pairs.json",
    proxy_url: str = "http://localhost:8000",
    output_path: str = "eval/stress_sets/adversarial_results.json",
) -> dict:
    """
    Run all adversarial pairs through the system and collect results.

    For each non-equivalent pair:
      1. Send query_a -> cache the answer
      2. Send query_b -> does the system serve query_a's answer? (wrong hit!)
      3. Record whether the system correctly rejected the cached answer

    Returns per-category error breakdown.
    """
    with open(pairs_path) as f:
        pairs = json.load(f)

    client = httpx.Client(base_url=proxy_url, timeout=60.0)
    results = []
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})

    for pair in pairs:
        qa, qb = pair["query_a"], pair["query_b"]
        label = pair["label"]
        category = pair["category"]

        # Step 1: Cache query_a's answer
        resp_a = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": qa}],
        })
        data_a = resp_a.json()

        # Step 2: Send query_b — does it serve from cache?
        resp_b = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": qb}],
        })
        data_b = resp_b.json()

        meta_b = data_b["x_routecache"]
        served_from_cache = meta_b["cache"] in ("hit", "verified_hit")

        # Determine correctness
        if label == "non_equivalent":
            # System should NOT serve from cache (different meaning)
            correct = not served_from_cache
        else:
            # System SHOULD serve from cache (same meaning)
            correct = served_from_cache

        category_stats[category]["total"] += 1
        if correct:
            category_stats[category]["correct"] += 1
        else:
            category_stats[category]["wrong"] += 1

        results.append({
            "query_a": qa,
            "query_b": qb,
            "label": label,
            "category": category,
            "served_from_cache": served_from_cache,
            "correct": correct,
            "similarity": meta_b.get("similarity"),
            "zone": meta_b.get("zone"),
            "risk_flags": meta_b.get("risk_flags", []),
            "verifier_verdict": meta_b.get("verifier", {}).get("verdict"),
        })

    # Save detailed results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print category breakdown
    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION — Category-Level Error Breakdown")
    print("=" * 70)

    for category, stats in sorted(category_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        wrong = stats["wrong"]
        accuracy = correct / total if total > 0 else 0
        print(f"\n  {category}:")
        print(f"    Total: {total}  Correct: {correct}  Wrong: {wrong}  Accuracy: {accuracy:.0%}")

    # Overall
    total_all = sum(s["total"] for s in category_stats.values())
    correct_all = sum(s["correct"] for s in category_stats.values())
    print(f"\n  OVERALL: {correct_all}/{total_all} correct ({correct_all/total_all:.0%})")

    return dict(category_stats)



def analyze_by_category(results_path: str = "eval/stress_sets/adversarial_results.json") -> dict:
    """
    Analyze saved evaluation results by category.
    
    Useful for offline analysis after running the evaluation.
    
    Args:
        results_path: Path to the saved results JSON file.
        
    Returns:
        Dictionary with per-category statistics.
    """
    with open(results_path) as f:
        results = json.load(f)
    
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})
    
    for result in results:
        category = result["category"]
        correct = result["correct"]
        
        category_stats[category]["total"] += 1
        if correct:
            category_stats[category]["correct"] += 1
        else:
            category_stats[category]["wrong"] += 1
    
    # Print breakdown
    print("\n" + "=" * 70)
    print("CATEGORY-LEVEL ERROR ANALYSIS")
    print("=" * 70)
    
    for category, stats in sorted(category_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        wrong = stats["wrong"]
        accuracy = correct / total if total > 0 else 0
        print(f"\n  {category}:")
        print(f"    Total: {total}  Correct: {correct}  Wrong: {wrong}  Accuracy: {accuracy:.0%}")
    
    # Overall
    total_all = sum(s["total"] for s in category_stats.values())
    correct_all = sum(s["correct"] for s in category_stats.values())
    accuracy_all = correct_all / total_all if total_all > 0 else 0
    print(f"\n  OVERALL: {correct_all}/{total_all} correct ({accuracy_all:.0%})")
    print("=" * 70)
    
    return dict(category_stats)
