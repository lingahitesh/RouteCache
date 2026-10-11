"""
Adversarial near-duplicate construction.

Builds pairs where surface similarity is HIGH but the correct answer
DIFFERS. These are the exact failure mode a bi-encoder cache is worst
at — and the exact scenario contributions C1 (verification) and C2
(risk-adaptive thresholds) exist to handle.

Six adversarial categories (Lesson 11, Section 6):
  1. Number swap:   "order 123" vs "order 124"
  2. Negation:      "Can I return?" vs "Can I NOT return?"
  3. Entity swap:   "GDP of Japan" vs "GDP of China"
  4. Year/date:     "population in 2020" vs "population in 2021"
  5. Order reversal: "Is A better than B?" vs "Is B better than A?"
  6. Code variation: "sort ascending" vs "sort descending"

Construction procedure:
  1. Start from PAWS test set (if available) — these are machine-generated
     paraphrases from Wikipedia with both equivalent and non-equivalent labels
  2. Add hand-built pairs for each of the six categories
  3. Label each pair as "equivalent" or "non_equivalent"
  4. For non-equivalent pairs, note WHICH category it tests

Traceability: FR-8.9 (C4), Lesson 11 Section 6
Build Plan: Step 14
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AdversarialPair:
    """A single adversarial test pair."""
    query_a: str
    query_b: str
    label: str             # "equivalent" or "non_equivalent"
    category: str          # "number_swap", "negation", "entity_swap", etc.
    expected_answer_differs: bool


# ─── Hand-Built Adversarial Pairs ─────────────────────────────────────

ADVERSARIAL_PAIRS = [
    # ── Number swaps ──
    AdversarialPair(
        "What is the return policy for order 4471?",
        "What is the return policy for order 4472?",
        "non_equivalent", "number_swap", True,
    ),
    AdversarialPair(
        "How much RAM does the iPhone 15 have?",
        "How much RAM does the iPhone 14 have?",
        "non_equivalent", "number_swap", True,
    ),
    AdversarialPair(
        "What is the price of product #1234?",
        "What is the price of product #1235?",
        "non_equivalent", "number_swap", True,
    ),
    AdversarialPair(
        "Compare Python 3.11 and 3.10 performance.",
        "Compare Python 3.12 and 3.11 performance.",
        "non_equivalent", "number_swap", True,
    ),
    AdversarialPair(
        "What is 15 * 37?",
        "What is 15 * 38?",
        "non_equivalent", "number_swap", True,
    ),

    # ── Negation ──
    AdversarialPair(
        "Can I return this item?",
        "Can I NOT return this item?",
        "non_equivalent", "negation", True,
    ),
    AdversarialPair(
        "Is Python good for beginners?",
        "Is Python NOT good for beginners?",
        "non_equivalent", "negation", True,
    ),
    AdversarialPair(
        "Does this function handle null values?",
        "Does this function NOT handle null values?",
        "non_equivalent", "negation", True,
    ),
    AdversarialPair(
        "Should I use a database for this?",
        "Should I avoid using a database for this?",
        "non_equivalent", "negation", True,
    ),
    AdversarialPair(
        "Is it safe to delete this file?",
        "Is it NOT safe to delete this file?",
        "non_equivalent", "negation", True,
    ),

    # ── Entity swaps ──
    AdversarialPair(
        "What is the GDP of Japan?",
        "What is the GDP of China?",
        "non_equivalent", "entity_swap", True,
    ),
    AdversarialPair(
        "Who is the CEO of Google?",
        "Who is the CEO of Microsoft?",
        "non_equivalent", "entity_swap", True,
    ),
    AdversarialPair(
        "What language is Django written in?",
        "What language is Rails written in?",
        "non_equivalent", "entity_swap", True,
    ),
    AdversarialPair(
        "How do I install TensorFlow?",
        "How do I install PyTorch?",
        "non_equivalent", "entity_swap", True,
    ),
    AdversarialPair(
        "What is the population of Tokyo?",
        "What is the population of Osaka?",
        "non_equivalent", "entity_swap", True,
    ),

    # ── Year/date (temporal) ──
    AdversarialPair(
        "What was the population of India in 2020?",
        "What was the population of India in 2021?",
        "non_equivalent", "temporal", True,
    ),
    AdversarialPair(
        "What is the latest version of Python as of 2024?",
        "What is the latest version of Python as of 2025?",
        "non_equivalent", "temporal", True,
    ),
    AdversarialPair(
        "Who won the World Cup in 2018?",
        "Who won the World Cup in 2022?",
        "non_equivalent", "temporal", True,
    ),

    # ── Order reversal ──
    AdversarialPair(
        "Is Python faster than JavaScript?",
        "Is JavaScript faster than Python?",
        "non_equivalent", "order_reversal", True,
    ),
    AdversarialPair(
        "Is React better than Angular?",
        "Is Angular better than React?",
        "non_equivalent", "order_reversal", True,
    ),

    # ── Code variation ──
    AdversarialPair(
        "Sort a list in ascending order in Python.",
        "Sort a list in descending order in Python.",
        "non_equivalent", "code_variation", True,
    ),
    AdversarialPair(
        "Write a function that returns the maximum value.",
        "Write a function that returns the minimum value.",
        "non_equivalent", "code_variation", True,
    ),

    # ── True paraphrases (equivalent — should NOT be flagged) ──
    AdversarialPair(
        "How do I reverse a list in Python?",
        "How can I reverse a Python list?",
        "equivalent", "true_paraphrase", False,
    ),
    AdversarialPair(
        "What is machine learning?",
        "Explain what ML is.",
        "equivalent", "true_paraphrase", False,
    ),
    AdversarialPair(
        "What is the capital of France?",
        "What city is the capital of France?",
        "equivalent", "true_paraphrase", False,
    ),
]


def save_adversarial_set(path: str = "eval/stress_sets/adversarial_pairs.json") -> None:
    """Save the adversarial set to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pairs = [
        {
            "query_a": p.query_a,
            "query_b": p.query_b,
            "label": p.label,
            "category": p.category,
            "expected_answer_differs": p.expected_answer_differs,
        }
        for p in ADVERSARIAL_PAIRS
    ]
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} adversarial pairs to {path}")


def load_adversarial_set(path: str = "eval/stress_sets/adversarial_pairs.json") -> list[dict]:
    """Load the adversarial set from disk."""
    with open(path) as f:
        return json.load(f)