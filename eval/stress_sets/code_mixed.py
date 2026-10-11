"""
Code-mixed query slice for C4 evaluation.

SRS FR-8.9(b): "a code-mixed slice (Hinglish or Bengali written in
Roman script) of at least 200 queries with paraphrase pairs."

Survey existing resources first (Phase 0). If nothing suitable exists,
construct from scratch and document the procedure.

IMPORTANT: The SRS says "check for existing benchmarks before building
your own." This file provides the construction framework; the actual
queries come from one of:
  1. An existing Hinglish paraphrase dataset (if found in Phase 0 survey)
  2. Hand-built + template-generated pairs (fallback)

Traceability: FR-8.9 (C4)
Build Plan: Step 14
"""

import json
from pathlib import Path


# ─── Hinglish paraphrase templates ───────────────────────────────────
# These are examples. A real set needs 200+ queries with verified labels.

HINGLISH_SEED_PAIRS = [
    # (query_hinglish, query_english, label)
    ("Python mein list kaise reverse karein?",
     "How do I reverse a list in Python?",
     "equivalent"),

    ("Machine learning kya hota hai?",
     "What is machine learning?",
     "equivalent"),

    ("France ki capital kya hai?",
     "What is the capital of France?",
     "equivalent"),

    ("Mera order 4471 ka return policy kya hai?",
     "What is the return policy for order 4471?",
     "equivalent"),

    ("Mera order 4471 ka return policy kya hai?",
     "What is the return policy for order 4472?",
     "non_equivalent"),   # number swap in code-mixed context

    ("Kya Python beginners ke liye accha hai?",
     "Is Python good for beginners?",
     "equivalent"),

    ("Kya Python beginners ke liye accha NAHI hai?",
     "Is Python NOT good for beginners?",
     "equivalent"),   # both are negation queries — same meaning

    ("TensorFlow kaise install karein?",
     "How do I install TensorFlow?",
     "equivalent"),

    ("PyTorch kaise install karein?",
     "How do I install TensorFlow?",
     "non_equivalent"),   # entity swap: PyTorch vs TensorFlow
]


def build_code_mixed_set(
    output_path: str = "eval/stress_sets/code_mixed_pairs.json",
) -> None:
    """
    Build or load the code-mixed evaluation set.

    If an existing dataset was found in the Phase 0 survey, load and
    format it. Otherwise, use the seed pairs above and document that
    the set is hand-built (not a benchmark).
    """
    pairs = [
        {
            "query_a": a,
            "query_b": b,
            "label": label,
            "source": "hand_built_hinglish",
        }
        for a, b, label in HINGLISH_SEED_PAIRS
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pairs)} code-mixed pairs to {output_path}")
    print()
    print("NOTE: This is a seed set, not a full benchmark.")
    print("Expand to >= 200 pairs for the final evaluation.")
    print("Document the construction procedure per FR-8.9.")