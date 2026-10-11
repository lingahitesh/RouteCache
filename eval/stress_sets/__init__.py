"""
Stress test sets for adversarial and code-mixed evaluation.

This package contains generators for:
  - adversarial.py: Near-duplicate pairs across 6 adversarial categories
  - code_mixed.py: Hinglish (Hindi+English) test queries
  - error_analysis.py: Category-level error breakdown

These are Contribution C4 from the Build Plan.

Traceability: FR-9.1, FR-9.2, FR-9.3
Build Plan: Step 14
"""

from eval.stress_sets.adversarial import AdversarialPair
from eval.stress_sets.code_mixed import build_code_mixed_set
from eval.stress_sets.error_analysis import analyze_by_category

__all__ = [
    "AdversarialPair",
    "build_code_mixed_set",
    "analyze_by_category",
]
