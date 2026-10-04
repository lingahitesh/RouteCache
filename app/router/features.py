"""
Router feature extraction.

Extracts features from a query that predict whether the small model
can answer it adequately, or whether the large model is needed.

The feature set is deliberately simple to start — logistic regression
on these features is the baseline. LightGBM with richer features is
the comparison (FR-3.5).

Features:
  - length: character count (longer queries tend to be harder)
  - token_count: approximate token count (words / 0.75)
  - has_code: presence of code tokens
  - has_math: presence of math expressions
  - reasoning_signal: heuristic for multi-step reasoning
  - entity_count: number of named entities
  - risk_feature_count: from Step 9's extraction

Traceability: FR-3.1
Build Plan: Step 10
"""

import re
from dataclasses import dataclass
import numpy as np

from app.risk.risk import extract as extract_risk, RiskFeatures


@dataclass
class RouterFeatures:
    """
    Features used by the router to predict P(large_needed).

    raw_query is kept so the decision engine can pass it to the verifier
    if needed (Step 11).
    """
    raw_query: str
    length: int
    token_count: int
    has_code: bool
    has_math: bool
    reasoning_signal: float
    entity_count: int
    risk_feature_count: int

    def to_vector(self) -> np.ndarray:
        """Convert to a numeric feature vector for the classifier."""
        return np.array([
            self.length,
            self.token_count,
            int(self.has_code),
            int(self.has_math),
            self.reasoning_signal,
            self.entity_count,
            self.risk_feature_count,
        ], dtype=np.float32)


# Reasoning keywords that suggest multi-step thinking
REASONING_KEYWORDS = re.compile(
    r'\b(?:explain|why|how|compare|contrast|analyze|evaluate|'
    r'what\s+(?:are|is)\s+the\s+(?:difference|relationship|impact)|'
    r'step\s+by\s+step|pros?\s+and\s+cons?|trade-?off|'
    r'design|implement|architect|debug|optimize)\b',
    re.IGNORECASE
)


def extract_router_features(query: str, risk: RiskFeatures | None = None) -> RouterFeatures:
    """
    Extract router features from a query.

    Args:
        query: The user's query text.
        risk:  Pre-computed risk features (to avoid duplicate extraction).
               If None, extracts them here.

    Returns:
        RouterFeatures ready for the classifier.
    """
    if risk is None:
        risk = extract_risk(query)

    # Approximate token count (words / 0.75 is a rough estimate)
    words = len(query.split())
    token_count = int(words / 0.75)

    # Reasoning signal: count of reasoning-related keywords, normalized
    reasoning_matches = len(REASONING_KEYWORDS.findall(query))
    reasoning_signal = min(reasoning_matches / 3.0, 1.0)  # cap at 1.0

    # Entity count from risk signals
    entity_count = len(risk.raw_signals.get("entities", []))

    return RouterFeatures(
        raw_query=query,
        length=len(query),
        token_count=token_count,
        has_code=risk.has_code,
        has_math=risk.has_math,
        reasoning_signal=reasoning_signal,
        entity_count=entity_count,
        risk_feature_count=risk.count,
    )