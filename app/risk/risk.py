"""
Risk feature extraction and adaptive threshold resolution.

Extracts six risk features from each query (FR-2.10):
  1. numbers   — digits, quantities, specific values
  2. negation  — "not", "n't", "never", "without", etc.
  3. entities  — named entities (people, places, orgs, products)
  4. code      — code fences, syntax tokens, indentation
  5. math      — mathematical expressions, "solve", "calculate"
  6. temporal  — dates, times, "today", "currently", "last year"

These are the query properties where a near-miss is most likely to
produce a WRONG cached answer. "How do I reverse a list?" and
"How can I reverse a Python list?" are safe to conflate. "Order 123"
and "order 124" are not. The risk features detect that difference.

Implementation: regex first (fast, predictable, no model dependency),
spaCy second (for entities and negation dependency parsing). The Build
Plan says "regex first, spaCy once the regex version works."

Traceability: FR-2.10 (C2), Detailed Design Doc Section 4.1
Build Plan: Step 9
"""

import re
import time
from dataclasses import dataclass, field

from config.config import load_config


# ─── Regex Patterns ───────────────────────────────────────────────────

# Numbers: digits, decimals, ordinals, quantities
NUMBER_PATTERN = re.compile(
    r'\b\d[\d,.]*\b'           # "123", "3.14", "1,000"
    r'|\b\d+(?:st|nd|rd|th)\b' # "1st", "2nd", "3rd"
    r'|\b(?:first|second|third|fourth|fifth)\b',  # ordinal words
    re.IGNORECASE
)

# Negation: words and contractions that flip meaning
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|neither|nor|none|nobody|nothing|nowhere|"
    r"hardly|barely|scarcely|seldom|without|cannot|can't|won't|"
    r"wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|"
    r"don't|doesn't|didn't|hasn't|haven't|hadn't)\b",
    re.IGNORECASE
)

# Code: fences, syntax tokens, common programming constructs
CODE_FENCE_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`')
CODE_TOKEN_PATTERN = re.compile(
    r'(?:def |class |import |from |return |if |elif |else:|'
    r'for |while |try:|except |raise |yield |async |await )'
    r'|[a-zA-Z_]\w*\(.*?\)'     # function calls: foo(), bar(x)
    r'|==|!=|>=|<=|=>|->|\+=|-='  # operators
    r'|\b(?:null|None|True|False|undefined|NaN)\b'
    r'|(?:^|\n)\s{4,}\S',         # indented lines (4+ spaces)
    re.MULTILINE
)

# Math: expressions, operator words, math-specific terms
MATH_PATTERN = re.compile(
    r'\d+\s*[+\-*/^%]\s*\d+'    # "2 + 3", "15 * 37"
    r'|\b(?:solve|calculate|compute|evaluate|simplify|derive|integrate|'
    r'differentiate|sum|product|factorial|logarithm|sqrt|root|'
    r'equation|formula|theorem|proof)\b'
    r'|[=<>]{1,2}'               # =, ==, <, >, <=, >=
    r'|\b(?:x|y|z)\s*=',         # variable assignment
    re.IGNORECASE
)

# Temporal: dates, times, relative time references
TEMPORAL_PATTERN = re.compile(
    r'\b(?:today|tomorrow|yesterday|currently|now|recent|latest|'
    r'this\s+(?:year|month|week|morning)|'
    r'last\s+(?:year|month|week|night)|'
    r'next\s+(?:year|month|week)|'
    r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|'
    r'(?:19|20)\d{2}|'           # years: 1900-2099
    r'\d{1,2}/\d{1,2}/\d{2,4}|'  # dates: 9/28/2026
    r'\d{1,2}:\d{2})\b',         # times: 14:30
    re.IGNORECASE
)

# Entity patterns (regex fallback — used when spaCy is not available)
ENTITY_PATTERN = re.compile(
    r'\b(?:order|product|item|ticket|case|invoice|account|'
    r'id|ID|#)\s*(?:#?\d+|[A-Z]{2,}-\d+)'   # "order 4471", "JIRA-123"
    r'|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'   # "Elon Musk", "New York"
)


# ─── Data Structures ─────────────────────────────────────────────────

@dataclass
class RiskFeatures:
    """
    Risk features extracted from a single query.

    Each boolean flag indicates whether a specific risk category was
    detected. raw_signals stores the actual matched spans for
    debugging and error analysis (SRS Section 8.5).
    """
    has_number: bool = False
    has_negation: bool = False
    has_entity: bool = False
    has_code: bool = False
    has_math: bool = False
    has_temporal: bool = False
    raw_signals: dict[str, list[str]] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def active_flags(self) -> list[str]:
        """List of risk feature names that are active."""
        flags = []
        if self.has_number: flags.append("numbers")
        if self.has_negation: flags.append("negation")
        if self.has_entity: flags.append("entities")
        if self.has_code: flags.append("code")
        if self.has_math: flags.append("math")
        if self.has_temporal: flags.append("temporal")
        return flags

    @property
    def count(self) -> int:
        """Number of active risk features."""
        return len(self.active_flags)

    def to_dict(self) -> dict:
        """For JSON serialization in logs."""
        return {
            "has_number": self.has_number,
            "has_negation": self.has_negation,
            "has_entity": self.has_entity,
            "has_code": self.has_code,
            "has_math": self.has_math,
            "has_temporal": self.has_temporal,
            "active_flags": self.active_flags,
        }


# ─── Extraction (Regex-Based) ────────────────────────────────────────

def _find_all(pattern: re.Pattern, text: str) -> list[str]:
    """Return all matched spans as strings."""
    return [m.group() for m in pattern.finditer(text)]


def extract(query: str, active_features: list[str] | None = None) -> RiskFeatures:
    """
    Extract risk features from a query string.

    This is the regex-based implementation. It runs in <1ms on any query.
    A spaCy-enhanced version (for entity NER and dependency-parse negation)
    can be added later — the regex version is the required baseline.

    Args:
        query: The user's query text.
        active_features: Which features to extract. Defaults to all six.
                         Pass config.risk.features to respect the config.

    Returns:
        RiskFeatures with boolean flags and raw_signals for each category.
    """
    if active_features is None:
        config = load_config()
        active_features = config.risk.features

    start = time.time()
    raw = {}
    features = RiskFeatures()

    if "numbers" in active_features:
        matches = _find_all(NUMBER_PATTERN, query)
        features.has_number = len(matches) > 0
        raw["numbers"] = matches

    if "negation" in active_features:
        matches = _find_all(NEGATION_PATTERN, query)
        features.has_negation = len(matches) > 0
        raw["negation"] = matches

    if "entities" in active_features:
        matches = _find_all(ENTITY_PATTERN, query)
        features.has_entity = len(matches) > 0
        raw["entities"] = matches

    if "code" in active_features:
        fence_matches = _find_all(CODE_FENCE_PATTERN, query)
        token_matches = _find_all(CODE_TOKEN_PATTERN, query)
        all_matches = fence_matches + token_matches
        features.has_code = len(all_matches) > 0
        raw["code"] = all_matches

    if "math" in active_features:
        matches = _find_all(MATH_PATTERN, query)
        features.has_math = len(matches) > 0
        raw["math"] = matches

    if "temporal" in active_features:
        matches = _find_all(TEMPORAL_PATTERN, query)
        features.has_temporal = len(matches) > 0
        raw["temporal"] = matches

    features.raw_signals = raw
    features.latency_ms = (time.time() - start) * 1000

    return features


# ─── Adaptive Threshold Resolution ───────────────────────────────────

def compute_risk_score(features: RiskFeatures) -> float:
    """
    Aggregate risk features into a single 0–1 score.

    Simple approach: each active feature contributes equally.
    6 features max, so score = count / 6.

    A more sophisticated model (trained on wrong-hit data) replaces
    this in later phases — this is the starting point.
    """
    return features.count / 6.0


def resolve_thresholds(
    features: RiskFeatures,
    risk_score: float,
    base_t_low: float,
    base_t_high: float,
    policy: str,
    adaptive_slope: float = 0.05,
) -> tuple[float, float]:
    """
    Resolve the effective T_low and T_high for this query.

    If policy == "fixed": return base thresholds unchanged.
    If policy == "adaptive": tighten thresholds proportional to risk.

    Tightening means RAISING thresholds — a riskier query needs
    HIGHER similarity to be served from cache, and an even HIGHER
    similarity to skip verification entirely.

    Detailed Design Doc Section 4.2 worked example:
      base_t_low=0.82, base_t_high=0.93, adaptive_slope=0.05
      risk_score=0.1 (low risk)  -> t_low=0.825, t_high=0.935
      risk_score=0.9 (high risk) -> t_low=0.865, t_high=0.975

    Args:
        features:       The extracted RiskFeatures.
        risk_score:     Aggregated 0–1 risk score.
        base_t_low:     Config's cache.t_low.
        base_t_high:    Config's cache.t_high.
        policy:         "fixed" or "adaptive".
        adaptive_slope: How much risk_score shifts thresholds.

    Returns:
        (effective_t_low, effective_t_high)
    """
    if policy == "fixed":
        return base_t_low, base_t_high

    # Adaptive: shift thresholds up proportionally to risk
    offset = risk_score * adaptive_slope
    t_low = min(base_t_low + offset, 1.0)
    t_high = min(base_t_high + offset, 1.0)

    # Safety: t_high must stay above t_low
    if t_high <= t_low:
        t_high = min(t_low + 0.01, 1.0)

    return t_low, t_high