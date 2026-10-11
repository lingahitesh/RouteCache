"""
Decision policy protocol and shared data structures.

The decision engine is the core of C3 (contribution 3): instead of
treating cache and router as two independent stages in a pipeline,
it compares the expected cost of each possible action and picks the
cheapest one that stays under the wrong-hit risk budget.

Traceability: FR-3.9, FR-3.10
Build Plan: Step 11
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol

from app.cache.models import CacheCandidate
from app.risk.risk import RiskFeatures
from app.router.features import RouterFeatures
from app.verifier.base import VerifierResult


class Action(Enum):
    """The four possible actions the system can take."""
    SERVE = "serve"       # serve cached answer directly
    VERIFY = "verify"     # verify, then serve or fall through
    SMALL = "small"       # call the small model
    LARGE = "large"       # call the large model


@dataclass
class DecisionTrace:
    """
    The complete record of a decision — not just the action, but WHY.

    This is what gets logged (NFR-9). The full expected_costs table
    lets error analysis see why an action was chosen, not only which one.

    Detailed Design Doc Section 2.6: "DecisionTrace — not just the bare
    Action — is what the API layer logs."
    """
    action: Action
    zone: Literal["SERVE", "VERIFY", "MISS"] | None
    similarity: float | None
    risk_score: float | None
    risk_flags: list[str] = field(default_factory=list)
    thresholds: dict[str, float] | None = None
    router_score: float | None = None
    verifier_result: VerifierResult | None = None
    expected_costs: dict[str, float] = field(default_factory=dict)
    provenance_override: bool = False    # True if provenance override fired


class DecisionPolicy(Protocol):
    """
    Protocol that both policies implement.

    ExpectedCostPolicy (C3, FR-3.9): cost-aware joint decision.
    FixedRulePolicy (FR-3.10 baseline): always verify in the band.
    """
    def decide(
        self,
        candidate: CacheCandidate | None,
        risk_features: RiskFeatures,
        router_features: RouterFeatures,
        zone: str,
        risk_score: float,
        effective_t_low: float,
        effective_t_high: float,
    ) -> DecisionTrace:
        ...