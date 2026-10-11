"""
FixedRulePolicy — the required baseline for FR-3.10.

Simple rules:
  - SERVE zone: serve directly (no cost comparison)
  - VERIFY zone: always verify; serve if SAFE, else route to model
  - MISS zone: route based on router score

This is what makes the FR-3.9 vs. FR-3.10 comparison (RQ4) a policy
swap, not a rewrite: both policies implement DecisionPolicy, and the
route handler doesn't know which one is running.

Build Plan: Step 11
"""

from config.config import load_config
from app.decision.base import Action, DecisionTrace
from app.cache.models import CacheCandidate
from app.risk.risk import RiskFeatures
from app.router.features import RouterFeatures
from app.verifier import create_verifier


class FixedRulePolicy:
    """
    Fixed-rule decision policy (FR-3.10 baseline).

    No expected-cost comparison. No provenance override.
    SERVE zone = serve. VERIFY zone = verify. MISS = route.
    """

    def __init__(self):
        config = load_config()
        self.router_threshold = config.router.threshold
        self.verifier = create_verifier()

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

        # ── MISS: route to model ──
        if zone == "MISS" or candidate is None:
            from app.router import create_router
            router = create_router()
            p_large = router.predict(router_features)
            action = Action.LARGE if p_large >= self.router_threshold else Action.SMALL

            return DecisionTrace(
                action=action,
                zone="MISS",
                similarity=None,
                risk_score=risk_score,
                risk_flags=risk_features.active_flags,
                router_score=p_large,
            )

        # ── SERVE: serve directly ──
        if zone == "SERVE":
            return DecisionTrace(
                action=Action.SERVE,
                zone="SERVE",
                similarity=candidate.similarity,
                risk_score=risk_score,
                risk_flags=risk_features.active_flags,
                thresholds={"t_low": effective_t_low, "t_high": effective_t_high},
            )

        # ── VERIFY: always verify in the band ──
        verifier_result = self.verifier.check(
            new_query=router_features.raw_query,
            cached_query=candidate.query,
        )

        if verifier_result.verdict == "SAFE":
            action = Action.SERVE
        else:
            # Verification failed — route to model
            from app.router import create_router
            router = create_router()
            p_large = router.predict(router_features)
            action = Action.LARGE if p_large >= self.router_threshold else Action.SMALL

        return DecisionTrace(
            action=action,
            zone="VERIFY",
            similarity=candidate.similarity,
            risk_score=risk_score,
            risk_flags=risk_features.active_flags,
            thresholds={"t_low": effective_t_low, "t_high": effective_t_high},
            verifier_result=verifier_result,
            router_score=router.predict(router_features) if verifier_result.verdict != "SAFE" else None,
        )