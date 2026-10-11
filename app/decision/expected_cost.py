"""
ExpectedCostPolicy — the joint cache-router decision engine (C3).

This is the core algorithmic contribution (FR-3.9): instead of treating
the cache and router as independent stages, it compares the expected cost
of all four possible actions (SERVE, VERIFY, SMALL, LARGE) and picks
the cheapest one, with a penalty for wrong-hit risk.

Expected cost of action A:
    E[cost(A)] = dollar_cost(A) + penalty_lambda * P(wrong_hit | A)

The penalty_lambda parameter controls the cost-vs-safety trade-off.
High lambda = conservative (prefers verification / large model).
Low lambda = aggressive (serves from cache more often).

This is Detailed Design Document Section 4.6 — the FULL algorithm
with every branch explicit, including the provenance override.

Traceability: FR-3.9 (C3)
Build Plan: Step 11
"""

from config.config import load_config
from app.decision.base import Action, DecisionTrace
from app.cache.models import CacheCandidate
from app.risk.risk import RiskFeatures
from app.router.features import RouterFeatures
from app.verifier import create_verifier
from app.router import create_router
from app.cost import compute_cost


class ExpectedCostPolicy:
    """
    Cost-aware joint decision policy (C3).

    Compares expected cost across all four actions:
      SERVE:  ~$0 cost, but risk of serving a wrong cached answer
      VERIFY: ~$0 + verifier latency cost, lower wrong-hit risk
      SMALL:  cheap model call, quality depends on query difficulty
      LARGE:  expensive model call, highest quality
    """

    def __init__(self):
        config = load_config()
        self.config = config
        self.penalty_lambda = config.decision.penalty_lambda
        self.hard_query_threshold = config.decision.hard_query_threshold
        self.router_threshold = config.router.threshold
        self.verifier = create_verifier()
        self.router = create_router()

        # Cost constants (approximate, from config)
        # These are per-request estimates used in the expected-cost formula
        self.C_serve = 0.0        # serving from cache costs nothing
        self.C_verify = 0.0001    # verifier compute cost (CPU time, negligible $)
        self.C_small = (
            config.models["small"].price_in * 500 +
            config.models["small"].price_out * 200
        )  # estimate for ~500 in, ~200 out tokens
        self.C_large = (
            config.models["large"].price_in * 500 +
            config.models["large"].price_out * 200
        )

    def _estimate_wrong_prob_serve(self, risk_score: float,
                                    provenance_tier: str) -> float:
        """
        Estimate P(wrong hit | SERVE) — serving cached answer without verification.

        Higher risk = higher chance of wrong hit.
        Small-model provenance = higher chance than large-model provenance.
        """
        base = 0.05 * risk_score  # risk contribution
        if provenance_tier == "small":
            base += 0.03  # small-model answers are less reliable
        return min(base, 1.0)

    def _estimate_wrong_prob_verify(self, risk_score: float,
                                     provenance_tier: str) -> float:
        """
        Estimate P(wrong hit | VERIFY) — serving after verification.

        Much lower than SERVE, because the verifier catches most wrong hits.
        But not zero — verifiers have false-positive rates.
        """
        base = 0.01 * risk_score
        if provenance_tier == "small":
            base += 0.005
        return min(base, 1.0)

    def _estimate_wrong_prob_route(self, action: Action, p_large: float) -> float:
        """
        Estimate P(wrong answer | routing to this model).

        If routing to SMALL when P(large needed) is high, the wrong
        probability is higher.
        """
        if action == Action.SMALL:
            return p_large * 0.1   # higher p_large = more likely small fails
        else:  # LARGE
            return 0.005           # large model rarely wrong

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
        """
        The full C3 algorithm from Detailed Design Doc Section 4.6.

        Every branch is explicit, including the provenance override.
        """

        # ── MISS: no cache candidate, go straight to routing ──
        if zone == "MISS" or candidate is None:
            p_large = self.router.predict(router_features)
            action = Action.LARGE if p_large >= self.router_threshold else Action.SMALL

            return DecisionTrace(
                action=action,
                zone="MISS",
                similarity=None,
                risk_score=risk_score,
                risk_flags=risk_features.active_flags,
                router_score=p_large,
            )

        # ── SERVE or VERIFY: build the expected-cost table ──
        provenance_tier = candidate.provenance.source_model
        p_large = self.router.predict(router_features)

        # Estimate wrong-hit probabilities for each action
        p_wrong = {
            Action.SERVE: self._estimate_wrong_prob_serve(risk_score, provenance_tier),
            Action.VERIFY: self._estimate_wrong_prob_verify(risk_score, provenance_tier),
            Action.SMALL: self._estimate_wrong_prob_route(Action.SMALL, p_large),
            Action.LARGE: self._estimate_wrong_prob_route(Action.LARGE, p_large),
        }

        # Dollar costs for each action
        costs = {
            Action.SERVE: self.C_serve,
            Action.VERIFY: self.C_serve + self.C_verify,
            Action.SMALL: self.C_small,
            Action.LARGE: self.C_large,
        }

        # Expected cost = dollar cost + lambda * P(wrong)
        expected = {
            a: costs[a] + self.penalty_lambda * p_wrong[a]
            for a in Action
        }

        # Pick the action with lowest expected cost
        best = min(expected, key=expected.get)

        # ── Provenance override (Lesson 8, Section 19) ──
        # Never SERVE a small-tier entry the router flags as hard.
        # This runs BEFORE the verifier fallback below.
        provenance_override = False
        if (best == Action.SERVE
                and provenance_tier == "small"
                and p_large > self.hard_query_threshold):
            best = Action.VERIFY
            provenance_override = True

        # ── If VERIFY was chosen, actually run the verifier ──
        verifier_result = None
        if best == Action.VERIFY:
            verifier_result = self.verifier.check(
                new_query=router_features.raw_query,
                cached_query=candidate.query,
            )
            if verifier_result.verdict == "SAFE":
                best = Action.SERVE  # verification passed
            else:
                # Verification failed — route to model
                best = Action.LARGE if p_large >= self.router_threshold else Action.SMALL

        return DecisionTrace(
            action=best,
            zone=zone,
            similarity=candidate.similarity,
            risk_score=risk_score,
            risk_flags=risk_features.active_flags,
            thresholds={"t_low": effective_t_low, "t_high": effective_t_high},
            router_score=p_large,
            verifier_result=verifier_result,
            expected_costs={a.value: round(v, 8) for a, v in expected.items()},
            provenance_override=provenance_override,
        )