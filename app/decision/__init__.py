"""
Decision policy factory.

Usage:
    from app.decision import create_decision_policy
    policy = create_decision_policy()   # reads config
    trace = policy.decide(candidate, risk, router_features, zone, ...)
"""

from config.config import load_config
from app.decision.base import DecisionPolicy, DecisionTrace, Action


def create_decision_policy() -> DecisionPolicy:
    """
    Create the configured decision policy.

    The choice is made at startup, not per-request — the same policy
    handles all requests in a given run. To compare policies, change
    the config and re-run the evaluation harness.
    """
    # For now, default to ExpectedCostPolicy (C3).
    # FixedRulePolicy is used by explicitly importing it for baseline runs.
    from app.decision.expected_cost import ExpectedCostPolicy
    return ExpectedCostPolicy()