"""
Router protocol — the contract both router implementations must satisfy.

Traceability: FR-3.5
Build Plan: Step 10
"""

from typing import Protocol
from app.router.features import RouterFeatures


class Router(Protocol):
    """
    Protocol for model routing.

    Both LogisticRouter and LightGBMRouter implement this.
    Swap implementations by changing config.router.model.
    """
    def predict(self, features: RouterFeatures) -> float:
        """
        Predict the probability that the large model is needed.

        Args:
            features: Extracted query features.

        Returns:
            P(large_needed) in [0, 1]. If >= config.router.threshold,
            route to the large model; otherwise route to small.
        """
        ...