"""
Cost computation for RouteCache.

Implements SRS Appendix E.1:
    cost(request) = in_tokens * price_in(model) + out_tokens * price_out(model)
    savings %     = 1 - (total cost of system) / (total cost of always-large)

Traceability: FR-6.1, FR-6.5, SRS Appendix E.1
Build Plan: Step 3
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import Config


def compute_cost(tokens_in: int, tokens_out: int,
                 price_in: float, price_out: float) -> float:
    """
    Cost of a single request given raw token counts and per-token prices.

    Args:
        tokens_in:  Number of input (prompt) tokens.
        tokens_out: Number of output (completion) tokens.
        price_in:   USD per input token for this model.
        price_out:  USD per output token for this model.

    Returns:
        Cost in USD (float). Will be very small — e.g. $0.00000405
        for 19 in + 2 out on the large model.
    """
    return tokens_in * price_in + tokens_out * price_out


def compute_cost_for_model(tokens_in: int, tokens_out: int,
                           model_key: str, config: "Config") -> float:
    """
    Cost of a single request, looking up prices from config by model key.

    Args:
        tokens_in:  Number of input tokens.
        tokens_out: Number of output tokens.
        model_key:  "small", "large", or "judge".
        config:     The loaded Config object.

    Returns:
        Cost in USD.

    Raises:
        KeyError: if model_key is not in config.models.
    """
    model = config.models[model_key]
    return compute_cost(tokens_in, tokens_out, model.price_in, model.price_out)


def compute_baseline_cost(tokens_in: int, tokens_out: int,
                          config: "Config") -> float:
    """
    What this request WOULD have cost on the always-large baseline.
    Used for FR-6.5's "cost saved" computation.

    Note: this assumes the same token counts, which is an approximation —
    different models produce different-length outputs. For a precise
    comparison, you need the large model's actual output, which is only
    available if you ran both (as in router label generation, Step 10).
    """
    return compute_cost_for_model(tokens_in, tokens_out, "large", config)


def compute_savings_percent(system_cost: float, baseline_cost: float) -> float:
    """
    Percentage of cost saved vs the always-large baseline.

    SRS Appendix E.1: savings % = 1 - (system cost) / (baseline cost)

    IMPORTANT (FR-6.5): Never report this number alone. Always report
    it alongside the wrong-hit rate from NFR-3.

    Returns:
        Savings as a fraction (0.0 to 1.0). Multiply by 100 for percent.
        Returns 0.0 if baseline_cost is 0 (avoid division by zero).
    """
    if baseline_cost == 0:
        return 0.0
    return 1.0 - (system_cost / baseline_cost)