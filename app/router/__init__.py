"""
Router factory — returns the configured router implementation.

Usage:
    from app.router import create_router
    router = create_router()
    p_large = router.predict(features)
"""

from config.config import load_config
from app.router.base import Router
from app.router.logistic import LogisticRouter


def create_router() -> Router:
    config = load_config()
    if config.router.model == "logistic_regression":
        r = LogisticRouter()
        try:
            r.load()
        except FileNotFoundError:
            pass  # Not trained yet — will fail on predict() with a clear error
        return r
    elif config.router.model == "lightgbm":
        raise NotImplementedError(
            "LightGBM router is deferred. Train and validate the logistic "
            "regression baseline first (FR-3.5)."
        )
    else:
        raise ValueError(f"Unknown router model: {config.router.model}")