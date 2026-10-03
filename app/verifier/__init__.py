"""
Verifier factory — returns the configured verifier implementation.

Usage:
    from app.verifier import create_verifier
    verifier = create_verifier()   # reads config.verifier.type
    result = verifier.check(new_query, cached_query)
"""

from config.config import load_config
from app.verifier.base import Verifier, VerifierResult
from app.verifier.cross_encoder import CrossEncoderVerifier


def create_verifier() -> Verifier:
    """
    Create a verifier instance based on config.verifier.type.

    Returns:
        A Verifier implementation (CrossEncoderVerifier or NLIVerifier).

    Raises:
        NotImplementedError: if NLI is selected but not yet built.
        ValueError: if the verifier type is unknown.
    """
    config = load_config()

    if config.verifier.type == "cross_encoder":
        return CrossEncoderVerifier(
            model_name=config.verifier.model_name,
            pass_threshold=config.verifier.pass_threshold,
        )
    elif config.verifier.type == "nli":
        from app.verifier.nli import NLIVerifier
        return NLIVerifier()
    else:
        raise ValueError(f"Unknown verifier type: {config.verifier.type}")