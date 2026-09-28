"""
RouteCache configuration loader.

Reads config.yaml, validates all fields via Pydantic, and returns a
typed Config object that every other module imports.

Traceability: FR-9.1 (all parameters from YAML config)
Build Plan: Step 1
"""

import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from typing import Literal

load_dotenv()


# ─── Model Definitions ───────────────────────────────────────────────

class ModelConfig(BaseModel):
    """One LLM provider entry (small, large, or judge)."""
    name: str                          # model name as the provider expects it
    price_in: float                    # USD per input token
    price_out: float                   # USD per output token
    base_url: str                      # OpenAI-compatible API base URL
    api_key_env: str | None = None     # env var name holding the API key


# ─── Cache ────────────────────────────────────────────────────────────

class CacheConfig(BaseModel):
    """Three-zone cache settings (SRS FR-2.2, FR-2.5)."""
    t_low: float                       # lower similarity threshold
    t_high: float                      # upper similarity threshold
    ttl_seconds: int                   # default TTL for cache entries
    ttl_seconds_temporal: int          # shorter TTL for time-sensitive entries
    eviction: Literal["lru", "lfu"]    # eviction policy
    max_entries: int                   # max cache size (NFR-1)
    top_k: int = 5                     # FAISS search returns this many candidates

    @field_validator("t_high")
    @classmethod
    def t_high_above_t_low(cls, v, info):
        if "t_low" in info.data and v <= info.data["t_low"]:
            raise ValueError(f"t_high ({v}) must be > t_low ({info.data['t_low']})")
        return v


# ─── Risk ─────────────────────────────────────────────────────────────

class RiskConfig(BaseModel):
    """Risk-adaptive threshold settings (SRS FR-2.10, C2)."""
    policy: Literal["fixed", "adaptive"]
    adaptive_slope: float = 0.05       # how much risk_score shifts thresholds
    features: list[str]                # active risk detectors


# ─── Verifier ─────────────────────────────────────────────────────────

class VerifierConfig(BaseModel):
    """Verification settings for borderline cache hits (SRS FR-2.6, C1)."""
    type: Literal["cross_encoder", "nli"]
    model_name: str                    # HuggingFace model identifier
    pass_threshold: float              # score above = SAFE
    timeout_ms: int = 5000             # max ms before timeout error


# ─── Router ───────────────────────────────────────────────────────────

class RouterConfig(BaseModel):
    """Model routing settings (SRS FR-3.1–FR-3.5)."""
    model: Literal["logistic_regression", "lightgbm"]
    threshold: float                   # P(large) decision boundary
    model_path: str = "data/router_model.pkl"


# ─── Decision Engine ─────────────────────────────────────────────────

class DecisionConfig(BaseModel):
    """Joint cache-router decision settings (SRS FR-3.9, C3)."""
    penalty_lambda: float              # weight on wrong-hit risk
    wrong_hit_ceiling: float           # NFR-3 target
    hard_query_threshold: float = 0.70 # provenance override trigger


# ─── Logging ──────────────────────────────────────────────────────────

class LoggingConfig(BaseModel):
    """Logging and database settings (SRS FR-6.4, NFR-9)."""
    level: str = "INFO"
    db_path: str = "data/routecache.db"
    json_log_path: str = "logs/requests.jsonl"


# ─── Embedding ────────────────────────────────────────────────────────

class EmbeddingConfig(BaseModel):
    """Local embedding model settings (SRS FR-2.1)."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    normalize: bool = True


# ─── Top-Level Config ────────────────────────────────────────────────

class Config(BaseModel):
    """
    Top-level configuration object. Loaded once at process start.
    Every module receives this (or a slice of it) via dependency injection.
    """
    models: dict[str, ModelConfig]
    cache: CacheConfig
    risk: RiskConfig
    verifier: VerifierConfig
    router: RouterConfig
    decision: DecisionConfig
    logging: LoggingConfig = LoggingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()


def load_config(path: str = "config.yaml") -> Config:
    """
    Load and validate the YAML config file.

    Args:
        path: Filename relative to the config/ directory.
              Default: "config.yaml"

    Returns:
        A fully validated Config object.

    Raises:
        FileNotFoundError: if the YAML file doesn't exist.
        pydantic.ValidationError: if any field is missing or has the wrong type.
            This is intentional — fail loudly on bad config, never silently
            use defaults for required fields.
    """
    full_path = os.path.join(os.path.dirname(__file__), path)
    with open(full_path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)