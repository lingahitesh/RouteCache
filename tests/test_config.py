"""
Step 1 definition of done: load your actual config.yaml and assert
every field is the expected type. Nothing else needs to run yet.
"""

from config.config import load_config, Config, ModelConfig, CacheConfig

def test_load_config_returns_config():
    cfg = load_config()
    assert isinstance(cfg, Config)

def test_models_section():
    cfg = load_config()
    assert "small" in cfg.models
    assert "large" in cfg.models
    assert "judge" in cfg.models
    for name, model in cfg.models.items():
        assert isinstance(model, ModelConfig)
        assert isinstance(model.name, str) and len(model.name) > 0
        assert model.price_in >= 0
        assert model.price_out >= 0
        assert model.base_url.startswith("http")

def test_cache_thresholds_ordered():
    cfg = load_config()
    assert 0 < cfg.cache.t_low < cfg.cache.t_high <= 1.0

def test_risk_features_valid():
    cfg = load_config()
    valid = {"numbers", "negation", "entities", "code", "math", "temporal"}
    for f in cfg.risk.features:
        assert f in valid, f"Unknown risk feature: {f}"

def test_verifier_threshold_range():
    cfg = load_config()
    assert 0 < cfg.verifier.pass_threshold < 1.0

def test_decision_penalty_positive():
    cfg = load_config()
    assert cfg.decision.penalty_lambda > 0

def test_embedding_dimension():
    cfg = load_config()
    assert cfg.embedding.dimension == 384  # MiniLM-L6-v2