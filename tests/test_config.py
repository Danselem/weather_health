"""Config composition tests for env/model overrides."""

from src.config import compose_config


def test_default_config():
    cfg = compose_config("config")
    assert cfg.env.name == "dev"
    assert cfg.modeling.model_family == "logistic_regression"


def test_model_and_env_overrides():
    cfg = compose_config("config", ["env=prod", "model=random_forest"])
    assert cfg.env.name == "prod"
    assert cfg.modeling.model_family == "random_forest"
    assert cfg.modeling.n_trials == 10


def test_explicit_model_family_override():
    cfg = compose_config("config", ["modeling.model_family=gradient_boosting"])
    assert cfg.modeling.model_family == "gradient_boosting"
