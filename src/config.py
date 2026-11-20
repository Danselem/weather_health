"""Small helpers for composing Hydra configs into one runtime config."""

from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

MERGED_SECTIONS = ("modeling", "data", "mlflow", "prefect", "logging", "cloud")


def compose_config(config_dir: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Compose Hydra config, then flatten the env/model groups."""
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = Path.cwd() / config_dir

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name="config", overrides=overrides or [])
        OmegaConf.resolve(cfg)

    return flatten_groups(cfg, overrides or [])


def flatten_groups(cfg: DictConfig, overrides: list[str] | None = None) -> DictConfig:
    """Turn cfg.env.* and cfg.model.* into top-level runtime settings.

    Precedence: base config < model group default < env group < explicit CLI.
    """
    overrides = overrides or []
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    root: dict[str, Any] = {str(key): value for key, value in container.items()}

    env_block = root.pop("env", None)
    model_block = root.pop("model", None)

    if model_block:
        root["modeling"] = OmegaConf.merge(root.get("modeling", {}), model_block)

    if env_block:
        for section in MERGED_SECTIONS:
            if section in env_block:
                root[section] = OmegaConf.merge(root.get(section, {}), env_block[section])
        root["env"] = env_block.get("env", {})

    if any(o.startswith("model=") for o in overrides) and model_block:
        root["modeling"].model_family = model_block["model_family"]

    for override in overrides:
        if override.startswith("modeling.model_family="):
            root["modeling"].model_family = override.split("=", 1)[1]

    return cast(DictConfig, OmegaConf.create(root))


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a Hydra config directory or a plain YAML file."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_file():
        cfg = OmegaConf.load(path)
        OmegaConf.resolve(cfg)
        return cast(DictConfig, cfg)

    return compose_config(path, overrides)
