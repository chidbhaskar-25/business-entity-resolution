"""Configuration loader and schema management."""

import os
from typing import Any, Dict, Optional
import yaml


class Config(dict):
    """Dictionary subclass providing dot-notation attribute access."""

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def load_config(config_path: str = "config.yaml") -> Config:
    """Loads a YAML configuration file into a Config object.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Config object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(data)
