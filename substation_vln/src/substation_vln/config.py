"""Helpers for tool YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from substation_vln.paths import PROJECT_ROOT


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return an empty dict for an empty file."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def config_path(config: dict[str, Any], key: str, default: str | Path | None = None) -> Path | None:
    value = config_value(config, key, default)
    return resolve_project_path(value)
