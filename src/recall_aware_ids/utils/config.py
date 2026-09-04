"""
utils/config.py — YAML configuration loader.
Loads and merges experiment, model, defenses, attacks, and controller configs.
Compatible with Python 3.9+.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import yaml


_CONFIG_DIR = Path(__file__).parents[4] / "configs"


def _resolve_config_dir(config_dir=None) -> Path:
    """Resolve config directory, searching up from package root if needed."""
    if config_dir:
        return Path(config_dir)
    # Try the computed path first
    candidate = _CONFIG_DIR
    if candidate.exists():
        return candidate
    # Fall back: walk up from cwd looking for configs/
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        trial = parent / "configs"
        if (trial / "experiment.yaml").exists():
            return trial
    raise FileNotFoundError(
        f"Cannot find configs/ directory. Tried: {_CONFIG_DIR}, cwd={cwd}"
    )



def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load all YAML config files and return as a merged dict keyed by filename stem."""
    d = _resolve_config_dir(config_dir)
    configs: dict[str, Any] = {}
    for yaml_file in sorted(d.glob("*.yaml")):
        configs[yaml_file.stem] = load_yaml(yaml_file)
    return configs


def get_experiment_config(config_dir: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    d = _resolve_config_dir(config_dir)
    return load_yaml(d / "experiment.yaml")


def get_controller_config(
    config_id: str,
    config_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    d = _resolve_config_dir(config_dir)
    controllers = load_yaml(d / "controllers.yaml")
    configs = controllers["controller_configurations"]
    if config_id not in configs:
        raise ValueError(f"Unknown controller config: {config_id}. Valid: {list(configs)}")
    return configs[config_id]
