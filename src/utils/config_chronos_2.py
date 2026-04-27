"""YAML configuration loading and validation.

Loads the three config files
(``model_chronos_2.yaml``, ``dataset_chronos_2.yaml``,
``evaluation_chronos_2.yaml``) and merges them into a single dictionary.
Supports CLI argument overrides.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a single YAML file and return its contents.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fp:
        data = yaml.safe_load(fp)
    return data if data is not None else {}


def load_configs(
    config_dir: str = "config",
    model_config_path: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    eval_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge all configuration files.

    Reads ``model_chronos_2.yaml``, ``dataset_chronos_2.yaml``, and
    ``evaluation_chronos_2.yaml`` from *config_dir* (or from explicit
    override paths) and returns a single merged dictionary with top-level
    keys ``"model"``, ``"dataset"``, and ``"evaluation"``.

    Args:
        config_dir: Directory containing the three YAML files.
        model_config_path: Override path for the model config.
        dataset_config_path: Override path for the dataset config.
        eval_config_path: Override path for the evaluation config.

    Returns:
        Merged configuration dictionary.
    """
    base = Path(config_dir)

    model_path = (
        Path(model_config_path) if model_config_path else base / "model_chronos_2.yaml"
    )
    dataset_path = (
        Path(dataset_config_path)
        if dataset_config_path
        else base / "dataset_chronos_2.yaml"
    )
    eval_path = (
        Path(eval_config_path)
        if eval_config_path
        else base / "evaluation_chronos_2.yaml"
    )

    logger.info("Loading configs from: %s, %s, %s", model_path, dataset_path, eval_path)

    merged: Dict[str, Any] = {}
    merged.update(_load_yaml(model_path))
    merged.update(_load_yaml(dataset_path))
    merged.update(_load_yaml(eval_path))

    logger.info("Configuration loaded successfully.")
    return merged
