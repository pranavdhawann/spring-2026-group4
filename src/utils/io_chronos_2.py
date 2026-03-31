"""I/O utilities for persisting evaluation results.

Creates timestamped experiment directories for saving outputs.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def make_experiment_dir(base_dir: str) -> Path:
    """Create a timestamped subdirectory under *base_dir*.

    Args:
        base_dir: Root experiments directory.

    Returns:
        Path to the newly created directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(base_dir) / timestamp
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", path)
    return path
