"""Shared constants for the Chronos 2 evaluation pipeline."""

from typing import Dict, List

# Default quantile levels for probabilistic evaluation
DEFAULT_QUANTILE_LEVELS: List[float] = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

# Mapping from pandas frequency strings to seasonal periods
FREQ_TO_SEASONALITY_MAP: Dict[str, int] = {
    # Sub-hourly
    "T": 60,
    "min": 60,
    "S": 60,
    # Hourly
    "H": 24,
    "h": 24,
    "BH": 24,
    "bh": 24,
    # Daily
    "D": 7,
    "B": 5,
    # Weekly
    "W": 52,
    "W-SUN": 52,
    "W-MON": 52,
    # Monthly
    "M": 12,
    "MS": 12,
    "ME": 12,
    "BM": 12,
    "BMS": 12,
    "BME": 12,
    # Quarterly
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "BQ": 4,
    "BQS": 4,
    # Yearly
    "Y": 1,
    "YS": 1,
    "YE": 1,
    "A": 1,
    "AS": 1,
    "BA": 1,
    "BAS": 1,
}

# Maximum context length supported by Chronos 2
MAX_CONTEXT_LENGTH: int = 8192
