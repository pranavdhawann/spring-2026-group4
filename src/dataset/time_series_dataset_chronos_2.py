"""Domain object representing a single univariate time series."""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesDataset:
    """Container for a single univariate time series.

    Attributes:
        series_id: Unique identifier for this series (typically the column name).
        values: 1-D array of numeric observations, ordered chronologically.
        timestamps: 1-D array of ``datetime64`` values aligned with *values*.
        frequency: Pandas frequency alias (e.g. ``"H"``, ``"D"``).
        seasonality: Integer seasonal period used for MASE scaling.
    """

    series_id: str
    values: np.ndarray
    timestamps: np.ndarray
    frequency: str
    seasonality: int


def from_dataframe(
    df: pd.DataFrame,
    timestamp_column: str,
    target_columns: List[str],
    frequency: str,
    seasonality: int,
) -> List[TimeSeriesDataset]:
    """Build :class:`TimeSeriesDataset` instances from a DataFrame.

    Each target column is treated as an independent univariate series.

    Args:
        df: Preprocessed DataFrame with a timestamp column and one or more
            numeric target columns.
        timestamp_column: Name of the timestamp column.
        target_columns: List of column names to convert into series objects.
        frequency: Pandas frequency alias for the data.
        seasonality: Seasonal period integer.

    Returns:
        A list of :class:`TimeSeriesDataset` objects, one per target column.
    """
    timestamps = df[timestamp_column].values
    datasets: List[TimeSeriesDataset] = []

    for col in target_columns:
        datasets.append(
            TimeSeriesDataset(
                series_id=col,
                values=df[col].values.astype(np.float64),
                timestamps=timestamps,
                frequency=frequency,
                seasonality=seasonality,
            )
        )

    return datasets
