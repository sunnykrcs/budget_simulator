"""
Coordinate Management for Bayesian Hierarchical Models
======================================================

This module centralizes all coordinate and dimension definitions used by the
Bayesian forecasting model.

PyMC uses coordinates ("coords") and dimensions ("dims") to label random
variables. Using named dimensions improves:

- readability
- debugging
- posterior analysis with ArviZ
- plotting
- xarray interoperability

This module validates encoded hierarchy indices and constructs reusable
coordinate dictionaries consumed by every model component.

Author
------
Bayesian Revenue Forecasting Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Required encoded hierarchy columns
# ---------------------------------------------------------------------

CHANNEL_INDEX = "channel_idx"
CAMPAIGN_TYPE_INDEX = "campaign_type_idx"
CAMPAIGN_INDEX = "campaign_idx"

REQUIRED_INDEX_COLUMNS = (
    CHANNEL_INDEX,
    CAMPAIGN_TYPE_INDEX,
    CAMPAIGN_INDEX,
)

# ---------------------------------------------------------------------
# Coordinate names used throughout PyMC
# ---------------------------------------------------------------------

OBSERVATION_DIM = "observation"
CHANNEL_DIM = "channel"
CAMPAIGN_TYPE_DIM = "campaign_type"
CAMPAIGN_DIM = "campaign"


@dataclass(slots=True)
class ModelDims:
    """
    Stores the dimensionality of every hierarchical level.

    Parameters
    ----------
    n_observations:
        Number of rows.

    n_channels:
        Number of unique marketing channels.

    n_campaign_types:
        Number of campaign types.

    n_campaigns:
        Number of unique campaigns.
    """

    n_observations: int
    n_channels: int
    n_campaign_types: int
    n_campaigns: int

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ModelDims":
        """
        Infer model dimensions from an encoded dataframe.

        Parameters
        ----------
        df:
            Feature-engineered dataframe.

        Returns
        -------
        ModelDims
        """

        validate_index_columns(df)

        return cls(
            n_observations=len(df),
            n_channels=df[CHANNEL_INDEX].nunique(),
            n_campaign_types=df[CAMPAIGN_TYPE_INDEX].nunique(),
            n_campaigns=df[CAMPAIGN_INDEX].nunique(),
        )

    @property
    def summary(self) -> Dict[str, int]:
        """
        Dictionary representation of model dimensions.
        """

        return {
            OBSERVATION_DIM: self.n_observations,
            CHANNEL_DIM: self.n_channels,
            CAMPAIGN_TYPE_DIM: self.n_campaign_types,
            CAMPAIGN_DIM: self.n_campaigns,
        }


# ---------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------


def validate_index_columns(df: pd.DataFrame) -> None:
    """
    Validate that all encoded hierarchy columns exist.

    Parameters
    ----------
    df:
        Encoded dataframe.

    Raises
    ------
    KeyError
        Missing required columns.
    """

    missing = [
        column
        for column in REQUIRED_INDEX_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing encoded hierarchy columns: "
            f"{', '.join(missing)}"
        )


def validate_contiguous_indices(
    series: pd.Series,
    name: str,
) -> None:
    """
    Ensure encoded indices are contiguous.

    Expected

    0,1,2,...,N-1

    Parameters
    ----------
    series:
        Encoded integer index.

    name:
        Column name.

    Raises
    ------
    ValueError
        If gaps exist.
    """

    values = np.sort(series.unique())

    expected = np.arange(len(values))

    if not np.array_equal(values, expected):
        raise ValueError(
            f"{name} must contain contiguous indices "
            f"0..{len(values)-1}"
        )


def validate_hierarchy(df: pd.DataFrame) -> None:
    """
    Validate all hierarchy encodings.

    Parameters
    ----------
    df:
        Encoded feature dataframe.
    """

    validate_index_columns(df)

    validate_contiguous_indices(
        df[CHANNEL_INDEX],
        CHANNEL_INDEX,
    )

    validate_contiguous_indices(
        df[CAMPAIGN_TYPE_INDEX],
        CAMPAIGN_TYPE_INDEX,
    )

    validate_contiguous_indices(
        df[CAMPAIGN_INDEX],
        CAMPAIGN_INDEX,
    )

# ---------------------------------------------------------------------
# Coordinate builders
# ---------------------------------------------------------------------


def build_coords(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build PyMC coordinate dictionary.

    Parameters
    ----------
    df
        Feature-engineered dataframe containing encoded hierarchy indices.

    Returns
    -------
    Dict[str, np.ndarray]
        Coordinate dictionary suitable for ``pm.Model(coords=...)``.

    Examples
    --------
    >>> coords = build_coords(df)
    >>> with pm.Model(coords=coords):
    ...     ...
    """

    validate_hierarchy(df)

    dims = ModelDims.from_dataframe(df)

    coords: Dict[str, np.ndarray] = {
        OBSERVATION_DIM: np.arange(dims.n_observations),
        CHANNEL_DIM: np.arange(dims.n_channels),
        CAMPAIGN_TYPE_DIM: np.arange(dims.n_campaign_types),
        CAMPAIGN_DIM: np.arange(dims.n_campaigns),
    }

    return coords


# ---------------------------------------------------------------------
# Dimension mapping
# ---------------------------------------------------------------------


def build_dimension_dict() -> Dict[str, str]:
    """
    Build a mapping between encoded dataframe columns and
    PyMC coordinate names.

    Returns
    -------
    Dict[str, str]
        Dictionary used when specifying model dimensions.

    Examples
    --------
    >>> build_dimension_dict()
    {
        "channel_idx": "channel",
        ...
    }
    """

    return {
        CHANNEL_INDEX: CHANNEL_DIM,
        CAMPAIGN_TYPE_INDEX: CAMPAIGN_TYPE_DIM,
        CAMPAIGN_INDEX: CAMPAIGN_DIM,
    }


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------


def get_dimension_sizes(df: pd.DataFrame) -> Dict[str, int]:
    """
    Return the size of every model dimension.

    Parameters
    ----------
    df
        Encoded dataframe.

    Returns
    -------
    Dict[str, int]
        Mapping of coordinate names to their sizes.
    """

    dims = ModelDims.from_dataframe(df)
    return dims.summary


def get_index_arrays(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Extract encoded hierarchy indices as NumPy arrays.

    Parameters
    ----------
    df
        Encoded dataframe.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing integer arrays for indexing
        hierarchical effects inside the PyMC model.
    """

    validate_hierarchy(df)

    return {
        CHANNEL_INDEX: df[CHANNEL_INDEX].to_numpy(dtype=np.int64),
        CAMPAIGN_TYPE_INDEX: df[
            CAMPAIGN_TYPE_INDEX
        ].to_numpy(dtype=np.int64),
        CAMPAIGN_INDEX: df[
            CAMPAIGN_INDEX
        ].to_numpy(dtype=np.int64),
    }


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "ModelDims",
    "build_coords",
    "build_dimension_dict",
    "get_dimension_sizes",
    "get_index_arrays",
    "validate_hierarchy",
    "validate_index_columns",
    "validate_contiguous_indices",
    "CHANNEL_INDEX",
    "CAMPAIGN_TYPE_INDEX",
    "CAMPAIGN_INDEX",
    "OBSERVATION_DIM",
    "CHANNEL_DIM",
    "CAMPAIGN_TYPE_DIM",
    "CAMPAIGN_DIM",
]