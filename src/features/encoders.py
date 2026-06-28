"""
Categorical Encoders
====================

This module provides reusable categorical encoders for the
Bayesian Revenue Forecasting pipeline.

Unlike one-hot encoding, these encoders generate integer indices,
which are required for hierarchical Bayesian models.

Example
-------
>>> encoder = CategoryEncoder()
>>> df = encoder.fit_transform(df, "channel")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import json
import pandas as pd


@dataclass(slots=True)
class CategoryEncoder:
    """
    Integer encoder for categorical variables.

    Attributes
    ----------
    mapping : dict
        Maps category names to integer indices.

    inverse_mapping : dict
        Maps integer indices back to category names.
    """

    mapping: dict = field(default_factory=dict)

    inverse_mapping: dict = field(default_factory=dict)

    def fit(
        self,
        series: pd.Series
    ) -> None:
        """
        Learn category-to-index mapping.

        Parameters
        ----------
        series : pd.Series
            Input categorical data.
        """

        categories = sorted(
            series.astype(str).unique()
        )

        self.mapping = {
            category: idx
            for idx, category in enumerate(categories)
        }

        self.inverse_mapping = {
            idx: category
            for category, idx in self.mapping.items()
        }

    def transform(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Transform categories into integer indices.

        Parameters
        ----------
        series : pd.Series

        Returns
        -------
        pd.Series
        """

        if not self.mapping:
            raise RuntimeError(
                "Encoder has not been fitted."
            )

        encoded = (
            series.astype(str)
            .map(self.mapping)
        )

        if encoded.isna().any():
            unknown = (
                series[encoded.isna()]
                .unique()
            )

            raise ValueError(
                f"Unknown categories: {unknown}"
            )

        return encoded.astype(int)

    def fit_transform(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Fit the encoder and transform the input series.

        Parameters
        ----------
        series : pd.Series

        Returns
        -------
        pd.Series
            Encoded integer indices.
        """

        self.fit(series)

        return self.transform(series)

    def inverse_transform(
        self,
        encoded: pd.Series
    ) -> pd.Series:
        """
        Convert integer indices back to category names.

        Parameters
        ----------
        encoded : pd.Series

        Returns
        -------
        pd.Series
        """

        if not self.inverse_mapping:
            raise RuntimeError(
                "Encoder has not been fitted."
            )

        decoded = encoded.map(self.inverse_mapping)

        if decoded.isna().any():
            unknown = encoded[decoded.isna()].unique()

            raise ValueError(
                f"Unknown encoded values: {unknown}"
            )

        return decoded

    def save_mapping(
        self,
        filepath: str
    ) -> None:
        """
        Save category mapping to a JSON file.

        Parameters
        ----------
        filepath : str
            Output file path.
        """

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=4)


    def load_mapping(
        self,
        filepath: str
    ) -> None:
        """
        Load category mapping from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the saved mapping.
        """

        with open(filepath, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        # JSON keys remain strings, but values may need conversion
        self.mapping = {
            key: int(value)
            for key, value in self.mapping.items()
        }

        self.inverse_mapping = {
            value: key
            for key, value in self.mapping.items()
        }