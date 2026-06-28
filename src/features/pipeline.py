"""
Feature Engineering Pipeline
============================

Orchestrates the complete feature engineering workflow for the Bayesian
Probabilistic Revenue Forecasting project.

Pipeline Stages
---------------
1. Deterministic Feature Engineering
2. Fourier Seasonality
3. Spend Response Transformations
4. Hierarchical Encoding
5. Numerical Feature Scaling

Example
-------
>>> pipeline = FeaturePipeline()
>>> train = pipeline.fit_transform(train_df)
>>> test = pipeline.transform(test_df)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import (
    STANDARD_SCALE_COLUMNS,
    STANDARD_SCALE_PREFIXES,
)

from .feature_engineering import FeatureEngineer
from .seasonality import FourierSeasonality
from .spend_response import SpendResponse
from .hierarchy import HierarchyEncoder
from .transformers import (
    StandardScaler,
    TransformationPipeline,
)


@dataclass(slots=True)
class FeaturePipelineConfig:
    """
    Configuration for the feature engineering pipeline.
    """

    spend_column: str = "spend"

    encoder_directory: str = (
        "data/artifacts/encoders"
    )

    scale_numeric_features: bool = True

    use_adstock: bool = True

    use_hill: bool = True

    use_logistic: bool = False


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    Responsibilities
    ----------------
    * Deterministic feature engineering
    * Fourier seasonality generation
    * Spend response transformations
    * Hierarchical encoding
    * Numerical scaling

    Notes
    -----
    This class orchestrates existing feature modules.
    It intentionally does not implement feature
    engineering logic itself.
    """

    def __init__(
        self,
        config: FeaturePipelineConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else FeaturePipelineConfig()
        )

        self.feature_engineer = FeatureEngineer()

        self.seasonality = FourierSeasonality()

        self.spend_response = SpendResponse()

        self.hierarchy = HierarchyEncoder()

        # Created during fit()
        self.transformer_pipeline = None

        self.is_fitted = False

        self.feature_columns: list[str] = []

    # ---------------------------------------------------------
    # Internal validation
    # ---------------------------------------------------------

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate required columns.

        Parameters
        ----------
        df : pd.DataFrame
        """

        required = [
            "date",
            "channel",
            "campaign_type",
            "campaign_id",
            "spend",
            "revenue",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _get_scaling_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Determine which columns should be standardized.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
        """

        columns = []

        for column in df.columns:

            if column in STANDARD_SCALE_COLUMNS:

                columns.append(column)

                continue

            if any(
                column.startswith(prefix)
                for prefix in STANDARD_SCALE_PREFIXES
            ):
                columns.append(column)

        return columns

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "FeaturePipeline":
        """
        Fit the complete feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe.

        Returns
        -------
        FeaturePipeline
        """

        self._validate_dataframe(df)

        data = df.copy()

        # -------------------------------------------------
        # 1. Deterministic Feature Engineering
        # -------------------------------------------------

        data = self.feature_engineer.transform(data)

        # -------------------------------------------------
        # 2. Fourier Seasonality
        # -------------------------------------------------

        data = self.seasonality.transform(data)

        # -------------------------------------------------
        # 3. Spend Response Features
        # -------------------------------------------------

        spend_features = self.spend_response.transform(
            data[self.config.spend_column],
            use_adstock=self.config.use_adstock,
            use_hill=self.config.use_hill,
            use_logistic=self.config.use_logistic,
        )

        # Remove duplicate spend column
        spend_features = spend_features.drop(
            columns=["raw_spend"],
            errors="ignore",
        )

        data = pd.concat(
            [
                data,
                spend_features,
            ],
            axis=1,
        )

        # -------------------------------------------------
        # 4. Hierarchy Encoding
        # -------------------------------------------------

        data = self.hierarchy.fit_transform(data)

        # -------------------------------------------------
        # 5. Numerical Scaling
        # -------------------------------------------------

        if self.config.scale_numeric_features:

            scaling_columns = self._get_scaling_columns(
                data
            )

            self.transformer_pipeline = (
                TransformationPipeline(
                    transformers=[
                        StandardScaler(
                            columns=scaling_columns
                        )
                    ]
                )
            )

            data = (
                self.transformer_pipeline
                .fit_transform(data)
            )

        # Save final feature names

        self.feature_columns = list(data.columns)

        self.is_fitted = True

        return self

    # ---------------------------------------------------------
    # Fit + Transform
    # ---------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit the pipeline and return the transformed
        dataframe.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        self.fit(df)

        return self.transform(df)

    # ---------------------------------------------------------
    # Transform
    # ---------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted."
            )

        self._validate_dataframe(df)

        data = df.copy()

        # Feature Engineering
        data = self.feature_engineer.transform(data)

        # Fourier Features
        data = self.seasonality.transform(data)

        # Spend Response
        spend_features = self.spend_response.transform(
            data[self.config.spend_column],
            use_adstock=self.config.use_adstock,
            use_hill=self.config.use_hill,
            use_logistic=self.config.use_logistic,
        )

        spend_features = spend_features.drop(
            columns=["raw_spend"],
            errors="ignore",
        )

        data = pd.concat(
            [data, spend_features],
            axis=1,
        )

        # Hierarchy Encoding
        data = self.hierarchy.transform(data)

        # Numerical Scaling
        if (
            self.config.scale_numeric_features
            and self.transformer_pipeline is not None
        ):
            data = self.transformer_pipeline.transform(data)

        # Match training feature order
        data = data.reindex(
            columns=self.feature_columns,
            fill_value=0,
        )

        return data


    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """
        Whether the pipeline has been fitted.
        """

        return self.is_fitted

    @property
    def n_features(self) -> int:
        """
        Number of engineered features.
        """

        return len(self.feature_columns)

    @property
    def hierarchy_metadata(self) -> dict:
        """
        Return hierarchy metadata.

        Useful when constructing Bayesian model
        coordinates and dimensions.
        """

        return self.hierarchy.metadata()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a summary of the fitted feature pipeline.

        Returns
        -------
        pd.DataFrame
            Pipeline summary.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted."
            )

        hierarchy = self.hierarchy.metadata()

        summary = pd.DataFrame(
            {
                "Property": [
                    "Total Features",
                    "Channels",
                    "Campaign Types",
                    "Campaigns",
                    "Scaling Enabled",
                    "Adstock Enabled",
                    "Hill Enabled",
                    "Logistic Enabled",
                ],
                "Value": [
                    self.n_features,
                    hierarchy["n_channels"],
                    hierarchy["n_campaign_types"],
                    hierarchy["n_campaigns"],
                    self.config.scale_numeric_features,
                    self.config.use_adstock,
                    self.config.use_hill,
                    self.config.use_logistic,
                ],
            }
        )

        return summary

    # ---------------------------------------------------------
    # Feature Utilities
    # ---------------------------------------------------------

    def feature_names(
        self,
    ) -> list[str]:
        """
        Return engineered feature names.

        Returns
        -------
        list[str]
        """

        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted."
            )

        return self.feature_columns.copy()

    def has_feature(
        self,
        feature_name: str,
    ) -> bool:
        """
        Check whether a feature exists.

        Parameters
        ----------
        feature_name : str

        Returns
        -------
        bool
        """

        return feature_name in self.feature_columns

    def get_scaling_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Public wrapper around the internal scaling
        column selector.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
        """

        return self._get_scaling_columns(df)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation.
        """

        return (
            f"{self.__class__.__name__}("
            f"fitted={self.is_fitted}, "
            f"features={self.n_features if self.is_fitted else 0})"
        )

    __str__ = __repr__