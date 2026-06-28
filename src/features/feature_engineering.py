"""
Feature Engineering Module
==========================

This module contains deterministic feature engineering utilities for the
Bayesian Probabilistic Revenue Forecasting project.

Responsibilities
----------------
- Time-based features
- Marketing KPI features
- Spend transformations
- Revenue transformations
- Lag features
- Rolling statistics
- Data validation

Notes
-----
This module intentionally does NOT implement:

- Fourier seasonality
- Adstock
- Hill saturation
- Hierarchical encoding

Those belong in dedicated modules inside src/features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureEngineer:
    """
    Generate deterministic features used by the Bayesian forecasting model.

    Parameters
    ----------
    date_col : str
        Name of the date column.

    spend_col : str
        Name of the spend column.

    revenue_col : str
        Name of the revenue column.

    lag_periods : Iterable[int]
        Lag periods to generate.

    rolling_windows : Iterable[int]
        Rolling windows used for statistical summaries.
    """

    date_col: str = "date"
    spend_col: str = "spend"
    revenue_col: str = "revenue"

    lag_periods: Iterable[int] = field(
        default_factory=lambda: (1, 3, 7, 14)
    )

    rolling_windows: Iterable[int] = field(
        default_factory=lambda: (3, 7, 14, 30)
    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete deterministic feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        data = df.copy()

        data = self._validate_dataframe(data)

        data = self._generate_time_features(data)

        data = self._generate_marketing_features(data)

        data = self._generate_spend_features(data)

        data = self._generate_revenue_features(data)

        data = self._generate_lag_features(data)

        data = self._generate_rolling_statistics(data)

        data = self._validate_numeric_values(data)

        return data

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the input dataframe before feature engineering.

        This method:
        - Ensures the required columns exist.
        - Converts the date column to datetime.
        - Sorts the dataframe chronologically.
        - Resets the index.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Validated dataframe.
        """

        required_columns = [
            self.date_col,
            self.spend_col,
            self.revenue_col,
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df[self.date_col] = pd.to_datetime(df[self.date_col])

        df = (
            df.sort_values(self.date_col)
              .reset_index(drop=True)
        )

        return df

    def _generate_time_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate calendar-based time features.

        Features
        --------
        year
        quarter
        month
        week
        day
        day_of_week
        day_of_year
        trend

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        dt = df[self.date_col].dt

        df["year"] = dt.year

        df["quarter"] = dt.quarter

        df["month"] = dt.month

        # ISO week number
        df["week"] = (
            dt.isocalendar()
              .week
              .astype(int)
        )

        df["day"] = dt.day

        df["day_of_week"] = dt.dayofweek

        df["day_of_year"] = dt.dayofyear

        # Continuous trend index for downstream Bayesian trend modelling
        df["trend"] = np.arange(
            len(df),
            dtype=np.int32
        )

        return df

    def _generate_marketing_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate common digital marketing performance metrics.

        Features
        --------
        ctr
        cpc
        cpm
        conversion_rate
        roas

        Notes
        -----
        Metrics are only created if the required columns exist.
        Division-by-zero is safely handled by assigning 0.
        """

        # Click Through Rate
        if {"clicks", "impressions"}.issubset(df.columns):
            df["ctr"] = np.where(
                df["impressions"] > 0,
                df["clicks"] / df["impressions"],
                0.0,
            )

        # Cost Per Click
        if {"spend", "clicks"}.issubset(df.columns):
            df["cpc"] = np.where(
                df["clicks"] > 0,
                df["spend"] / df["clicks"],
                0.0,
            )

        # Cost Per Mille (1000 impressions)
        if {"spend", "impressions"}.issubset(df.columns):
            df["cpm"] = np.where(
                df["impressions"] > 0,
                (df["spend"] / df["impressions"]) * 1000,
                0.0,
            )

        # Conversion Rate
        if {"conversions", "clicks"}.issubset(df.columns):
            df["conversion_rate"] = np.where(
                df["clicks"] > 0,
                df["conversions"] / df["clicks"],
                0.0,
            )

        # Return on Ad Spend
        df["roas"] = np.where(
            df[self.spend_col] > 0,
            df[self.revenue_col] / df[self.spend_col],
            0.0,
        )

        return df


    def _generate_spend_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate deterministic spend transformations.

        Notes
        -----
        Hill saturation and Adstock are implemented separately
        in spend_response.py.
        """

        spend = df[self.spend_col].clip(lower=0)

        df["log_spend"] = np.log1p(spend)

        df["sqrt_spend"] = np.sqrt(spend)

        df["spend_squared"] = spend.pow(2)

        return df

    def _generate_revenue_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate revenue-based engineered features.

        Features
        --------
        log_revenue
        revenue_growth
        revenue_share

        Notes
        -----
        Revenue growth is calculated using percentage change.
        Revenue share represents the contribution of each row's
        revenue to the total dataset revenue.
        """

        revenue = df[self.revenue_col].clip(lower=0)

        # Log transformation
        df["log_revenue"] = np.log1p(revenue)

        # Percentage growth
        df["revenue_growth"] = (
            revenue
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # Share of total revenue
        total_revenue = revenue.sum()

        if total_revenue > 0:
            df["revenue_share"] = revenue / total_revenue
        else:
            df["revenue_share"] = 0.0

        return df


    def _generate_lag_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate lag features for spend and revenue.

        Parameters
        ----------
        lag_periods : Iterable[int]

        Default
        -------
        1, 3, 7, 14
        """

        for lag in self.lag_periods:

            df[f"spend_lag_{lag}"] = (
                df[self.spend_col]
                .shift(lag)
            )

            df[f"revenue_lag_{lag}"] = (
                df[self.revenue_col]
                .shift(lag)
            )

        return df

    def _generate_rolling_statistics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate rolling statistical features for spend and revenue.

        Statistics
        ----------
        mean
        std
        min
        max

        Windows
        -------
        3
        7
        14
        30
        """

        feature_map = {
            self.spend_col: "spend",
            self.revenue_col: "revenue",
        }

        for column, prefix in feature_map.items():

            for window in self.rolling_windows:

                rolling = (
                    df[column]
                    .rolling(
                        window=window,
                        min_periods=1
                    )
                )

                df[f"{prefix}_mean_{window}"] = rolling.mean()

                df[f"{prefix}_std_{window}"] = (
                    rolling.std()
                    .fillna(0)
                )

                df[f"{prefix}_min_{window}"] = rolling.min()

                df[f"{prefix}_max_{window}"] = rolling.max()

        return df


    def _validate_numeric_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace invalid numeric values produced during
        feature engineering.

        Operations
        ----------
        Replace:
            NaN
            +Inf
            -Inf

        With:
            0
        """

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        df[numeric_columns] = (
            df[numeric_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        return df