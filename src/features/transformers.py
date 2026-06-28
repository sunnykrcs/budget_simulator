"""
Feature Transformers
====================

Reusable numerical transformers for the Bayesian Revenue Forecasting
pipeline.

Each transformer follows a scikit-learn–style API:

    fit()
    transform()
    fit_transform()
    inverse_transform()

These transformers are later composed inside
features/pipeline.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Base Transformer
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BaseTransformer(ABC):
    """
    Abstract base class for all transformers.
    """

    columns: list[str]

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
    ) -> "BaseTransformer":
        pass

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(df)

        return self.transform(df)

    @abstractmethod
    def inverse_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass


# ---------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------


@dataclass(slots=True)
class IdentityTransformer(BaseTransformer):
    """
    No-op transformer.
    """

    def fit(self, df: pd.DataFrame):

        return self

    def transform(self, df: pd.DataFrame):

        return df.copy()

    def inverse_transform(self, df: pd.DataFrame):

        return df.copy()


# ---------------------------------------------------------------------
# Log Transformer
# ---------------------------------------------------------------------


@dataclass(slots=True)
class LogTransformer(BaseTransformer):
    """
    Applies log1p transformation.
    """

    def fit(self, df: pd.DataFrame):

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = df.copy()

        for col in self.columns:

            result[col] = np.log1p(result[col])

        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = df.copy()

        for col in self.columns:

            result[col] = np.expm1(result[col])

        return result


# ---------------------------------------------------------------------
# Standard Scaler
# ---------------------------------------------------------------------


@dataclass(slots=True)
class StandardScaler(BaseTransformer):
    """
    Standard score normalization.
    """

    means: dict = field(default_factory=dict)

    stds: dict = field(default_factory=dict)

    def fit(
        self,
        df: pd.DataFrame,
    ):

        for col in self.columns:

            self.means[col] = df[col].mean()

            std = df[col].std()

            self.stds[col] = std if std != 0 else 1.0

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            result[col] = (
                result[col] - self.means[col]
            ) / self.stds[col]

        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            result[col] = (
                result[col] * self.stds[col]
            ) + self.means[col]

        return result


# ---------------------------------------------------------------------
# Min-Max Scaler
# ---------------------------------------------------------------------


@dataclass(slots=True)
class MinMaxScaler(BaseTransformer):
    """
    Scale features into [0,1].
    """

    mins: dict = field(default_factory=dict)

    maxs: dict = field(default_factory=dict)

    def fit(
        self,
        df: pd.DataFrame,
    ):

        for col in self.columns:

            self.mins[col] = df[col].min()

            self.maxs[col] = df[col].max()

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            denom = self.maxs[col] - self.mins[col]

            if denom == 0:
                denom = 1

            result[col] = (
                result[col] - self.mins[col]
            ) / denom

        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            result[col] = (
                result[col]
                * (self.maxs[col] - self.mins[col])
            ) + self.mins[col]

        return result


# ---------------------------------------------------------------------
# Robust Scaler
# ---------------------------------------------------------------------


@dataclass(slots=True)
class RobustScaler(BaseTransformer):
    """
    Robust scaling using median and IQR.
    """

    medians: dict = field(default_factory=dict)

    iqrs: dict = field(default_factory=dict)

    def fit(
        self,
        df: pd.DataFrame,
    ):

        for col in self.columns:

            q1 = df[col].quantile(0.25)

            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            self.medians[col] = df[col].median()

            self.iqrs[col] = iqr if iqr != 0 else 1.0

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            result[col] = (
                result[col] - self.medians[col]
            ) / self.iqrs[col]

        return result

    def inverse_transform(
        self,
        df: pd.DataFrame,
    ):

        result = df.copy()

        for col in self.columns:

            result[col] = (
                result[col] * self.iqrs[col]
            ) + self.medians[col]

        return result


# ---------------------------------------------------------------------
# Transformation Pipeline
# ---------------------------------------------------------------------


@dataclass(slots=True)
class TransformationPipeline:
    """
    Sequential transformer pipeline.
    """

    transformers: list[BaseTransformer]

    # def fit(
    #     self,
    #     df: pd.DataFrame,
    # ) -> "TransformationPipeline":

    #     current = df.copy()

    #     for transformer in self.transformers:

    #         current = transformer.fit_transform(current)

    #     return self

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "TransformationPipeline":

        current = df.copy()

        for transformer in self.transformers:
            transformer.fit(current)
            current = transformer.transform(current)

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        current = df.copy()

        for transformer in self.transformers:

            current = transformer.transform(current)

        return current

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        current = df.copy()

        for transformer in self.transformers:

            current = transformer.fit_transform(current)

        return current

    def inverse_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        current = df.copy()

        for transformer in reversed(self.transformers):

            current = transformer.inverse_transform(current)

        return current