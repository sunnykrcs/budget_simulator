"""
Seasonality Feature Engineering
===============================

Generate Fourier series features for modeling periodic seasonality.

This module supports:

- Yearly seasonality
- Monthly seasonality

The generated features can be directly used as predictors in the
Bayesian forecasting model.

Example
-------
>>> generator = FourierSeasonality(yearly_order=3)
>>> df = generator.transform(df)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(slots=True)
class FourierSeasonality:
    """
    Generate configurable Fourier seasonal features.

    Parameters
    ----------
    yearly_order : int
        Number of yearly Fourier components.

    monthly_order : int
        Number of monthly Fourier components.

    date_col : str
        Name of date column.
    """

    yearly_order: int = 3
    monthly_order: int = 2

    date_col: str = "date"

    def transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate all configured Fourier features.

        Returns
        -------
        pd.DataFrame
        """

        data = df.copy()

        data = self._yearly_fourier(data)

        data = self._monthly_fourier(data)

        return data

    def _yearly_fourier(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate yearly Fourier series features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        day_of_year = df[self.date_col].dt.dayofyear

        period = 365.25

        for k in range(1, self.yearly_order + 1):

            angle = (
                2
                * np.pi
                * k
                * day_of_year
                / period
            )

            df[f"yearly_sin_{k}"] = np.sin(angle)

            df[f"yearly_cos_{k}"] = np.cos(angle)

        return df


    def _monthly_fourier(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate monthly Fourier series features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        day = df[self.date_col].dt.day

        period = 30.44

        for k in range(1, self.monthly_order + 1):

            angle = (
                2
                * np.pi
                * k
                * day
                / period
            )

            df[f"monthly_sin_{k}"] = np.sin(angle)

            df[f"monthly_cos_{k}"] = np.cos(angle)

        return df

if __name__ == "__main__":

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5)
    })

    generator = FourierSeasonality()

    result = generator.transform(df)

    print(result.head())