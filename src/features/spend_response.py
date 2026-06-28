"""
Spend Response Transformations
==============================

This module implements nonlinear marketing spend transformations
commonly used in Marketing Mix Modeling (MMM).

Implemented
-----------
- Geometric Adstock
- Hill Saturation
- Logistic Saturation

These transformations help model:

- Carry-over advertising effects
- Diminishing returns
- Saturation behaviour
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SpendResponse:
    """
    Marketing spend response transformations.

    Parameters
    ----------
    alpha : float
        Adstock decay parameter.

    hill_alpha : float
        Hill slope parameter.

    hill_gamma : float
        Hill half-saturation parameter.
    """

    alpha: float = 0.5

    hill_alpha: float = 2.0

    hill_gamma: float = 1000.0

    def geometric_adstock(
        self,
        spend: pd.Series
    ) -> pd.Series:
        """
        Apply geometric adstock.

        Parameters
        ----------
        spend : pd.Series

        Returns
        -------
        pd.Series
        """

        spend = spend.fillna(0)

        adstock = np.zeros(len(spend))

        adstock[0] = spend.iloc[0]

        for i in range(1, len(spend)):
            adstock[i] = (
                spend.iloc[i]
                + self.alpha * adstock[i - 1]
            )

        return pd.Series(
            adstock,
            index=spend.index,
            name="adstock_spend"
        )

    def hill_saturation(
        self,
        spend: pd.Series
    ) -> pd.Series:
        """
        Apply the Hill saturation function.

        Parameters
        ----------
        spend : pd.Series
            Advertising spend after optional adstock transformation.

        Returns
        -------
        pd.Series
            Hill-transformed spend.
        """

        spend = spend.clip(lower=0)

        numerator = np.power(spend, self.hill_alpha)

        denominator = (
            numerator +
            np.power(self.hill_gamma, self.hill_alpha)
        )

        hill = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0
        )

        return pd.Series(
            hill,
            index=spend.index,
            name="hill_spend"
        )

    def logistic_saturation(
        self,
        spend: pd.Series,
        k: float = 0.01,
        midpoint: float = 1000.0
    ) -> pd.Series:
        """
        Apply logistic saturation.

        Parameters
        ----------
        spend : pd.Series

        k : float
            Logistic growth rate.

        midpoint : float
            Inflection point.

        Returns
        -------
        pd.Series
        """

        spend = spend.clip(lower=0)

        logistic = 1 / (
            1 +
            np.exp(
                -k * (spend - midpoint)
            )
        )

        return pd.Series(
            logistic,
            index=spend.index,
            name="logistic_spend"
        )

    def transform(
        self,
        spend: pd.Series,
        use_adstock: bool = True,
        use_hill: bool = True,
        use_logistic: bool = False
    ) -> pd.DataFrame:
        """
        Apply the selected spend-response transformations.

        Parameters
        ----------
        spend : pd.Series
            Raw spend values.

        use_adstock : bool
            Whether to apply geometric adstock.

        use_hill : bool
            Whether to compute Hill saturation.

        use_logistic : bool
            Whether to compute Logistic saturation.

        Returns
        -------
        pd.DataFrame
            DataFrame containing transformed spend features.
        """

        transformed = pd.DataFrame(index=spend.index)

        current = spend.copy()

        transformed["raw_spend"] = current

        if use_adstock:
            current = self.geometric_adstock(current)
            transformed["adstock_spend"] = current

        if use_hill:
            transformed["hill_spend"] = self.hill_saturation(current)

        if use_logistic:
            transformed["logistic_spend"] = self.logistic_saturation(current)

        return transformed

