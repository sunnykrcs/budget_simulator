"""
Bayesian Hierarchical Revenue Forecasting Model
==============================================

This module assembles the complete Bayesian hierarchical revenue
forecasting model using reusable statistical components.

Responsibilities
----------------
- Build PyMC coordinates
- Register model data
- Create hierarchical priors
- Construct the linear predictor
- Define the observation likelihood

This module intentionally does NOT perform posterior sampling.
Sampling is implemented in ``sampler.py``.
"""

from __future__ import annotations

from dataclasses import dataclass,field
from typing import Optional

import pandas as pd
import pymc as pm

from src.model.coords import (
    build_coords,
    validate_hierarchy,
)
from src.model.priors import (
    PriorConfig,
    build_campaign_prior,
    build_campaign_type_prior,
    build_channel_prior,
    build_global_intercept,
    build_observation_sigma,
    build_spend_coefficient,
)
from src.model.likelihood import (
    build_linear_predictor,
    build_revenue_likelihood,
)


# ---------------------------------------------------------------------
# Required dataframe columns
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "spend",
    "revenue",
    "channel_idx",
    "campaign_type_idx",
    "campaign_idx",
)


# ---------------------------------------------------------------------
# Bayesian model
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BayesianRevenueModel:
    """
    Bayesian hierarchical revenue forecasting model.

    Parameters
    ----------
    prior_config
        Configuration of all Bayesian priors.
    """

    prior_config: PriorConfig = field(
    default_factory=PriorConfig
    )

    _model: Optional[pm.Model] = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate prior configuration."""

        self.prior_config.validate()


    @property
    def model(self) -> pm.Model:
        """
        Return the constructed PyMC model.

        Raises
        ------
        RuntimeError
            If the model has not been built.
        """

        if self._model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        return self._model

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate that the dataframe contains all columns
        required for model construction.

        Parameters
        ----------
        df
            Feature-engineered dataframe.

        Raises
        ------
        KeyError
            If required columns are missing.
        """

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise KeyError(
                "Missing required columns: "
                f"{', '.join(missing)}"
            )

        validate_hierarchy(df)

    def build(
        self,
        df: pd.DataFrame,
    ) -> pm.Model:
        """
        Construct the Bayesian hierarchical model.

        Parameters
        ----------
        df
            Feature-engineered dataframe.

        Returns
        -------
        pm.Model
            Fully constructed PyMC model.
        """

        self._validate_dataframe(df)

        coords = build_coords(df)

        with pm.Model(coords=coords) as model:

            # -------------------------------------------------
            # Register model data
            # -------------------------------------------------

            spend = pm.Data(
                "spend",
                df["spend"].to_numpy(),
            )

            revenue = df["revenue"].to_numpy()

            channel_idx = pm.Data(
                "channel_idx",
                df["channel_idx"].to_numpy(),
            )

            campaign_type_idx = pm.Data(
                "campaign_type_idx",
                df["campaign_type_idx"].to_numpy(),
            )

            campaign_idx = pm.Data(
                "campaign_idx",
                df["campaign_idx"].to_numpy(),
            )

            # -------------------------------------------------
            # Priors
            # -------------------------------------------------

            intercept = build_global_intercept(
                self.prior_config
            )

            beta_spend = build_spend_coefficient(
                self.prior_config
            )

            channel_effect = build_channel_prior(
                self.prior_config
            )

            campaign_type_effect = (
                build_campaign_type_prior(
                    self.prior_config
                )
            )

            campaign_effect = build_campaign_prior(
                self.prior_config
            )

            sigma = build_observation_sigma(
                self.prior_config
            )

            # -------------------------------------------------
            # Linear predictor
            # -------------------------------------------------

            mu = build_linear_predictor(
                spend=spend,
                channel_idx=channel_idx,
                campaign_type_idx=campaign_type_idx,
                campaign_idx=campaign_idx,
                intercept=intercept,
                beta_spend=beta_spend,
                channel_effect=channel_effect,
                campaign_type_effect=campaign_type_effect,
                campaign_effect=campaign_effect,
            )

            # -------------------------------------------------
            # Observation model
            # -------------------------------------------------

            build_revenue_likelihood(
                revenue=revenue,
                mu=mu,
                sigma=sigma,
            )

            self._model = model

        return self._model

    @property
    def is_built(self) -> bool:
        """
        Check whether the Bayesian model has been built.

        Returns
        -------
        bool
            True if the model has been constructed,
            otherwise False.
        """

        return self._model is not None

    def __repr__(self) -> str:
        """
        String representation of the Bayesian model.

        Returns
        -------
        str
            Readable summary of the model state.
        """

        status = (
            "Built"
            if self.is_built
            else "Not Built"
        )

        return (
            f"{self.__class__.__name__}"
            f"(status='{status}')"
        )


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "BayesianRevenueModel",
]