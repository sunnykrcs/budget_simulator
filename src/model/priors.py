"""
Bayesian Prior Definitions
==========================

This module defines reusable prior distributions for the hierarchical
Bayesian revenue forecasting model.

The priors defined here are intentionally separated from the model
construction code so that:

- prior assumptions are centralized
- hyperparameters are configurable
- model code remains concise
- future model variants can reuse the same priors

Hierarchy
---------
Global
    ├── Intercept
    ├── Spend coefficient
    ├── Channel effects
    ├── Campaign type effects
    └── Campaign effects

The observation likelihood is implemented separately in
``likelihood.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymc as pm

from src.model.coords import (
    CAMPAIGN_DIM,
    CAMPAIGN_TYPE_DIM,
    CHANNEL_DIM,
)


# ---------------------------------------------------------------------
# Prior configuration
# ---------------------------------------------------------------------


@dataclass(slots=True)
class PriorConfig:
    """
    Hyperparameter configuration for Bayesian priors.

    Parameters
    ----------
    intercept_sigma
        Standard deviation of the global intercept prior.

    spend_sigma
        Standard deviation of the spend coefficient prior.

    channel_sigma
        Scale of the HalfNormal hyperprior for channel effects.

    campaign_type_sigma
        Scale of the HalfNormal hyperprior for campaign type effects.

    campaign_sigma
        Scale of the HalfNormal hyperprior for campaign effects.

    observation_sigma
        Scale of the observation noise prior.
    """

    intercept_sigma: float = 10.0
    spend_sigma: float = 5.0

    channel_sigma: float = 2.0
    campaign_type_sigma: float = 2.0
    campaign_sigma: float = 2.0

    observation_sigma: float = 5.0

    def validate(self) -> None:
        """
        Validate prior hyperparameters.

        Raises
        ------
        ValueError
            If any scale parameter is non-positive.
        """

        parameters = {
            "intercept_sigma": self.intercept_sigma,
            "spend_sigma": self.spend_sigma,
            "channel_sigma": self.channel_sigma,
            "campaign_type_sigma": self.campaign_type_sigma,
            "campaign_sigma": self.campaign_sigma,
            "observation_sigma": self.observation_sigma,
        }

        for name, value in parameters.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )


# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------




def _validate_config(
    config: PriorConfig,
) -> None:
    """
    Validate the prior configuration.

    Parameters
    ----------
    config
        Prior configuration object.
    """

    config.validate()


# ---------------------------------------------------------------------
# Global priors
# ---------------------------------------------------------------------


def build_global_intercept(
    config: PriorConfig,
):
    """
    Create the global intercept prior.

    Model
    -----

    α ~ Normal(0, intercept_sigma)

    Parameters
    ----------
    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
        Global intercept random variable.
    """

    _validate_config(config)

    return pm.Normal(
        "intercept",
        mu=0.0,
        sigma=config.intercept_sigma,
    )


def build_spend_coefficient(
    config: PriorConfig,
):
    """
    Prior for the global spend coefficient.

    Model
    -----

    β_spend ~ Normal(0, spend_sigma)

    Parameters
    ----------
    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
    """

    _validate_config(config)

    return pm.Normal(
        "beta_spend",
        mu=0.0,
        sigma=config.spend_sigma,
    )

# ---------------------------------------------------------------------
# Hierarchical priors
# ---------------------------------------------------------------------


def build_channel_prior(
    config: PriorConfig,
):
    """
    Create hierarchical priors for channel effects.

    Model
    -----
    σ_channel ~ HalfNormal(channel_sigma)

    β_channel ~ Normal(0, σ_channel)

    Parameters
    ----------
    n_channels
        Number of unique channels.

    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
        Channel-level effects.
    """

    _validate_config(config)

    sigma_channel = pm.HalfNormal(
        "sigma_channel",
        sigma=config.channel_sigma,
    )

    return pm.Normal(
        "channel_effect",
        mu=0.0,
        sigma=sigma_channel,
        dims=CHANNEL_DIM,
    )


def build_campaign_type_prior(
    config: PriorConfig,
):
    """
    Create hierarchical priors for campaign type effects.

    Model
    -----
    σ_campaign_type ~ HalfNormal(campaign_type_sigma)

    β_campaign_type ~ Normal(0, σ_campaign_type)

    Parameters
    ----------
    n_campaign_types
        Number of campaign types.

    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
        Campaign-type effects.
    """


    _validate_config(config)

    sigma_campaign_type = pm.HalfNormal(
        "sigma_campaign_type",
        sigma=config.campaign_type_sigma,
    )

    return pm.Normal(
        "campaign_type_effect",
        mu=0.0,
        sigma=sigma_campaign_type,
        dims=CAMPAIGN_TYPE_DIM,
    )


def build_campaign_prior(
    config: PriorConfig,
):
    """
    Create hierarchical priors for campaign effects.

    Model
    -----
    σ_campaign ~ HalfNormal(campaign_sigma)

    β_campaign ~ Normal(0, σ_campaign)

    Parameters
    ----------
    n_campaigns
        Number of campaigns.

    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
        Campaign-level effects.
    """


    _validate_config(config)

    sigma_campaign = pm.HalfNormal(
        "sigma_campaign",
        sigma=config.campaign_sigma,
    )

    return pm.Normal(
        "campaign_effect",
        mu=0.0,
        sigma=sigma_campaign,
        dims=CAMPAIGN_DIM,
    )

# ---------------------------------------------------------------------
# Observation noise prior
# ---------------------------------------------------------------------


def build_observation_sigma(
    config: PriorConfig,
):
    """
    Create the observation noise prior.

    Model
    -----
    σ ~ HalfNormal(observation_sigma)

    Parameters
    ----------
    config
        Prior configuration.

    Returns
    -------
    pm.TensorVariable
        Observation noise standard deviation.
    """

    _validate_config(config)

    return pm.HalfNormal(
        "sigma",
        sigma=config.observation_sigma,
    )


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "PriorConfig",
    "build_global_intercept",
    "build_spend_coefficient",
    "build_channel_prior",
    "build_campaign_type_prior",
    "build_campaign_prior",
    "build_observation_sigma",
]