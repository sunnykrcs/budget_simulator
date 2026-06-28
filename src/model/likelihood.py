"""
Observation Likelihood
======================

This module defines the deterministic linear predictor and
the observation likelihood for the Bayesian hierarchical
revenue forecasting model.

Responsibilities
----------------
- Construct the expected revenue
- Combine hierarchical effects
- Define the observation likelihood

This module intentionally does NOT create a PyMC model or
perform posterior sampling.
"""

from __future__ import annotations

import pymc as pm


def build_linear_predictor(
    spend,
    channel_idx,
    campaign_type_idx,
    campaign_idx,
    intercept,
    beta_spend,
    channel_effect,
    campaign_type_effect,
    campaign_effect,
):
    """
    Construct the expected revenue.

    Model
    -----

    μ =
        α
        + β_spend * spend
        + β_channel[channel_idx]
        + β_campaign_type[campaign_type_idx]
        + β_campaign[campaign_idx]

    Parameters
    ----------
    spend
        Transformed spend feature.

    channel_idx
        Encoded channel indices.

    campaign_type_idx
        Encoded campaign-type indices.

    campaign_idx
        Encoded campaign indices.

    intercept
        Global intercept.

    beta_spend
        Global spend coefficient.

    channel_effect
        Channel-level effects.

    campaign_type_effect
        Campaign-type effects.

    campaign_effect
        Campaign-level effects.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Expected revenue.
    """

    mu = (
        intercept
        + beta_spend * spend
        + channel_effect[channel_idx]
        + campaign_type_effect[campaign_type_idx]
        + campaign_effect[campaign_idx]
    )

    return pm.Deterministic(
        "mu",
        mu,
    )


def build_revenue_likelihood(
    revenue,
    mu,
    sigma,
):
    """
    Define the Gaussian observation model.

    Parameters
    ----------
    revenue
        Observed revenue.

    mu
        Expected revenue.

    sigma
        Observation standard deviation.

    Returns
    -------
    pm.TensorVariable
        Revenue likelihood.
    """

    return pm.Normal(
        "revenue",
        mu=mu,
        sigma=sigma,
        observed=revenue,
    )