"""
Tests for src.model.priors
==========================

Run

    python tests/test_priors.py

or

    pytest tests/test_priors.py -v
"""

import pymc as pm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.priors import (
    PriorConfig,
    build_campaign_prior,
    build_campaign_type_prior,
    build_channel_prior,
    build_global_intercept,
    build_observation_sigma,
    build_spend_coefficient,
)


def test_prior_config():
    """Test PriorConfig validation."""

    config = PriorConfig()

    config.validate()

    print("\nPriorConfig validation passed.")


def test_invalid_prior_config():
    """Negative sigma should fail."""

    try:
        config = PriorConfig(
            intercept_sigma=-1.0,
        )

        config.validate()

    except ValueError:

        print("Invalid configuration correctly detected.")

        return

    raise AssertionError(
        "PriorConfig should raise ValueError."
    )


def test_global_priors():
    """Test intercept and spend priors."""

    config = PriorConfig()

    with pm.Model():

        intercept = build_global_intercept(config)

        beta_spend = build_spend_coefficient(config)

        variables = list(pm.modelcontext(None).named_vars.keys())

        print("\nGlobal Priors")
        print("----------------")

        print("Intercept :", intercept)

        print("Spend Beta:", beta_spend)

        assert "intercept" in variables

        assert "beta_spend" in variables


def test_channel_prior():
    """Test channel prior."""

    config = PriorConfig()

    coords = {
        "channel": range(3),
    }

    with pm.Model(coords=coords):

        channel = build_channel_prior(config)

        variables = list(pm.modelcontext(None).named_vars.keys())

        print("\nChannel Prior")
        print("----------------")

        print(channel)

        assert "sigma_channel" in variables

        assert "channel_effect" in variables


def test_campaign_type_prior():
    """Test campaign-type prior."""

    config = PriorConfig()

    coords = {
        "campaign_type": range(4),
    }

    with pm.Model(coords=coords):

        campaign_type = build_campaign_type_prior(
            config,
        )

        variables = list(pm.modelcontext(None).named_vars.keys())

        print("\nCampaign Type Prior")
        print("----------------------")

        print(campaign_type)

        assert "sigma_campaign_type" in variables

        assert "campaign_type_effect" in variables


def test_campaign_prior():
    """Test campaign prior."""

    config = PriorConfig()

    coords = {
        "campaign": range(5),
    }

    with pm.Model(coords=coords):

        campaign = build_campaign_prior(config)

        variables = list(pm.modelcontext(None).named_vars.keys())

        print("\nCampaign Prior")
        print("----------------")

        print(campaign)

        assert "sigma_campaign" in variables

        assert "campaign_effect" in variables


def test_observation_sigma():
    """Test observation sigma prior."""

    config = PriorConfig()

    with pm.Model():

        sigma = build_observation_sigma(config)

        variables = list(pm.modelcontext(None).named_vars.keys())

        print("\nObservation Sigma")
        print("--------------------")

        print(sigma)

        assert "sigma" in variables


if __name__ == "__main__":

    print("=" * 70)
    print("Testing priors.py")
    print("=" * 70)

    test_prior_config()

    test_invalid_prior_config()

    test_global_priors()

    test_channel_prior()

    test_campaign_type_prior()

    test_campaign_prior()

    test_observation_sigma()

    print("\n" + "=" * 70)
    print("All Prior Tests Passed Successfully")
    print("=" * 70)