"""
Tests for src.model.bayesian_model
=================================

Run:

    python tests/test_bayesian_model.py

or

    pytest tests/test_bayesian_model.py -v
"""

import pandas as pd
import pymc as pm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.bayesian_model import BayesianRevenueModel


def create_test_dataframe():
    """Create a small encoded dataframe."""

    return pd.DataFrame(
        {
            "spend": [100.0, 200.0, 150.0, 250.0],
            "revenue": [120.0, 260.0, 170.0, 310.0],
            "channel_idx": [0, 0, 1, 1],
            "campaign_type_idx": [0, 1, 0, 1],
            "campaign_idx": [0, 1, 2, 3],
        }
    )


def test_model_build():

    df = create_test_dataframe()

    builder = BayesianRevenueModel()

    model = builder.build(df)

    print("\nModel successfully built.")

    assert isinstance(model, pm.Model)

    assert builder.is_built


def test_named_variables():

    df = create_test_dataframe()

    builder = BayesianRevenueModel()

    model = builder.build(df)

    variables = model.named_vars

    print("\nNamed Variables")
    print("----------------")

    for name in variables:
        print(name)

    expected = [
        "spend",
        "revenue",
        "intercept",
        "beta_spend",
        "sigma_channel",
        "channel_effect",
        "sigma_campaign_type",
        "campaign_type_effect",
        "sigma_campaign",
        "campaign_effect",
        "sigma",
        "mu",
    ]

    for variable in expected:

        assert variable in variables


def test_model_property():

    df = create_test_dataframe()

    builder = BayesianRevenueModel()

    builder.build(df)

    assert isinstance(
        builder.model,
        pm.Model,
    )


def test_repr():

    builder = BayesianRevenueModel()

    print(builder)

    assert "Not Built" in repr(builder)

    builder.build(create_test_dataframe())

    print(builder)

    assert "Built" in repr(builder)


if __name__ == "__main__":

    print("=" * 70)
    print("Testing bayesian_model.py")
    print("=" * 70)

    test_model_build()

    test_named_variables()

    test_model_property()

    test_repr()

    print("\n" + "=" * 70)
    print("All Bayesian Model Tests Passed Successfully")
    print("=" * 70)