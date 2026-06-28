"""
Test sampler.py
===============

This script verifies that the Bayesian sampler can

- Build the Bayesian model
- Run posterior sampling
- Generate posterior predictive samples
- Return an InferenceData object

Run

    python tests/test_sampler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Allow importing from src/
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from src.model.bayesian_model import BayesianRevenueModel
from src.model.sampler import (
    BayesianSampler,
    SamplerConfig,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def create_dataset(
    n: int = 60,
) -> pd.DataFrame:
    """
    Create a synthetic dataset for testing.
    """

    rng = np.random.default_rng(42)

    spend = rng.uniform(20, 200, n)

    revenue = (
        15
        + 3.2 * spend
        + rng.normal(0, 20, n)
    )

    return pd.DataFrame(
        {
            "spend": spend,
            "revenue": revenue,
            "channel_idx": rng.integers(0, 3, n),
            "campaign_type_idx": rng.integers(0, 2, n),
            "campaign_idx": rng.integers(0, 8, n),
        }
    )


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("Testing sampler.py")
    print("=" * 70)

    df = create_dataset()

    print("\nDataset")
    print("-" * 70)
    print(df.head())

    # -------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------

    print("\nBuilding Bayesian model...")

    model = BayesianRevenueModel()

    model.build(df)

    print("✓ Model built successfully.")

    # -------------------------------------------------------------
    # Configure sampler
    # -------------------------------------------------------------

    config = SamplerConfig(
        draws=100,
        tune=100,
        chains=2,
        cores=1,
        target_accept=0.90,
        posterior_predictive=True,
        random_seed=42,
    )

    sampler = BayesianSampler(
        model=model,
        config=config,
    )

    # -------------------------------------------------------------
    # Sample posterior
    # -------------------------------------------------------------

    print("\nRunning posterior sampling...")

    idata = sampler.fit()

    print("✓ Sampling completed.")

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    assert isinstance(
        idata,
        az.InferenceData,
    )

    assert sampler.is_sampled

    print("\nInferenceData Groups")
    print("-" * 70)

    for group in idata.groups():
        print(group)

    print("\nPosterior Variables")
    print("-" * 70)

    print(
        list(
            idata.posterior.data_vars
        )
    )

    if hasattr(
        idata,
        "posterior_predictive",
    ):

        print("\nPosterior Predictive Variables")
        print("-" * 70)

        print(
            list(
                idata.posterior_predictive.data_vars
            )
        )

    print("\nPosterior Summary")
    print("-" * 70)

    print(
        sampler.diagnostics().head()
    )

    print("\nSampler Status")
    print("-" * 70)

    print(sampler)

    print("\nInferenceData")
    print("-" * 70)

    print(sampler.summary())

    print("\n" + "=" * 70)
    print("sampler.py PASSED")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()