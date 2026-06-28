"""
Posterior Sampler
=================

This module performs posterior inference for the Bayesian
Hierarchical Revenue Forecasting model.

Responsibilities
----------------
- Validate sampling configuration
- Perform posterior sampling
- Optionally sample posterior predictive distributions
- Store inference results
- Expose a reusable sampling interface

This module intentionally does NOT build the Bayesian model.
Model construction is implemented in ``bayesian_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import arviz as az
import pymc as pm

from src.model.bayesian_model import BayesianRevenueModel


# ---------------------------------------------------------------------
# Sampler configuration
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SamplerConfig:
    """
    Configuration for Bayesian posterior sampling.

    Parameters
    ----------
    draws
        Number of posterior samples.

    tune
        Number of warm-up iterations.

    chains
        Number of independent MCMC chains.

    cores
        Number of CPU cores used for sampling.

    target_accept
        Target acceptance probability for NUTS.

    random_seed
        Random seed for reproducibility.

    posterior_predictive
        Whether posterior predictive samples should be drawn.

    idata_kwargs
        Additional keyword arguments forwarded to
        ``pm.sample``.
    """

    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 4

    target_accept: float = 0.90

    random_seed: Optional[int] = 42

    posterior_predictive: bool = True

    idata_kwargs: Optional[dict] = None

    def validate(self) -> None:
        """
        Validate sampling configuration.

        Raises
        ------
        ValueError
            If any configuration value is invalid.
        """

        if self.draws <= 0:
            raise ValueError(
                "draws must be greater than zero."
            )

        if self.tune < 0:
            raise ValueError(
                "tune cannot be negative."
            )

        if self.chains <= 0:
            raise ValueError(
                "chains must be greater than zero."
            )

        if self.cores <= 0:
            raise ValueError(
                "cores must be greater than zero."
            )

        if not (0.5 < self.target_accept < 1.0):
            raise ValueError(
                "target_accept must lie between "
                "0.5 and 1.0."
            )


# ---------------------------------------------------------------------
# Bayesian sampler
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BayesianSampler:
    """
    Bayesian posterior sampler.

    Parameters
    ----------
    model
        Built Bayesian revenue model.

    config
        Sampling configuration.
    """

    model: BayesianRevenueModel

    config: SamplerConfig = field(
        default_factory=SamplerConfig
    )

    _idata: Optional[az.InferenceData] = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate sampler initialization.
        """

        self.config.validate()

        if not self.model.is_built:
            raise RuntimeError(
                "Bayesian model has not been built."
            )

    @property
    def inference_data(
        self,
    ) -> az.InferenceData:
        """
        Return posterior samples.

        Returns
        -------
        arviz.InferenceData

        Raises
        ------
        RuntimeError
            If sampling has not yet been performed.
        """

        if self._idata is None:
            raise RuntimeError(
                "Model has not been sampled."
            )

        return self._idata

    @property
    def is_sampled(self) -> bool:
        """
        Whether posterior sampling has completed.

        Returns
        -------
        bool
        """

        return self._idata is not None

    def fit(
        self,
    ) -> az.InferenceData:
        """
        Run Bayesian posterior sampling.

        Returns
        -------
        arviz.InferenceData
            Posterior samples.
        """

        with self.model.model:

            idata = pm.sample(
                draws=self.config.draws,
                tune=self.config.tune,
                chains=self.config.chains,
                cores=self.config.cores,
                target_accept=self.config.target_accept,
                random_seed=self.config.random_seed,
                return_inferencedata=True,
                idata_kwargs=(
                    self.config.idata_kwargs
                    if self.config.idata_kwargs
                    is not None
                    else {
                        "log_likelihood": True,
                    }
                ),
            )

            if self.config.posterior_predictive:

                posterior_predictive = (
                    pm.sample_posterior_predictive(
                        idata,
                        random_seed=self.config.random_seed,
                    )
                )

                idata.extend(
                    posterior_predictive,
                    join="left",
                )

        self._idata = idata

        return self._idata

    def summary(
        self,
    ) -> az.data.inference_data.InferenceData:
        """
        Return the sampled inference data.

        This is a convenience wrapper around the
        ``inference_data`` property.

        Returns
        -------
        arviz.InferenceData
            Posterior inference data.
        """

        return self.inference_data

    def diagnostics(self):
        """
        Generate posterior diagnostics.

        Returns
        -------
        pandas.DataFrame
            ArviZ summary statistics.
        """

        return az.summary(
            self.inference_data,
            round_to=3,
        )

    def __repr__(
        self,
    ) -> str:
        """
        String representation of the sampler.

        Returns
        -------
        str
            Readable sampler status.
        """

        status = (
            "Sampled"
            if self.is_sampled
            else "Not Sampled"
        )

        return (
            f"{self.__class__.__name__}"
            f"(status='{status}')"
        )


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "SamplerConfig",
    "BayesianSampler",
]