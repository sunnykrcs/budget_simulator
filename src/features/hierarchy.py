"""
Hierarchy Encoder
=================

Creates hierarchical integer indices required by the Bayesian model.

This module wraps three CategoryEncoder instances to encode:

- channel -> channel_idx
- campaign_type -> campaign_type_idx
- campaign_id -> campaign_idx

campaign_name is intentionally NOT encoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .encoders import CategoryEncoder


REQUIRED_COLUMNS = [
    "channel",
    "campaign_type",
    "campaign_id",
]


@dataclass(slots=True)
class HierarchyEncoder:
    """
    Encodes hierarchical categorical variables into integer indices.

    Attributes
    ----------
    channel_encoder : CategoryEncoder
    campaign_type_encoder : CategoryEncoder
    campaign_encoder : CategoryEncoder
    """

    channel_encoder: CategoryEncoder = field(default_factory=CategoryEncoder)
    campaign_type_encoder: CategoryEncoder = field(default_factory=CategoryEncoder)
    campaign_encoder: CategoryEncoder = field(default_factory=CategoryEncoder)

    fitted: bool = False

    def _validate_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate required columns exist.
        """

        missing = [
            col
            for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "HierarchyEncoder":
        """
        Fit all hierarchy encoders.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        HierarchyEncoder
        """

        self._validate_columns(df)

        self.channel_encoder.fit(df["channel"])

        self.campaign_type_encoder.fit(df["campaign_type"])

        self.campaign_encoder.fit(df["campaign_id"])

        self.fitted = True

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform dataframe using fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        if not self.fitted:
            raise RuntimeError(
                "HierarchyEncoder has not been fitted."
            )

        self._validate_columns(df)

        result = df.copy()

        result["channel_idx"] = (
            self.channel_encoder.transform(
                result["channel"]
            )
        )

        result["campaign_type_idx"] = (
            self.campaign_type_encoder.transform(
                result["campaign_type"]
            )
        )

        result["campaign_idx"] = (
            self.campaign_encoder.transform(
                result["campaign_id"]
            )
        )

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit encoders and transform dataframe.
        """

        self.fit(df)

        return self.transform(df)

    def save(
        self,
        directory: str,
    ) -> None:
        """
        Save all encoder mappings.

        Files created
        -------------

        channel.json

        campaign_type.json

        campaign.json
        """

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.channel_encoder.save_mapping(
            str(path / "channel.json")
        )

        self.campaign_type_encoder.save_mapping(
            str(path / "campaign_type.json")
        )

        self.campaign_encoder.save_mapping(
            str(path / "campaign.json")
        )

    @classmethod
    def load(
        cls,
        directory: str,
    ) -> "HierarchyEncoder":
        """
        Load encoder mappings from disk.
        """

        path = Path(directory)

        encoder = cls()

        encoder.channel_encoder.load_mapping(
            str(path / "channel.json")
        )

        encoder.campaign_type_encoder.load_mapping(
            str(path / "campaign_type.json")
        )

        encoder.campaign_encoder.load_mapping(
            str(path / "campaign.json")
        )

        encoder.fitted = True

        return encoder

    @property
    def n_channels(self) -> int:
        """
        Number of unique channels.
        """

        return len(
            self.channel_encoder.mapping
        )

    @property
    def n_campaign_types(self) -> int:
        """
        Number of unique campaign types.
        """

        return len(
            self.campaign_type_encoder.mapping
        )

    @property
    def n_campaigns(self) -> int:
        """
        Number of unique campaigns.
        """

        return len(
            self.campaign_encoder.mapping
        )

    def metadata(
        self,
    ) -> dict:
        """
        Return hierarchy metadata.

        Useful for defining PyMC model dimensions.
        """

        return {
            "n_channels": self.n_channels,
            "n_campaign_types": self.n_campaign_types,
            "n_campaigns": self.n_campaigns,
        }

    def summary(
        self,
    ) -> pd.DataFrame:
        """
        Return hierarchy summary.
        """

        return pd.DataFrame(
            {
                "Hierarchy": [
                    "Channel",
                    "Campaign Type",
                    "Campaign",
                ],
                "Unique Groups": [
                    self.n_channels,
                    self.n_campaign_types,
                    self.n_campaigns,
                ],
            }
        )

    def inverse_lookup(
        self,
        index: int,
        level: str,
    ) -> str:
        """
        Convert an encoded index back to its original value.

        Parameters
        ----------
        index : int

        level : str
            One of:
            - channel
            - campaign_type
            - campaign
        """

        if level == "channel":
            return self.channel_encoder.inverse_mapping[index]

        if level == "campaign_type":
            return self.campaign_type_encoder.inverse_mapping[index]

        if level == "campaign":
            return self.campaign_encoder.inverse_mapping[index]

        raise ValueError(
            "level must be one of "
            "['channel', 'campaign_type', 'campaign']"
        )