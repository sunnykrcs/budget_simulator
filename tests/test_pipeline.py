"""
Tests for FeaturePipeline.
"""

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FeaturePipeline


def create_dataset() -> pd.DataFrame:
    """Create a small synthetic dataset."""

    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=20,
            ),
            "channel": [
                "Google",
                "Meta",
            ] * 10,
            "campaign_type": [
                "Search",
                "Performance Max",
            ] * 10,
            "campaign_id": [
                "C001",
                "C002",
            ] * 10,
            "campaign_name": [
                "Campaign A",
                "Campaign B",
            ] * 10,
            "spend": [
                100,
                150,
                120,
                180,
                140,
                200,
                160,
                220,
                180,
                240,
                200,
                260,
                220,
                280,
                240,
                300,
                260,
                320,
                280,
                340,
            ],
            "revenue": [
                300,
                420,
                330,
                480,
                360,
                540,
                390,
                600,
                420,
                660,
                450,
                720,
                480,
                780,
                510,
                840,
                540,
                900,
                570,
                960,
            ],
            "clicks": list(range(20, 40)),
            "impressions": list(range(1000, 1020)),
            "conversions": list(range(5, 25)),
            "roas": [3.0] * 20,
        }
    )


def main():

    df = create_dataset()

    print("=" * 70)
    print("Original Dataset")
    print("=" * 70)

    print(df.head())

    pipeline = FeaturePipeline()

    transformed = pipeline.fit_transform(df)

    print("\n")
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)

    print(pipeline.summary())

    print("\n")
    print("=" * 70)
    print("Transformed Dataset")
    print("=" * 70)

    print(transformed.head())

    print("\n")
    print("=" * 70)
    print("Shape")
    print("=" * 70)

    print(transformed.shape)

    print("\n")
    print("=" * 70)
    print("Feature Count")
    print("=" * 70)

    print(pipeline.n_features)

    print("\n")
    print("=" * 70)
    print("Hierarchy Metadata")
    print("=" * 70)

    print(pipeline.hierarchy_metadata)

    print("\n")
    print("=" * 70)
    print("Feature Names")
    print("=" * 70)

    print(pipeline.feature_names())

    print("\n")
    print("=" * 70)
    print("Testing transform()")
    print("=" * 70)

    new_df = create_dataset()

    new_features = pipeline.transform(new_df)

    print(new_features.head())

    print("Transform successful!")

    print("\n")
    print("=" * 70)
    print("Scaled Spend Mean")
    print("=" * 70)

    print(round(transformed["spend"].mean(), 6))

    print(round(transformed["spend"].std(), 6))

    assert pipeline.hierarchy_metadata == {
        "n_channels": 2,
        "n_campaign_types": 2,
        "n_campaigns": 2,
    }

if __name__ == "__main__":
    main()