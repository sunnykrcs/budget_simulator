import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hierarchy import HierarchyEncoder


def main():

    # Load processed dataset
    df = pd.read_csv("data/master_dataset.csv")

    print("=" * 70)
    print("Original Dataset")
    print("=" * 70)
    print(df.head())

    # Create encoder
    encoder = HierarchyEncoder()

    # Fit + Transform
    df_encoded = encoder.fit_transform(df)

    print("\n" + "=" * 70)
    print("Encoded Columns")
    print("=" * 70)

    print(
        df_encoded[
            [
                "channel",
                "channel_idx",
                "campaign_type",
                "campaign_type_idx",
                "campaign_id",
                "campaign_idx",
            ]
        ].head(10)
    )

    # Metadata
    print("\n" + "=" * 70)
    print("Metadata")
    print("=" * 70)

    print(encoder.metadata())

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(encoder.summary())

    # Save mappings
    encoder.save("data/artifacts/encoders")

    print("\n✓ Encoder mappings saved.")

    # Load mappings
    loaded_encoder = HierarchyEncoder.load(
        "data/artifacts/encoders"
    )

    print("✓ Encoder mappings loaded.")

    # Transform again
    df_loaded = loaded_encoder.transform(df)

    # Verify identical encodings
    assert (
        df_encoded["channel_idx"]
        == df_loaded["channel_idx"]
    ).all()

    assert (
        df_encoded["campaign_type_idx"]
        == df_loaded["campaign_type_idx"]
    ).all()

    assert (
        df_encoded["campaign_idx"]
        == df_loaded["campaign_idx"]
    ).all()

    print("\n✓ Reload verification passed.")

    # Test inverse lookup
    print("\n" + "=" * 70)
    print("Inverse Lookup")
    print("=" * 70)

    print(
        "Channel 0 ->",
        loaded_encoder.inverse_lookup(0, "channel"),
    )

    print(
        "Campaign Type 0 ->",
        loaded_encoder.inverse_lookup(
            0,
            "campaign_type",
        ),
    )

    print(
        "Campaign 0 ->",
        loaded_encoder.inverse_lookup(
            0,
            "campaign",
        ),
    )

    print("\n✓ All tests passed.")

    print("\nChannel Mapping")
    print(encoder.channel_encoder.mapping)

    print("\nCampaign Type Mapping")
    print(encoder.campaign_type_encoder.mapping)

    print("\nCampaign Mapping (first 10)")
    for i, (k, v) in enumerate(encoder.campaign_encoder.mapping.items()):
        print(f"{k} -> {v}")
        if i == 9:
            break
    for category, idx in encoder.channel_encoder.mapping.items():
        assert encoder.inverse_lookup(idx, "channel") == category

    for category, idx in encoder.campaign_type_encoder.mapping.items():
        assert encoder.inverse_lookup(idx, "campaign_type") == category

    print("✓ Inverse lookup verified.")

if __name__ == "__main__":
    main()