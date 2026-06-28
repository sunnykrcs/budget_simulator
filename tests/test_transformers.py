import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transformers import (
    IdentityTransformer,
    LogTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    TransformationPipeline,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv("data/master_dataset.csv")

    columns = ["spend", "revenue"]

    print_header("Original Data")

    print(df[columns].head())

    # --------------------------------------------------------
    # Identity Transformer
    # --------------------------------------------------------

    print_header("Identity Transformer")

    identity = IdentityTransformer(columns=columns)

    df_identity = identity.fit_transform(df)

    assert df_identity[columns].equals(df[columns])

    print("✓ Identity Transformer Passed")

    # --------------------------------------------------------
    # Log Transformer
    # --------------------------------------------------------

    print_header("Log Transformer")

    log = LogTransformer(columns=columns)

    df_log = log.fit_transform(df)

    print(df_log[columns].head())

    df_original = log.inverse_transform(df_log)

    np.testing.assert_allclose(
        df_original[columns],
        df[columns],
        rtol=1e-8,
        atol=1e-12,
    )

    print("✓ Log Transformer Passed")

    # --------------------------------------------------------
    # Standard Scaler
    # --------------------------------------------------------

    print_header("Standard Scaler")

    scaler = StandardScaler(columns=columns)

    df_scaled = scaler.fit_transform(df)

    print(df_scaled[columns].describe())

    df_inverse = scaler.inverse_transform(df_scaled)

    np.testing.assert_allclose(
        df_inverse[columns],
        df[columns],
        rtol=1e-8,
        atol=1e-12,
    )

    print("✓ Standard Scaler Passed")

    # --------------------------------------------------------
    # Min-Max Scaler
    # --------------------------------------------------------

    print_header("MinMax Scaler")

    minmax = MinMaxScaler(columns=columns)

    df_scaled = minmax.fit_transform(df)

    print(df_scaled[columns].describe())

    assert df_scaled[columns].min().min() >= 0

    assert df_scaled[columns].max().max() <= 1

    df_inverse = minmax.inverse_transform(df_scaled)

    np.testing.assert_allclose(
        df_inverse[columns],
        df[columns],
        rtol=1e-8,
        atol=1e-12,
    )

    print("✓ MinMax Scaler Passed")

    # --------------------------------------------------------
    # Robust Scaler
    # --------------------------------------------------------

    print_header("Robust Scaler")

    robust = RobustScaler(columns=columns)

    df_scaled = robust.fit_transform(df)

    print(df_scaled[columns].describe())

    df_inverse = robust.inverse_transform(df_scaled)

    np.testing.assert_allclose(
        df_inverse[columns],
        df[columns],
        rtol=1e-8,
        atol=1e-12,
    )

    print("✓ Robust Scaler Passed")

    # --------------------------------------------------------
    # Transformation Pipeline
    # --------------------------------------------------------

    print_header("Transformation Pipeline")

    pipeline = TransformationPipeline(
        transformers=[
            LogTransformer(columns=columns),
            StandardScaler(columns=columns),
        ]
    )

    df_pipeline = pipeline.fit_transform(df)

    print(df_pipeline[columns].head())

    df_inverse = pipeline.inverse_transform(df_pipeline)

    np.testing.assert_allclose(
        df_inverse[columns],
        df[columns],
        rtol=1e-8,
        atol=1e-12,
    )

    print("✓ Transformation Pipeline Passed")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header("ALL TESTS PASSED")

    print("✓ IdentityTransformer")

    print("✓ LogTransformer")

    print("✓ StandardScaler")

    print("✓ MinMaxScaler")

    print("✓ RobustScaler")

    print("✓ TransformationPipeline")


if __name__ == "__main__":
    main()