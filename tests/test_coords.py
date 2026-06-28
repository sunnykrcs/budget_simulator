"""
Tests for src.model.coords
"""

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.coords import (
    ModelDims,
    build_coords,
    build_dimension_dict,
    get_dimension_sizes,
    get_index_arrays,
    validate_hierarchy,
)


def create_test_dataframe():
    """Create a small encoded dataframe."""

    return pd.DataFrame(
        {
            "channel_idx": [0, 0, 1, 1],
            "campaign_type_idx": [0, 1, 0, 1],
            "campaign_idx": [0, 1, 2, 3],
        }
    )


def test_model_dims():
    """Test ModelDims."""

    df = create_test_dataframe()

    dims = ModelDims.from_dataframe(df)

    print("\nModel Dimensions")
    print("----------------")
    print(dims)

    assert dims.n_observations == 4
    assert dims.n_channels == 2
    assert dims.n_campaign_types == 2
    assert dims.n_campaigns == 4


def test_build_coords():
    """Test coordinate generation."""

    df = create_test_dataframe()

    coords = build_coords(df)

    print("\nCoordinates")
    print("-----------")
    print(coords)

    assert len(coords["observation"]) == 4
    assert len(coords["channel"]) == 2
    assert len(coords["campaign_type"]) == 2
    assert len(coords["campaign"]) == 4


def test_dimension_sizes():
    """Test dimension summary."""

    df = create_test_dataframe()

    sizes = get_dimension_sizes(df)

    print("\nDimension Sizes")
    print("----------------")
    print(sizes)

    assert sizes["observation"] == 4
    assert sizes["channel"] == 2
    assert sizes["campaign_type"] == 2
    assert sizes["campaign"] == 4


def test_dimension_dict():
    """Test dimension mapping."""

    mapping = build_dimension_dict()

    print("\nDimension Mapping")
    print("-----------------")
    print(mapping)

    assert mapping["channel_idx"] == "channel"
    assert mapping["campaign_type_idx"] == "campaign_type"
    assert mapping["campaign_idx"] == "campaign"


def test_index_arrays():
    """Test extraction of index arrays."""

    df = create_test_dataframe()

    arrays = get_index_arrays(df)

    print("\nIndex Arrays")
    print("------------")
    print(arrays)

    assert arrays["channel_idx"].tolist() == [0, 0, 1, 1]
    assert arrays["campaign_type_idx"].tolist() == [0, 1, 0, 1]
    assert arrays["campaign_idx"].tolist() == [0, 1, 2, 3]


def test_validation():
    """Validation should pass."""

    df = create_test_dataframe()

    validate_hierarchy(df)

    print("\nHierarchy validation passed.")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing coords.py")
    print("=" * 70)

    test_model_dims()
    test_build_coords()
    test_dimension_sizes()
    test_dimension_dict()
    test_index_arrays()
    test_validation()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)