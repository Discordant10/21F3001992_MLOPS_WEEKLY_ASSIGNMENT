import os
import pandas as pd
import pytest

DATA_PATH = "data/iris_data_adapted_for_feast.csv"

EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

VALID_SPECIES = [
    "setosa",
    "versicolor",
    "virginica"
]


@pytest.fixture(scope="module")
def iris_dataframe():
    """Load the Iris dataset once for all tests."""

    assert os.path.exists(DATA_PATH), (
        f"Dataset not found at {DATA_PATH}. "
        "Did you forget to run 'dvc pull'?"
    )

    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    """Dataset file should exist."""

    assert os.path.exists(DATA_PATH)


def test_dataset_not_empty(iris_dataframe):
    """Dataset should not be empty."""

    assert len(iris_dataframe) > 0


def test_expected_columns(iris_dataframe):
    """Dataset schema should match expected schema."""

    assert list(iris_dataframe.columns) == EXPECTED_COLUMNS


def test_no_missing_values(iris_dataframe):
    """Dataset should not contain missing values."""

    assert iris_dataframe.isnull().sum().sum() == 0


def test_numeric_feature_types(iris_dataframe):
    """All feature columns must be numeric."""

    for column in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(
            iris_dataframe[column]
        ), f"{column} is not numeric"


def test_species_column_exists(iris_dataframe):
    """Species column must exist."""

    assert "species" in iris_dataframe.columns


def test_valid_species_labels(iris_dataframe):
    """Species labels must be valid."""

    labels = set(iris_dataframe["species"].unique())

    assert labels.issubset(set(VALID_SPECIES))


def test_no_duplicate_rows(iris_dataframe):
    """Dataset should not contain duplicate rows."""

    duplicates = iris_dataframe.duplicated().sum()

    assert duplicates == 0


def test_positive_feature_values(iris_dataframe):
    """All measurements should be positive."""

    for column in FEATURE_COLUMNS:
        assert (iris_dataframe[column] > 0).all()


def test_reasonable_feature_ranges(iris_dataframe):
    """
    Feature values should fall within reasonable
    biological ranges for the Iris dataset.
    """

    assert iris_dataframe["sepal_length"].between(4, 8.5).all()

    assert iris_dataframe["sepal_width"].between(2, 5).all()

    assert iris_dataframe["petal_length"].between(1, 7.5).all()

    assert iris_dataframe["petal_width"].between(0.1, 3).all()


def test_dataset_size(iris_dataframe):
    """
    Iris dataset should contain exactly 150 samples.
    """

    assert len(iris_dataframe) == 150


def test_species_distribution(iris_dataframe):
    """
    Each class should contain 50 samples.
    """

    counts = iris_dataframe["species"].value_counts()

    assert counts["setosa"] == 50
    assert counts["versicolor"] == 50
    assert counts["virginica"] == 50