from dataclasses import dataclass
import pytest
from scripts.Models import ITGDatasetDF
from scripts.pipeline_tools import prepare_data
import pandas as pd


@pytest.fixture
def test_dataset():
    test_df = pd.DataFrame(
        data=[
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6],
        ],
        index=[6, 5, 4, 3, 2, 1],
        columns=["a", "b", "c", "d", "e"],
    )
    return ITGDatasetDF(test_df, "d", "e")


def test_index_assigment(test_dataset):

    df_index = list(test_dataset.data.index)
    df_cols = test_dataset.data["index"].to_list()
    assert df_index == df_cols


def test_sample(test_dataset):
    sample = test_dataset.sample(6)

    df_index = list(test_dataset.data.index)
    df_cols = test_dataset.data["index"].to_list()

    sample_index = list(sample.data.index)
    sample_col = sample.data["index"].to_list()

    # Make sure the sampled data frame has the same indices
    assert set(df_index) == set(sample_index)

    # Make sure the sampled data frame inices match the index column
    assert sample_col == sample_index


def test_remove_one(test_dataset):
    test_dataset.remove([1, 2, 3])

    remainig_idx = list(test_dataset.data.index)

    assert remainig_idx == [6, 5, 4]


def test_remove_two(test_dataset):
    drop_indices = list(test_dataset.data.iloc[[0, 1, 2]].index)

    test_dataset.remove(drop_indices)

    assert list(test_dataset.data.index) == [3, 2, 1]


@pytest.fixture
def get_train_val():
    TRAIN_PATH = "/Users/thandikiremadula/Desktop/UKAEA_data/train_data_clipped.pkl"

    VALID_PATH = "/Users/thandikiremadula/Desktop/UKAEA_data/valid_data_clipped.pkl"

    train_data, val_data = prepare_data(
        TRAIN_PATH, VALID_PATH, target_column="efiitg_gb", target_var="itg"
    )

    return train_data, val_data


def test_index_preservation(get_train_val):
    train_data, valid_data = get_train_val

    train_data = ITGDatasetDF(train_data, target_column="efiitg_gb", target_var="itg")
    valid_data = ITGDatasetDF(valid_data, target_column="efiitg_gb", target_var="itg")

    train_idx = train_data.data.index
    valid_idx = valid_data.data.index

    total = len(train_data) + len(valid_data)

    unique = set((list(train_idx) + list(valid_idx)))

    assert len(unique) == total
