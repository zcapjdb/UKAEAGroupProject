# from scripts.Models import ITG_Classifier, ITG_Regressor
import pandas as pd
import torch.nn as nn
import numpy as np

from scripts.utils import train_keys


# Data preparation functions
def prepare_data(train_path, valid_path):
    train_data = pd.read_pickle(train_path)
    validation_data = pd.read_pickle(valid_path)

    # Remove NaN's and add appropripate class labels
    keep_keys = train_keys + ["efiitg_gb"]

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]

    nt, nv = train_data.shape[0], validation_data.shape[0]
    nt_nan, nv_nan = (
        train_data["efiitg_gb"].isna().sum(),
        validation_data["efiitg_gb"].isna().sum(),
    )

    train_data = train_data.dropna()
    validation_data = validation_data.dropna()

    # Make sure the right number of NaN's have been dropped
    assert train_data.shape[0] + nt_nan == nt
    assert validation_data.shape[0] + nv_nan == nv

    train_data["itg"] = np.where(train_data["efiitg_gb"] != 0, 1, 0)
    validation_data["itg"] = np.where(validation_data["efiitg_gb"] != 0, 1, 0)

    # Make sure that the class label creation has been done correctly
    assert len(train_data["itg"].unique()) == 2
    assert len(validation_data["itg"].unique()) == 2

    return train_data, validation_data


# Active Learning diagonistic functions

# Lower or Higher uncertainty post training
def uncertainty_change(x, y):
    theta = np.arctan(y, x)
    theta = np.rad2deg(theta)

    total = theta.shape[0]

    increase = len(theta[theta < 45]) * 100 / total
    decrease = len(theta[theta > 45]) * 100 / total
    no_chnage = 100 - increase - decrease

    print(
        f" Decreased {decrease:.3f}% Increased: {increase:.3f} % No Change: {no_chnage:.3f} "
    )
