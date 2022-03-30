import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from scripts.utils import train_keys
from scripts.Models import ITGDataset
from tqdm.auto import tqdm 


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




# Regressor tools
def regressor_uncertainty(dataloader, regressor, keep = 0.1):
    """
    Calculates the uncertainty of the regressor on the points in the dataloader.
    Returns the most uncertain points.

    """

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data 100 times and return points with largest uncertainty
    runs = []
    for i in tqdm(range(100)):
        step_list = []
        for step, (x, y, z) in enumerate(dataloader):

            predictions = regressor(x.float()).detach().numpy()
            step_list.append(predictions)

        flattened_predictions = np.array(step_list).flatten()
        runs.append(flattened_predictions)

    out_std = np.std(np.array(runs), axis=0)

    top_indices = np.argsort(out_std)[-int(len(out_std) * keep):]

    uncertain_dataset = ITGDataset(
        dataloader.dataset.X[top_indices],
        dataloader.dataset.y[top_indices], 
        dataloader.dataset.z[top_indices].reshape(-1, 1)
        )

    uncertain_dataloader = DataLoader(uncertain_dataset, shuffle=True)

    return uncertain_dataloader

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
