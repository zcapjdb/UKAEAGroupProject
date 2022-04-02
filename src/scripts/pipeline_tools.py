from itertools import count
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from scripts.utils import train_keys
from scripts.Models import ITGDatasetDF, ITGDataset
from tqdm.auto import tqdm
import copy


# Data preparation functions
def prepare_data(train_path, valid_path, target_column, target_var):
    train_data = pd.read_pickle(train_path)
    validation_data = pd.read_pickle(valid_path)

    # Remove NaN's and add appropripate class labels
    keep_keys = train_keys + [target_column]

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]

    nt, nv = train_data.shape[0], validation_data.shape[0]
    nt_nan, nv_nan = (
        train_data[target_column].isna().sum(),
        validation_data[target_column].isna().sum(),
    )

    train_data = train_data.dropna()
    validation_data = validation_data.dropna()

    # Make sure the right number of NaN's have been dropped
    assert train_data.shape[0] + nt_nan == nt
    assert validation_data.shape[0] + nv_nan == nv

    train_data[target_var] = np.where(train_data[target_column] != 0, 1, 0)
    validation_data[target_var] = np.where(validation_data[target_column] != 0, 1, 0)

    # Make sure that the class label creation has been done correctly
    assert len(train_data[target_var].unique()) == 2
    assert len(validation_data[target_var].unique()) == 2

    return train_data, validation_data


# classifier tools
def select_unstable_data(dataset, batch_size, classifier):
    # create a data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # get initial size of the dataset
    init_size = len(dataloader.dataset)

    stable_points = []
    misclassified = []
    for i, (x, y, z, idx) in enumerate(tqdm(dataloader)):

        y_hat = classifier(x.float())

        # TODO: Verify which cutoff to use for the classifer
        pred_class = torch.round(y_hat.squeeze().detach()).numpy()
        pred_class = pred_class.astype(int)

        stable = idx[np.where(pred_class == 0)[0]]
        missed = idx[np.where(pred_class != y.numpy())[0]]

        stable_points.append(stable.detach().numpy())
        misclassified.append(missed.detach().numpy())

    # turn list of stable and misclassified points into flat arrays
    stable_points = np.concatenate(np.asarray(stable_points, dtype=object), axis=0)
    misclassified = np.concatenate(np.asarray(misclassified, dtype=object), axis=0)
    print(f"\nStable points: {len(stable_points)}")
    print(f"Misclassified points: {len(misclassified)}")

    # merge the two arrays
    drop_points = np.unique(np.concatenate((stable_points, misclassified), axis=0))
    dataset.remove(drop_points)

    # TODO: make a subset of the data with the misclassified points to retrain the classifier

    print(f"Percentage of misclassified points:  {100*len(misclassified) / init_size}%")
    print(f"\nDropped {init_size - len(dataset.data)} rows")


# Regressor tools
def retrain_regressor(
    train_loader,
    new_loader,
    val_loader,
    model,
    learning_rate,
    epochs=10,
    validation_step=True,
):
    print("\nRetraining regressor...")

    if validation_step:
        test_loss = model.validation_step(val_loader)
        print(f"Initial loss: {test_loss.item():.4f}")

    model.train()

    # instantiate optimiser
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("Train Step: ", epoch)
        loss = model.train_step(new_loader, opt)
        print(f"Loss: {loss.item():.4f}")

        if epoch % 2 == 0:
            print("Using Original Training Data:")
            loss = model.train_step(train_loader, opt)
            print(f"Loss: {loss.item():.4f}")

        if validation_step:
            print("Validation Step: ", epoch)
            test_loss = model.validation_step(val_loader)
            print(f"Test loss: {test_loss.item():.4f}")


def regressor_uncertainty(dataset, regressor, keep=0.1, n_runs=1):
    """
    Calculates the uncertainty of the regressor on the points in the dataloader.
    Returns the most uncertain points.

    """
    print("\nRunning MC Dropout....\n")

    dataloader = DataLoader(dataset, shuffle=False)
    data_copy = copy.deepcopy(dataset)

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data 100 times and return points with largest uncertainty
    runs = []
    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):

            predictions = regressor(x.float()).detach().numpy()
            step_list.append(predictions)

        flattened_predictions = np.array(step_list).flatten()
        runs.append(flattened_predictions)

    out_std = np.std(np.array(runs), axis=0)
    n_out_std = out_std.shape[0]

    drop_indices = np.argsort(out_std)[: n_out_std - int(n_out_std * keep)]
    # TODO: Check if this line does what I expect
    idx_drop = data_copy.data["index"].iloc[drop_indices]

    data_copy.remove(idx_drop)

    uncertain_dataloader = DataLoader(data_copy, shuffle=True)

    # # Using numpy is much faster, why!?
    # x_array = dataset.data[train_keys].values
    # y_array = dataset.data["efiitg_gb"].values
    # idx_array = dataset.data["index"].values

    # dataset2 = ITGDataset(x_array, y_array)
    # dataset2.indices = idx_array
    # dataloader_numpy =  DataLoader(dataset2, shuffle=False)

    # for i in tqdm(range(n_runs)):
    #     for step, (x, y) in enumerate(dataloader_numpy):

    #         predictions = regressor(x.float()).detach().numpy()

    return uncertain_dataloader, out_std[-int(n_out_std * keep) :]


# Active Learning diagonistic functions


def classifier_accuracy(dataset, target_var):

    n_total = len(dataset)
    counts = dataset.data.groupby(target_var).count()

    accuracy = counts.loc[0][0] * 100 / n_total

    print(f"\nCorrectly Classified {accuracy:.3f} %")


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
