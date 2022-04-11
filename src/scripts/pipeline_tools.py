from itertools import count
from turtle import color
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
import logging
from scripts.utils import train_keys
from scripts.Models import ITGDatasetDF, ITGDataset

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Data preparation functions
def prepare_data(
    train_path, valid_path, target_column, target_var, train_size=None, valid_size=None
):
    train_data = pd.read_pickle(train_path)
    validation_data = pd.read_pickle(valid_path)

    if train_size is not None:
        train_data = train_data.sample(train_size)

    if valid_size is not None:
        validation_data = validation_data.sample(valid_size)

    # Remove NaN's and add appropripate class labels
    keep_keys = train_keys + [target_column]

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]

    train_data = train_data.dropna()
    validation_data = validation_data.dropna()

    train_data[target_var] = np.where(train_data[target_column] != 0, 1, 0)
    validation_data[target_var] = np.where(validation_data[target_column] != 0, 1, 0)

    return train_data, validation_data


# classifier tools
def select_unstable_data(dataset, batch_size, classifier):
    # create a data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # get initial size of the dataset
    init_size = len(dataloader.dataset)

    stable_points = []
    misclassified = []

    logging.info("Running classifier selection...")

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

    logging.log(15, f"Stable points: {len(stable_points)}")
    logging.log(15, f"Misclassified points: {len(misclassified)}")

    # create new dataset with misclassified points
    misclassified_dataset = copy.deepcopy(dataset)
    misclassified_dataset.data = misclassified_dataset.data.loc[misclassified]

    # merge the two arrays of stable and misclassified points
    drop_points = np.unique(np.concatenate((stable_points, misclassified), axis=0))
    dataset.remove(drop_points)

    logging.info(
        f"Percentage of misclassified points:  {100*len(misclassified) / init_size}%"
    )
    logging.log(15, f"Dropped {init_size - len(dataset.data)} rows")

    return dataset, misclassified_dataset


# Function to retrain the classifier on the misclassified points
def retrain_classifier(
    misclassified_dataset,
    training_dataset,
    valid_dataset,
    classifier,
    learning_rate=5e-4,
    epochs=10,
    batch_size=128,
    validation_step=True,
    lam=1,
    loc=0.0,
    scale=0.01,
    patience=None,
):
    logging.info("Retraining classifier...\n")
    logging.log(15, f"Training on {len(misclassified_dataset)} points")

    train = copy.deepcopy(training_dataset)
    train.add(misclassified_dataset)

    # create data loaders
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    missed_loader = DataLoader(
        misclassified_dataset, batch_size=batch_size, shuffle=True
    )

    valid_loader = pandas_to_numpy_data(valid_dataset)

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    classifier.shrink_perturb(lam, loc, scale)

    if not patience:
        patience = epochs

    # instantiate optimiser
    opt = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=0.5*patience
    )
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []


    for epoch in range(epochs):

        logging.debug(f"Train Step:  {epoch}")

        loss, acc = classifier.train_step(train_loader, opt)
        train_loss.append(loss.item())
        train_acc.append(acc)

        if (validation_step and epoch % 5 == 0) or epoch == epochs - 1:

            logging.debug(f"Validation Step: {epoch}")

            validation_loss, validation_accuracy = classifier.validation_step(
                valid_loader
            )
            val_loss.append(validation_loss)
            val_acc.append(validation_accuracy)

            logging.debug(f"Evaluating on just the misclassified points")
            missed_loss, missed_acc = classifier.validation_step(missed_loader)

        if len(val_loss) > patience:
            if np.mean(val_acc[-patience:]) > val_acc[-1]:
                logging.debug("Early stopping criterion reached")
                break

    return train_loss, train_acc, val_loss, val_acc


# Regressor tools
def retrain_regressor(
    new_loader,
    val_loader,
    model,
    learning_rate=1e-4,
    epochs=5,
    validation_step=True,
    lam=1,
    loc=0.0,
    scale=0.01,
    patience=None,
):
    logging.info("Retraining regressor...\n")
    logging.log(15, f"Training on {len(new_loader.dataset)} points")

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    model.shrink_perturb(lam, loc, scale)

    if validation_step:
        test_loss = model.validation_step(val_loader)
        logging.log(15, f"Initial validation loss: {test_loss.item():.4f}")

    if not patience:
        patience = epochs

    model.train()

    # instantiate optimiser
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=0.5*patience
    )
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        logging.log(15, f"Epoch: {epoch}")
        loss = model.train_step(new_loader, opt)
        train_loss.append(loss.item())

        logging.log(15, f"Training Loss: {loss.item():.4f}")

        if (validation_step and epoch % 10 == 0) or epoch == epochs - 1:
            test_loss = model.validation_step(val_loader).item()
            val_loss.append(test_loss)

            logging.log(15, f"Test loss: {test_loss:.4f}")

        if len(val_loss) > patience:
            if np.mean(val_loss[-patience:]) < test_loss:
                logging.log(15, "Early stopping criterion reached")
                break

    return train_loss, val_loss


def regressor_uncertainty(
    dataset,
    regressor,
    keep=0.25,
    n_runs=10,
    train_data=False,
    plot=False,
    order_idx=None,
    valid_dataset=None,
):
    """
    Calculates the uncertainty of the regressor on the points in the dataloader.
    Returns the most uncertain points.

    """
    if train_data:
        logging.info("Running MC Dropout on Training Data....\n")
    else:
        logging.info("Running MC Dropout on Novel Data....\n")

    data_copy = copy.deepcopy(dataset)
    if train_data:
        batch_size = 2 ** (int(np.log(len(dataset)))) // 4
    else:
        batch_size = len(dataset)

    dataloader = DataLoader(data_copy, batch_size=batch_size, shuffle=False)

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data n_runs times and return points with largest uncertainty
    runs = []

    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):

            predictions = regressor(x.float()).detach().numpy()
            step_list.append(predictions)

        flat_list = [item for sublist in step_list for item in sublist]
        flattened_predictions = np.array(flat_list).flatten()
        runs.append(flattened_predictions)

    out_std = np.std(np.array(runs), axis=0)
    n_out_std = out_std.shape[0]

    logging.log(15, f"Number of points passed for MC dropout: {n_out_std}")

    idx_list = []
    for step, (x, y, z, idx) in enumerate(dataloader):
        idx_list.append(idx.detach().numpy())

    flat_list = [item for sublist in idx_list for item in sublist]
    idx_array = np.asarray(flat_list, dtype=object).flatten()

    if not train_data:
        top_idx = np.argsort(out_std)[-int(n_out_std * keep) :]
        drop_idx = np.argsort(out_std)[: n_out_std - int(n_out_std * keep)]
        drop_idx = idx_array[drop_idx]
        real_idx = idx_array[top_idx]

    if order_idx is None and train_data == False:
        # Add 100 - x% back into the validation data set
        logging.log(15, f"no valid before : {len(valid_dataset)}")

        temp_dataset = copy.deepcopy(dataset)
        temp_dataset.remove(indices=real_idx)
        valid_dataset.add(temp_dataset)

        logging.log(15, f"no valid after : {len(valid_dataset)}")

        del temp_dataset

        # Remove them from the sample
        data_copy.remove(drop_idx)

    if plot:
        if order_idx is not None:
            tag = "Final"
        else:
            tag = "Initial"

        plot_uncertainties(out_std, keep, tag)

    top_indices = np.argsort(out_std)[-int(len(out_std) * keep) :]

    if order_idx is not None:
        # matching the real indices to the array position
        reorder = np.array([np.where(idx_array == i) for i in order_idx]).flatten()

        real_idx = idx_array[reorder]

        # selecting the corresposing std ordered according to order_idx
        top_indices = reorder

        # Make sure the real indices match
        assert list(np.unique(real_idx)) == list(np.unique(order_idx))

        # Make sure they are in the same order
        assert real_idx.tolist() == order_idx.tolist(), logging.error("Ordering error")

    if not train_data:
        return data_copy, out_std[top_indices], real_idx

    else:
        return data_copy, out_std, idx_array

def pandas_to_numpy_data(dataset, batch_size=None):

    x_array = dataset.data[train_keys].values
    y_array = dataset.data[dataset.label].values
    z_array = dataset.data[dataset.target].values
    
    numpy_dataset = ITGDataset(x_array, y_array, z_array)

    if batch_size is None:
        batch_size = int(0.1 * len(y_array))

    numpy_loader = DataLoader(
        numpy_dataset, batch_size=batch_size, shuffle=True
    )
    return numpy_loader

# Active Learning diagonistic functions
def get_mse(y_hat, y):
    mse = np.mean((y - y_hat) ** 2)
    return mse


def mse_change(
    prediction_before,
    prediction_after,
    prediction_order,
    uncert_data_order,
    uncertain_loader,
    uncertainties=None,
    plot=False,
    data="novel",
    save_plots=False,
):
    idxs = prediction_order.astype(int)
    ground_truth = uncertain_loader.dataset.data.loc[idxs]
    # TODO: Change hard coded variable
    ground_truth = ground_truth["efiitg_gb"]
    ground_truth = ground_truth.to_numpy()

    idx = np.isin(prediction_order, uncert_data_order)

    pred_before = prediction_before[idx]
    pred_after = prediction_after[idx]
    ground_truth_subset = ground_truth[idx]

    mse_before = get_mse(pred_before, ground_truth_subset)
    mse_after = get_mse(pred_after, ground_truth_subset)

    logging.info(f"Change in MSE for {data} dataset: {mse_after-mse_before:.4f}\n")

    if plot:
        plot_mse_change(
            ground_truth_subset,
            pred_before,
            pred_after,
            uncertainties,
            data=data,
            save_plots=save_plots,
        )
    return mse_before, mse_after, (mse_after - mse_before)


def uncertainty_change(x, y, plot=True):
    total = x.shape[0]
    increase = len(x[y > x]) * 100 / total
    decrease = len(x[y < x]) * 100 / total
    no_change = 100 - increase - decrease

    if plot:
        plot_scatter(x, y)

    av_uncert_before = np.mean(x)
    av_uncer_after = np.mean(y)

    logging.info(
        f" Decreased {decrease:.3f}% Increased: {increase:.3f} % No Change: {no_change:.3f} "
    )

    logging.info(
        f"Initial Average Uncertainty: {av_uncert_before:.4f}, Final Average Uncertainty: {av_uncer_after:.4f}"
    )
    return av_uncer_after - av_uncert_before


# plotting functions


def plot_uncertainties(out_std: np.ndarray, keep: float, tag=None):
    plt.figure()
    plt.hist(out_std[np.argsort(out_std)[-int(len(out_std) * keep) :]], bins=50)
    # plt.show()
    name_uncertain = "standard_deviation_histogram_most_uncertain"
    if tag is not None:
        name_uncertain = f"{name_uncertain}_{tag}"
    plt.savefig(f"{name_uncertain}.png")

    plt.figure()
    plt.hist(out_std, bins=50)
    # plt.show()
    name = "standard_deviation_histogram"
    if tag is not None:
        name = f"{name}_{tag}"
    plt.savefig(f"{name}.png")


def plot_scatter(initial_std: np.ndarray, final_std: np.ndarray):
    plt.figure()
    plt.scatter(initial_std, final_std, s=3, alpha=1)
    # y = x dotted line to show no change
    plt.plot(
        [initial_std.min(), final_std.max()],
        [initial_std.min(), final_std.max()],
        "k--",
        lw=2,
    )
    plt.xlabel("Initial Standard Deviation")
    plt.ylabel("Final Standard Deviation")
    plt.savefig("scatter_plot.png")


def plot_mse_change(
    ground_truth,
    intial_prediction,
    final_prediction,
    uncertainties,
    data="novel",
    save_plots=False,
):
    if uncertainties is not None:
        uncert_before, uncert_after = uncertainties

    mse_before = get_mse(intial_prediction, ground_truth)
    mse_after = get_mse(final_prediction, ground_truth)
    delta_mse = mse_after - mse_before

    x_min = ground_truth.min()
    x_max = ground_truth.max()

    if data == "novel":
        title = "Novel Data"
        save_prefix = "novel"
    elif data == "train":
        title = "Training Data"
        save_prefix = "train"

    else:
        title = data
        save_prefix = data

    plt.figure()

    if uncertainties is not None:
        plt.scatter(ground_truth, intial_prediction, c=uncert_before)
        plt.colorbar(label="$\sigma$")
    else:
        plt.scatter(ground_truth, intial_prediction, color="forestgreen")

    plt.plot(
        np.linspace(x_min, x_max, 50),
        np.linspace(x_min, x_max, 50),
        ls="--",
        c="black",
        label=f"MSE: {mse_before:.4f}",
    )

    plt.xlabel("Ground Truth")
    plt.ylabel("Original Prediction")
    plt.title(title)
    plt.legend()
    if save_plots:
        plt.savefig(f"{save_prefix}_mse_before.png", dpi=300)

    plt.figure()
    if uncertainties is not None:
        plt.scatter(ground_truth, final_prediction, c=uncert_after)
        plt.colorbar(label="$\sigma$")
    else:
        plt.scatter(ground_truth, final_prediction, color="forestgreen")

    plt.plot(
        np.linspace(x_min, x_max, 50),
        np.linspace(x_min, x_max, 50),
        ls="--",
        c="black",
        label=f"$\Delta$MSE: {delta_mse:.4f}",
    )

    plt.xlabel("Ground Truth")
    plt.ylabel("Retrained Prediction")
    plt.title(title)
    plt.legend()

    if save_plots:
        plt.savefig(f"{save_prefix}_mse_after.png", dpi=300)
