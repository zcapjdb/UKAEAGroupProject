import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import copy
import logging
from scripts.utils import train_keys, target_keys
from pipeline.Models import ITGDatasetDF, ITGDataset, Classifier, Regressor
from typing import Union
import os

logging.getLogger("matplotlib").setLevel(logging.WARNING)

output_dict = {
    "train_loss_init": [], # Regressor performance before pipeline
    "test_loss_init": [],

    "retrain_losses": [], # regressor performance during retraining
    "retrain_test_losses": [],
    "post_test_loss": [], # regressor performance after retraining
    "post_test_loss_unscaled": [], # regressor performance after retraining, unscaled

    "n_train_points": [],
    "mse_before": [],
    "mse_after": [],
    "d_mse": [],
    "d_uncert": [],
    "d_novel_uncert": [],
    "novel_uncert_before": [],
    "novel_uncert_after": [],
    "holdout_pred_before": [],
    "holdout_pred_after": [],
    "holdout_ground_truth": [],
    "class_train_loss": [],
    "class_val_loss": [],
    "class_missed_loss": [],
    "class_train_acc": [],
    "class_val_acc": [],
    "class_missed_acc": [],
}

# Data preparation functions
def prepare_data(
    train_path: str,
    valid_path: str,
    test_path: str,
    target_column: str,
    train_size: int = None,
    valid_size: int = None,
    test_size: int = None,
    samplesize_debug: int = 1,
) -> (ITGDatasetDF, ITGDatasetDF, ITGDatasetDF, StandardScaler):
    """
    Loads the data from the given paths and prepares it for training.
    train_path, valid_path point to pickle files containing the data in dataframes.
    """

    train_data = pd.read_pickle(train_path)
    validation_data = pd.read_pickle(valid_path)
    test_data = pd.read_pickle(test_path)

    if train_size is not None:
        train_data = train_data.sample(samplesize_debug*train_size)

    if valid_size is not None:
        validation_data = validation_data.sample(samplesize_debug*valid_size)
    
    if test_size is not None:
        test_data = test_data.sample(samplesize_debug*test_size)


    if target_column not in target_keys:
        raise ValueError("Flux variable to supported")

    # Remove NaN's and add appropripate class labels
    keep_keys = train_keys + [target_column]

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]
    test_data = test_data[keep_keys]

    train_data = train_data.dropna()
    validation_data = validation_data.dropna()
    test_data = test_data.dropna()

    train_data["stable_label"] = np.where(train_data[target_column] != 0, 1, 0)
    validation_data["stable_label"] = np.where(
        validation_data[target_column] != 0, 1, 0
    )
    test_data["stable_label"] = np.where(
        test_data[target_column] != 0, 1, 0
    )

    scaler = StandardScaler()
    scaler.fit_transform(train_data.drop(["stable_label"], axis=1))

    train_dataset = ITGDatasetDF(train_data, target_column=target_column)
    valid_dataset = ITGDatasetDF(validation_data, target_column=target_column)
    test_dataset = ITGDatasetDF(test_data, target_column=target_column)

    train_dataset.scale(scaler)
    valid_dataset.scale(scaler)
    test_dataset.scale(scaler)

    return train_dataset, valid_dataset, test_dataset, scaler


# classifier tools
def select_unstable_data(
    dataset: ITGDatasetDF, batch_size: int, classifier: Classifier, device: torch.device = None,
) -> ITGDatasetDF:
    """
    Selects data classified as unstable by the classifier. 

    returns:
        dataset: the dataset with the unstable data removed.
        
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    init_size = len(dataloader.dataset)

    unstable_points = []

    logging.info("Running classifier selection...")

    for i, (x, y, z, idx) in enumerate(tqdm(dataloader)):
        x = x.to(device)
        y_hat = classifier(x.float())

        pred_class = torch.round(y_hat.squeeze().detach()).numpy()
        pred_class = pred_class.astype(int)

        unstable = idx[np.where(pred_class == 1)[0]]
        unstable_points.append(unstable.detach().numpy())

    # turn list of stable and misclassified points into flat arrays
    unstable_points = np.concatenate(np.asarray(unstable_points, dtype=object), axis=0)

    logging.log(15, f"Unstable points: {len(unstable_points)}")

    # create new dataset with misclassified points
    unstable_candidates = copy.deepcopy(dataset) 
    unstable_candidates.data = unstable_candidates.data.loc[unstable_points] 

    return unstable_candidates


def check_for_misclassified_data(candidates: ITGDatasetDF) -> ITGDatasetDF:
    candidate_loader = DataLoader(candidates, batch_size=1, shuffle=False)

    missed_points = []
    for (x, y, z, idx) in candidate_loader:
        # if y == 0, then it is misclassified, keep only the misclassified points
        if y.item() == 0:
            missed_points.append(idx.item())
    
    # create new dataset with misclassified points
    missed_candidates = copy.deepcopy(candidates)
    missed_candidates.data = missed_candidates.data.loc[missed_points]

    return missed_candidates.data , len(missed_points)

# Function to retrain the classifier on the misclassified points
def retrain_classifier(
    misclassified_dataset: ITGDatasetDF,
    training_dataset: ITGDatasetDF,
    valid_dataset: ITGDataset,
    classifier: Classifier,
    learning_rate: int = 5e-4,
    epochs: int = 10,
    batch_size: int = 1024,
    validation_step: bool = True,
    lam: Union[float, int] = 1,
    loc: float = 0.0,
    scale: float = 0.01,
    patience: Union[None, int] = None,
    disable_tqdm: bool = True,
) -> (list, list, list, list, list, list):
    """
    Retrain the classifier on the misclassified points.
    Data for retraining is taken from the combined training and misclassified datasets.
    Returns the losses and accuracies of the training and validation steps.
    """

    logging.info("Retraining classifier...\n")
    data_size = len(misclassified_dataset) + len(training_dataset)
    logging.log(15, f"Training on {data_size} points")

    train = copy.deepcopy(training_dataset)
    train.add(misclassified_dataset)

    # create data loaders
    train_loader = pandas_to_numpy_data(train, batch_size=batch_size, shuffle=True)
    missed_loader = pandas_to_numpy_data(misclassified_dataset, shuffle=True)
    valid_loader = pandas_to_numpy_data(valid_dataset)

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    classifier.shrink_perturb(lam, loc, scale)

    if not patience:
        patience = epochs

    # instantiate optimiser
    opt = torch.optim.Adam(classifier.parameters(), lr=learning_rate,weight_decay=1.e-4)
    # create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=0.5 * patience,
        min_lr=(1 / 16) * learning_rate,
    )
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    missed_loss = []
    missed_acc = []

    for epoch in range(epochs):

        logging.debug(f"Train Step:  {epoch}")

        loss, acc = classifier.train_step(train_loader, opt, epoch=epoch, disable_tqdm=disable_tqdm)
        train_loss.append(loss.item())
        train_acc.append(acc)

        if validation_step:

            logging.debug(f"Validation Step: {epoch}")

            validation_loss, validation_accuracy = classifier.validation_step(
                valid_loader, scheduler
            )
            val_loss.append(validation_loss)
            val_acc.append(validation_accuracy)

            logging.debug(f"Evaluating on just the misclassified points")
            miss_loss, miss_acc = classifier.validation_step(missed_loader)
            missed_loss.append(miss_loss)
            missed_acc.append(miss_acc)

        if len(val_loss) > patience:
            if np.mean(val_acc[-patience:]) > val_acc[-1]:
                logging.debug("Early stopping criterion reached")
                break

    return [train_loss, val_loss, missed_loss], [train_acc, val_acc, missed_acc]


# Regressor tools
def reoder_arrays(array, order1, order2):
    '''
    Inputs: 
        array: The array to be reordered
        order1: the desired index ordering
        order2: the current index ordering
    '''
    reorder = np.array([np.where(order2 == i) for i in order1]).flatten()
    return array[reorder]

def retrain_regressor(
    new_dataset: ITGDatasetDF,
    val_dataset: ITGDatasetDF,
    model: Regressor,
    learning_rate: int = 1e-4,
    epochs: int = 10,
    validation_step: bool = True,
    lam: Union[float, int] = 1,
    loc: float = 0.0,
    scale: float = 0.01,
    patience: Union[None, int] = None,
    batch_size: int = 1024,
    disable_tqdm: bool = True,
) -> (list, list):
    """
    Retrain the regressor on the most uncertain points.
    Data for retraining is taken from the combined training and uncertain datasets.
    Returns the losses of the training and validation steps.
    """

    logging.info("Retraining regressor...\n")
    logging.log(15, f"Training on {len(new_dataset)} points")

    new_loader = pandas_to_numpy_data(new_dataset, batch_size=batch_size, shuffle=True)
    val_loader = pandas_to_numpy_data(val_dataset, batch_size=batch_size, shuffle=False)

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    model.shrink_perturb(lam, loc, scale)

    if validation_step:
        test_loss = model.validation_step(val_loader)
        logging.log(15, f"Initial validation loss: {test_loss.item():.4f}")

    if not patience:
        patience = epochs

    model.train()

    # instantiate optimiser
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=0.5 * patience,
        min_lr=(1 / 16) * learning_rate,
    )
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        logging.log(15, f"Epoch: {epoch}")
        loss = model.train_step(new_loader, opt, epoch = epoch, disable_tqdm=disable_tqdm)
        train_loss.append(loss.item())

        logging.log(15, f"Training Loss: {loss.item():.4f}")

        if validation_step:
            test_loss = model.validation_step(val_loader, scheduler).item()
            val_loss.append(test_loss)

            logging.log(15, f"Test loss: {test_loss:.4f}")

        if len(val_loss) > patience:
            if np.mean(val_loss[-patience:]) < test_loss:
                logging.log(15, "Early stopping criterion reached")
                break

    return train_loss, val_loss

def get_uncertainty(
    dataset: ITGDatasetDF,
    regressor: Regressor,
    n_runs: int = 25,
    order_idx: Union[None, list, np.array] = None,
    train_data: bool = False,
    plot: bool = False,
    device : torch.device = None,
) -> (np.array, np.array):

    """
    Calculates the uncertainty of the regressor on the points in the dataloader.
    Returns the most uncertain points.
    If order_idx is provided, the points are ordered according to the order_idx to allow
    comparison before and after retraining.

    """

    data_copy = copy.deepcopy(dataset)

    if train_data:
        logging.info("Running MC Dropout on Training Data....\n")
    else:
        logging.info("Running MC Dropout on Novel Data....\n")

    batch_size = min(len(dataset), 512)
    dataloader = pandas_to_numpy_data(data_copy, batch_size=batch_size, shuffle=False)

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data n_runs times and return points with largest uncertainty
    runs = []

    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):
            x = x.to(device)
            predictions = regressor(x.float()).detach().numpy()
            step_list.append(predictions)

        flat_list = [item for sublist in step_list for item in sublist]
        flattened_predictions = np.array(flat_list).flatten()
        runs.append(flattened_predictions)

    out_std = np.std(np.array(runs), axis=0)
    n_out_std = out_std.shape[0]

    logging.log(15, f"Number of points passed for MC dropout: {n_out_std}")

    # Get the list of indices of the dataframe
    idx_list = []
    for step, (x, y, z, idx) in enumerate(dataloader):
        idx_list.append(idx.detach().numpy())

    # flatten the list of indices
    flat_list = [item for sublist in idx_list for item in sublist]
    idx_array = np.asarray(flat_list, dtype=object).flatten()

    if plot:
        if order_idx == None: 
            tag = "Initial"
        else: 
            tag = "Final"
        keep = 1.0
        plot_uncertainties(out_std, keep, tag)
    
    if order_idx is not None:
        # matching the real indices to the array position
        reorder = np.array([np.where(idx_array == i) for i in order_idx]).flatten()
        uncertain_data_idx = idx_array[reorder]

        # selecting the corresposing std ordered according to order_idx
        uncertain_list_indices = reorder

        # Make sure the real indices match
        assert list(np.unique(uncertain_data_idx)) == list(np.unique(order_idx))

        # Make sure they are in the same order
        assert uncertain_data_idx.tolist() == order_idx.tolist(), logging.error("Ordering error")

        return out_std[reorder], uncertain_data_idx
    else: 
        return out_std, idx_array

def get_most_uncertain(
    dataset: ITGDatasetDF,
    out_std_1: np.array, 
    idx_array_1: np.array, 
    keep: float = 0.25, 
    unlabelled_pool: Union[None, ITGDataset] = None,
    plot: bool= True,
    out_std_2: np.array  = None,
    idx_array_2: np.array = None,
):

    '''
    Inputs:

        dataset: dataset of points that MC drop out was ran on 
        out_std_1: standard deviation from MC dropout from regressor 1 
        idx_array_1: order of the datapoints from data loader 
        keep: percentage of most uncertain points to keep
        plot: Whether to plot the distribution of the uncertainties
        out_std_2: [optional] standard deviation from MC dropout from regressors 2 
        idx_array_2: [optional] order of the datapoints from data loader 

    Outputs:
        data_copy: a dataset object containing only the most uncertain points
        out_std: standard deviation of the most ucnertain points
        idx_array: idx_array used for ordering  (which one have we followed)

    '''
    data_copy = copy.deepcopy(dataset)
    n_out_std = out_std_1.shape[0]
    if out_std_2 != None: 
        assert idx_array_2 != None, "Missing index order of second std entry "

        # sort out the standard deviations so that match
        reorder = np.array([np.where(idx_array_2 == i) for i in idx_array_1]).flatten()
        out_std_2 = out_std_2[reorder]

        # add the uncertainties from the two regressors
        total_std = out_std_1 + out_std_2
        

        uncertain_list_indices = np.argsort(total_std)[-int(n_out_std * keep) :]
        certain_list_indices = np.argsort(total_std)[: n_out_std - int(n_out_std * keep)]

    else: 
        total_std = out_std_1
        uncertain_list_indices = np.argsort(out_std_1)[-int(n_out_std * keep) :]
        certain_list_indices = np.argsort(out_std_1)[: n_out_std - int(n_out_std * keep)]

    certain_data_idx = idx_array_1[certain_list_indices]
    uncertain_data_idx = idx_array_1[uncertain_list_indices]

    # Take the points that are not in the most uncertain points and add back into the validation set
    temp_dataset = copy.deepcopy(dataset)  
    temp_dataset.remove(indices=uncertain_data_idx)
    unlabelled_pool.add(temp_dataset)

    logging.log(15, f"no valid after : {len(unlabelled_pool)}")

    del temp_dataset

    # Remove them from the sample
    data_copy.remove(certain_data_idx)

        
    return data_copy, total_std[uncertain_list_indices], idx_array_1, unlabelled_pool


def regressor_uncertainty(
    dataset: ITGDatasetDF,
    regressor: Regressor,
    keep: float = 0.25,
    n_runs: int = 25,
    train_data: bool = False,
    plot: bool = False,
    order_idx: Union[None, list, np.array] = None,
    unlabelled_pool: Union[None, ITGDataset] = None,
    device : torch.device = None,
) -> (ITGDatasetDF, np.array, np.array):
    """
    Calculates the uncertainty of the regressor on the points in the dataloader.
    Returns the most uncertain points.
    If order_idx is provided, the points are ordered according to the order_idx to allow
    comparison before and after retraining.

    """

    data_copy = copy.deepcopy(dataset)

    if train_data:
        logging.info("Running MC Dropout on Training Data....\n")
    else:
        logging.info("Running MC Dropout on Novel Data....\n")

    batch_size = min(len(dataset), 512)
    dataloader = pandas_to_numpy_data(data_copy, batch_size=batch_size, shuffle=False)

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data n_runs times and return points with largest uncertainty
    runs = []

    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):
            x = x.to(device)
            predictions = regressor(x.float()).detach().numpy()
            step_list.append(predictions)

        flat_list = [item for sublist in step_list for item in sublist]
        flattened_predictions = np.array(flat_list).flatten()
        runs.append(flattened_predictions)

    out_std = np.std(np.array(runs), axis=0)
    n_out_std = out_std.shape[0]

    logging.log(15, f"Number of points passed for MC dropout: {n_out_std}")

    # Get the list of indices of the dataframe
    idx_list = []
    for step, (x, y, z, idx) in enumerate(dataloader):
        idx_list.append(idx.detach().numpy())

    # flatten the list of indices
    flat_list = [item for sublist in idx_list for item in sublist]
    idx_array = np.asarray(flat_list, dtype=object).flatten()

    # If not evaluating on the training data (or validation ?) sort the indices by uncertainty
    # and return the top keep% of the points

    uncertain_list_indices = np.argsort(out_std)[-int(n_out_std * keep) :]

    if not train_data:
        certain_list_indices = np.argsort(out_std)[: n_out_std - int(n_out_std * keep)]
        certain_data_idx = idx_array[certain_list_indices]
        uncertain_data_idx = idx_array[uncertain_list_indices]

    if order_idx is None and train_data == False:
        logging.log(15, f"no valid before : {len(unlabelled_pool)}")

        # Take the points that are not in the most uncertain points and add back into the validation set
        temp_dataset = copy.deepcopy(dataset)  
        temp_dataset.remove(indices=uncertain_data_idx)
        unlabelled_pool.add(temp_dataset)

        logging.log(15, f"no valid after : {len(unlabelled_pool)}")

        del temp_dataset

        # Remove them from the sample
        data_copy.remove(certain_data_idx)

    if plot:
        if order_idx is not None:
            tag = "Final"
        else:
            tag = "Initial"

        plot_uncertainties(out_std, keep, tag)


    if order_idx is not None:
        # matching the real indices to the array position
        reorder = np.array([np.where(idx_array == i) for i in order_idx]).flatten()
        uncertain_data_idx = idx_array[reorder]

        # selecting the corresposing std ordered according to order_idx
        uncertain_list_indices = reorder

        # Make sure the real indices match
        assert list(np.unique(uncertain_data_idx)) == list(np.unique(order_idx))

        # Make sure they are in the same order
        assert uncertain_data_idx.tolist() == order_idx.tolist(), logging.error("Ordering error")

    if not train_data:
        return data_copy, out_std[uncertain_list_indices], uncertain_data_idx, unlabelled_pool

    else:
        return data_copy, out_std, idx_array


def pandas_to_numpy_data(dataset: ITGDatasetDF, batch_size: int = None, shuffle: bool = True) -> DataLoader:
    """
    Helper function to convert pandas dataframe to numpy array and create a dataloader.
    Dataloaders created from numpy arrays are much faster than pandas dataframes.
    """

    x_array = dataset.data[train_keys].values
    y_array = dataset.data[dataset.label].values
    z_array = dataset.data[dataset.target].values
    idx_array = dataset.data["index"].values

    numpy_dataset = ITGDataset(x_array, y_array, z_array, idx_array)

    if batch_size is None:
        batch_size = int(0.1 * len(y_array))

    numpy_loader = DataLoader(numpy_dataset, batch_size=batch_size, shuffle=shuffle)
    return numpy_loader


# Active Learning diagonistic functions
def get_mse(y_hat: np.array, y: np.array) -> float:
    mse = np.mean((y - y_hat) ** 2)
    return mse


def mse_change(
    prediction_before: list,
    prediction_after: list,
    prediction_order: list,
    uncert_data_order: list,
    uncertain_dataset: Dataset,
    uncertainties: Union[None, list] = None,
    plot: bool = True,
    data: str = "novel",
    save_plots: bool = True,
    save_path:str = None, 
    iteration: int =None,
    lam: float = 1.0,
) -> (float, float, float):
    """
    Calculates the change in MSE between the before and after training.
    """

    if data == 'train':
        idxs = prediction_order.astype(int)
        ground_truth = uncertain_dataset.data.loc[idxs]

        ground_truth = ground_truth[uncertain_dataset.target]
        ground_truth = ground_truth.to_numpy()

        idx = np.isin(prediction_order, uncert_data_order)

        pred_before = prediction_before[idx]
        pred_after = prediction_after[idx]
        ground_truth_subset = ground_truth[idx]
    else:
        pred_before = prediction_before
        pred_after = prediction_after
        ground_truth = ground_truth[uncertain_dataste.target]
        ground_truth = ground_truth.to_numpy()

    mse_before = get_mse(pred_before, ground_truth_subset)
    mse_after = get_mse(pred_after, ground_truth_subset)
    perc_change = (mse_after - mse_before) * 100 / mse_before
    logging.info(f"% change in MSE for {data} dataset: {perc_change:.4f}%\n")

    if plot:
        plot_mse_change(
            ground_truth_subset,
            pred_before,
            pred_after,
            uncertainties,
            data=data,
            save_plots=save_plots,
            save_path=save_path, 
            iteration=iteration, 
            lam=lam,
            target=uncertain_dataset.target
        )

    return mse_before, mse_after, perc_change


def uncertainty_change(
    x: Union[list, np.array], 
    y: Union[np.array, list], 
    plot: bool = True, 
    plot_title: str = None, 
    iteration:int = None,
    save_path: str = "./",
) -> float:
    """
    Calculate the change in uncertainty after training for a given set of predictions
    with option to plot the results.

    """
    total = x.shape[0]
    increase = len(x[y > x]) * 100 / total
    decrease = len(x[y < x]) * 100 / total
    no_change = 100 - increase - decrease

    if plot:
        plot_scatter(x, y, plot_title, iteration, save_path)

    av_uncert_before = np.mean(x)
    av_uncer_after = np.mean(y)

    perc_change = (av_uncer_after - av_uncert_before) * 100 / av_uncert_before

    logging.info(
        f" Decreased {decrease:.3f}% Increased: {increase:.3f} % No Change: {no_change:.3f} "
    )

    logging.info(
        f"Initial Average Uncertainty: {av_uncert_before:.4f}, Final Average Uncertainty: {av_uncer_after:.4f}\n"
    )

    logging.info(f"% change: {perc_change:.5f}%")
    return perc_change


def plot_uncertainties(out_std: np.ndarray, keep: float, tag=None) -> None:
    """
    Plot the histogram of standard deviations of the the predictions,
    plotting the most uncertain points in a separate plot.
    """

    print('plotting...')
    plt.figure()
    plt.hist(out_std[np.argsort(out_std)[-int(len(out_std) * keep) :]], bins=50)

    name_uncertain = "standard_deviation_histogram_most_uncertain"
    if tag is not None:
        name_uncertain = f"{name_uncertain}_{tag}"
    plt.savefig(f"{name_uncertain}.png")
    plt.clf()

    plt.figure()
    plt.hist(out_std, bins=50)

    name = "standard_deviation_histogram"
    if tag is not None:
        name = f"{name}_{tag}"
    plt.savefig(f"{name}.png")
    plt.clf()


def plot_scatter(initial_std: np.ndarray, final_std: np.ndarray,title: str, it: int, save_dest: str) -> None:
    """
    Plot the scatter plot of the initial and final standard deviations of the predictions.
    """

    sns.jointplot(initial_std,final_std, kind='reg')

    plt.plot(
        [initial_std.min(), final_std.max()],
        [initial_std.min(), final_std.max()],
        "k--",
        lw=2,
    )
    plt.xlabel("Initial Standard Deviation")
    plt.ylabel("Final Standard Deviation")
    plt.title(title)
    save_path = os.path.join(save_dest, f"{it}_scatter_plot.png")
    plt.savefig(save_path)


def plot_mse_change(
    ground_truth: np.array,
    intial_prediction: np.array,
    final_prediction: np.array,
    uncertainties: list,
    target = "efiitg_gb",
    data: str = "novel",
    save_plots: bool = False,
    save_path=None,
    iteration=None,
    lam=1.0
) -> None:
    """

    Plot scatter plot of the regressor predictions vs the ground truth before and after training,
    option to color the points by uncertainty.
    """

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
        filename = f"{save_prefix}_mse_before_it_{iteration}.png"
        save_dest = os.path.join(save_path,target)

        if not os.path.exists(save_dest): os.mkdir(save_dest)

        save_dest = os.path.join(save_dest,f"{lam}")

        if not os.path.exists(save_dest): os.mkdir(save_dest)
        
        save_dest = os.path.join(save_dest,filename)
        
        plt.savefig(save_dest, dpi=300)

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
        filename = f"{save_prefix}_mse_after_it_{iteration}.png"
        save_dest = os.path.join(save_path,target)

        if not os.path.exists(save_dest): os.mkdir(save_dest)

        save_dest = os.path.join(save_dest,f"{lam}")

        if not os.path.exists(save_dest): os.mkdir(save_dest)
        
        save_dest = os.path.join(save_dest,filename)
        
        plt.savefig(save_dest, dpi=300)


def plot_classifier_retraining(
    train_loss: list,
    train_acc: list,
    val_loss: list,
    val_acc: list,
    missed_loss: list,
    missed_acc: list,
    save_path: Union[None, str] = None,
) -> None:
    """
    Plot the training and validation loss and accuracy for the classifier retraining.
    """

    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.plot(missed_loss, label="Misclassified points Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if save_path is not None:
        plt.savefig(f"{save_path}_classifier_loss_.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(missed_acc, label="Misclassified points Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    if save_path is not None:
        plt.savefig(f"{save_path}_classifier_accuracy_.png", dpi=300)
    plt.clf()
