import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import copy
import logging
from scripts.utils import train_keys, target_keys
from pipeline.Models import ITGDatasetDF, ITGDataset, Classifier, Regressor, EnsembleRegressor
from typing import Union
import os

output_dict = {
    "train_loss_init": [],  # Regressor performance before pipeline
    "test_loss_init": [],
    "test_loss_init_unscaled": [],
    "retrain_losses": [], # regressor performance during retraining
    "retrain_val_losses": [],
    "retrain_losses_unscaled": [],
    "retrain_val_losses_unscaled": [],
    "post_test_loss": [],  # regressor performance after retraining
    "post_test_loss_unscaled": [],  # regressor performance after retraining, unscaled
    "popback": [],  # regressor performance after retraining, unscaled, normalised
    "mean_scaler": [],
    "scale_scaler": [],
    "n_train_points": [],
    "n_candidates_classifier":[],
    "loss_0_5":[],
    "loss_20_25":[],
    "loss_40_45":[],
    "loss_60_65":[],
    "loss_80_85":[],
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

    "class_train_loss_init": [], 
    "class_test_loss_init": [],
    "class_train_acc_init": [],
    "class_test_acc_init": [],
    "class_precision_init": [],
    "class_recall_init": [],
    "class_f1_init": [],
    "class_auc_init": [],

    "class_train_loss": [],
    "class_val_loss": [],
    "class_missed_loss": [],
    "class_train_acc": [],
    "class_val_acc": [],
    "class_missed_acc": [],
    "holdout_class_acc": [],
    "holdout_class_loss": [],
    "holdout_class_precision": [],
    "holdout_class_recall": [],
    "holdout_class_f1": [],
    "holdout_class_auc": [],
    "class_retrain_iterations": [],
}



# Data preparation functions
def prepare_data(
    train_path: str,
    valid_path: str,
    test_path: str,
    fluxes: list,
    train_size: int = None,
    valid_size: int = None,
    test_size: int = None,
    samplesize_debug: int = 1,
    scale: bool = True,
) -> (ITGDatasetDF, ITGDatasetDF, ITGDatasetDF, StandardScaler):
    """
    Loads the data from the given paths and prepares it for training.
    train_path, valid_path point to pickle files containing the data in dataframes.
    """

    train_data = pd.read_pickle(train_path)
    validation_data = pd.read_pickle(valid_path)
    test_data = pd.read_pickle(test_path)

    if train_size is not None:
        if len(train_data)>samplesize_debug*train_size:
            train_data = train_data.sample(samplesize_debug * train_size)
        else:
            pass

    if valid_size is not None:
        if len(validation_data)> samplesize_debug*valid_size:
            validation_data = validation_data.sample(samplesize_debug * valid_size)
        else:
            pass

    if test_size is not None:
        if len(test_data)>samplesize_debug*test_size:
            test_data = test_data.sample(samplesize_debug * test_size)
        else:
            pass

    target_column = fluxes[0]

    if target_column not in target_keys:
        raise ValueError("Flux variable not supported")

    # Remove NaN's and add appropripate class labels
    keep_keys = train_keys + fluxes

    logging.debug(f"keep keys: {keep_keys}")

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]
    test_data = test_data[keep_keys]

    train_data = train_data.dropna(subset=[target_column]) 
    validation_data = validation_data.dropna(subset =[target_column])
    test_data = test_data.dropna(subset=[target_column])


    train_data["stable_label"] = np.where(train_data[target_column] != 0, 1, 0)
    validation_data["stable_label"] = np.where(
        validation_data[target_column] != 0, 1, 0
    )
    test_data["stable_label"] = np.where(test_data[target_column] != 0, 1, 0)

    train_dataset = ITGDatasetDF(train_data, target_column=target_column)
    valid_dataset = ITGDatasetDF(validation_data, target_column=target_column)
    test_dataset = ITGDatasetDF(test_data, target_column=target_column)

    train_dataset_regressor = copy.deepcopy(train_dataset)
    train_dataset_regressor.data = train_dataset.data.drop(train_dataset.data[train_dataset.data["stable_label"] == 0].index)

    if scale: 
        scaler = StandardScaler()
        scaler.fit(train_data.drop(["stable_label", "index"], axis=1))

        train_dataset.scale(scaler)
        valid_dataset.scale(scaler)
        test_dataset.scale(scaler)
        train_dataset_regressor.scale(scaler)

        return train_dataset, valid_dataset, test_dataset, scaler, train_dataset_regressor
    
    else:
        return train_dataset, valid_dataset, test_dataset, None, train_dataset_regressor

def scale_data(train_regr, train_class, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, scaler=None, unscale=False):
    if scaler is None: # --- ensures that future tasks can be scaled according to scaler of previous tasks
        scaler = StandardScaler()
        scaler.fit(train_regr.data.drop(["stable_label","index"], axis=1))
    train_regr.scale(scaler,unscale=unscale)
    train_class.scale(scaler,unscale=unscale)
    # -- scale eval now so don't scale its derivative dsets later
    valid_dataset.scale(scaler,unscale=unscale)
    valid_classifier.scale(scaler,unscale=unscale)
    holdout_set.scale(scaler,unscale=unscale)
    holdout_classifier.scale(scaler,unscale=unscale)
    unlabelled_pool.scale(scaler,unscale=unscale) 
    return train_regr, train_class, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, scaler

def get_data(cfg,scaler=None,apply_scaler=True,j=None):
    PATHS = cfg["data"]
    FLUX = cfg["flux"]        
    train_class, eval_dataset, test_dataset, scaler_, train_regressor = prepare_data(
            PATHS["train"],
            PATHS["validation"],
            PATHS["test"],
            fluxes=FLUX,
            samplesize_debug=1,
            scale=False,
        )
    if scaler is None:
        scaler = scaler_
    
    print('size of train, test, val', len(train_regressor),len(test_dataset), len(eval_dataset))
    # --- train sets
    if len(train_regressor)>cfg['hyperparams']['train_size']:
        train_regr = train_regressor.sample(cfg['hyperparams']['train_size'])
    else:
        train_regr = copy.deepcopy(train_regressor)

    # --- holdout sets are from the test set
    if len(test_dataset)>cfg['hyperparams']['test_size']:
        holdout_set = test_dataset.sample(cfg['hyperparams']['test_size'])  # holdout set
    else:
        holdout_set = copy.deepcopy(test_dataset)
    print('LEN TEST SET AT START', len(holdout_set)) 
           
    holdout_classifier = copy.deepcopy(holdout_set) # copy it for classifier
    holdout_set.data = holdout_set.data.drop(holdout_set.data[holdout_set.data["stable_label"] == 0].index) # delete stable points for regressor
    if j is not None: # --- only for CL
        #--- save unscaled test data
        save_class = copy.deepcopy(holdout_classifier)
        save_regr = copy.deepcopy(holdout_set)
        saved_tests = {f'task{j}': {'save_class':save_class,'save_regr':save_regr}} 
    else:
        saved_tests = None

    # --- valid sets
    eval_regressor = copy.deepcopy(eval_dataset)  
    eval_regressor.data = eval_regressor.data.drop(eval_regressor.data[eval_regressor.data["stable_label"] == 0].index)
    valid_dataset = eval_regressor.sample(cfg['hyperparams']['valid_size']) # --- valid for regressor
    valid_classifier = eval_dataset.sample(cfg['hyperparams']['valid_size'])  # --- valid for classifier, must come from original eval
    if len(train_class)>cfg['hyperparams']['train_size']:
        train_class = train_class.sample(cfg['hyperparams']['train_size'])
    else:
        pass

    # --- unlabelled pool is from the evaluation set minus the validation set (note, I'm not using "validation" and "evaluation" as synonyms)
    eval_dataset.remove(valid_dataset.data.index) 
    eval_dataset.remove(valid_classifier.data.index)
    unlabelled_pool = eval_dataset 

    if apply_scaler:
        train_regr, train_class, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, scaler = scale_data(train_regr, train_class, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, scaler=scaler)
        print('LEN TEST SET AT START', len(holdout_set)) 
    return train_regr, train_class, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, saved_tests, scaler



def get_regressor_model(
    regressor_type,
    device,
    scaler,
    flux,
    dropout,
    model_size,
    num_estimators = None):
        if regressor_type=='Regressor':
            print('WARNING: passed num_estimators but single regressor is trained')
            return Regressor(
            device=device, 
            scaler=scaler, 
            flux=flux, 
            dropout=dropout,
            model_size=model_size
            ) 
        elif regressor_type == 'EnsembleRegressor':
            if num_estimators is not None:
                return EnsembleRegressor(
                num_estimators = 10,
                device=device, 
                scaler=scaler, 
                flux=flux, 
                dropout=dropout,
                model_size=model_size
                )    
            else:
                raise ValueError('EnsembleRegressor needs number of estimators >1')

# classifier tools
def select_unstable_data(
    dataset: ITGDatasetDF,
    batch_size: int,
    classifier: Classifier,
    device: torch.device = None,
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

        pred_class = torch.round(y_hat.squeeze().detach().cpu()).numpy()
        pred_class = pred_class.astype(int)

        unstable = idx[np.where(pred_class == 1)[0]]
        unstable_points.append(unstable.detach().cpu().numpy())

    # turn list of stable and misclassified points into flat arrays
    unstable_points = np.concatenate(np.asarray(unstable_points, dtype=object), axis=0)

    logging.log(15, f"Unstable points: {len(unstable_points)}")

    # create new dataset with misclassified points
    unstable_candidates = copy.deepcopy(dataset)
    unstable_candidates.data = unstable_candidates.data.loc[unstable_points]

    return unstable_candidates


def check_for_misclassified_data(
    candidates: ITGDatasetDF, 
    uncertainty: Union[np.array, list], 
    indices: Union[np.array, list]
    ) -> ITGDatasetDF:
    
    candidate_loader = DataLoader(candidates, batch_size=1, shuffle=False)

    missed_points = []
    for (x, y, z, idx) in candidate_loader:
        # if y == 0, then it is misclassified, keep only the misclassified points
        if y.item() == 0:
            missed_points.append(idx.item())

    #!!!!!!!!!!!!!!!!!!!!SOMETHING IS SERIOUSLY WRONG HERE!!!!!!!!!!!!!!!!!!!!
    # create new dataset with misclassified points
    missed_candidates = copy.deepcopy(candidates)
    missed_candidates.data = missed_candidates.data.loc[missed_points]
    
    candidates.remove(missed_points)

    candidate_indices = np.array(list(candidates.data.index))
    indices = np.array(indices)
    # find the overlap of the two index lists
    mask = np.isin(indices, candidate_indices)
    indices = indices[mask]
    uncertainty = np.array(uncertainty)[mask]

    return candidates, missed_candidates.data, len(missed_points), indices, uncertainty


# Function to retrain the classifier on the misclassified points
def retrain_classifier(
    training_dataset: ITGDatasetDF,
    valid_dataset: ITGDataset,
    classifier: Classifier,
    learning_rate: int = 5e-4,
    epochs: int = 10,
    batch_size: int = 1024,
    validation_step: bool = True,
    lam: Union[float, int] = 1,
    loc: float = 0.0,
    scale: float = 0.05,
    patience: Union[None, int] = None,
    disable_tqdm: bool = True,
) -> (list, list, list, list, list, list):
    """
    Retrain the classifier on the misclassified points.
    Data for retraining is taken from the combined training and misclassified datasets.
    Returns the losses and accuracies of the training and validation steps.
    """

    logging.info("Retraining classifier...\n")

    train = copy.deepcopy(training_dataset)

    # create data loaders
    train_loader = pandas_to_numpy_data(train, batch_size=batch_size, shuffle=True)
    valid_loader = pandas_to_numpy_data(valid_dataset)

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    classifier.shrink_perturb(lam, loc, scale)

    if not patience:
        patience = epochs

    # instantiate optimiser
    opt = torch.optim.Adam(
        classifier.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
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

        loss, acc = classifier.train_step(
            train_loader, opt, epoch=epoch, disable_tqdm=disable_tqdm
        )
        train_loss.append(loss.item())
        train_acc.append(acc)

        if validation_step:

            logging.debug(f"Validation Step: {epoch}")

            validation_loss, validation_accuracy = classifier.validation_step(
                valid_loader, scheduler
            )
            val_loss.append(validation_loss)
            val_acc.append(validation_accuracy)



        if len(val_loss) > patience:
            if np.mean(val_acc[-patience:]) > val_acc[-1]:
                logging.debug("Early stopping criterion reached")
                break

    return [train_loss, val_loss], [train_acc, val_acc]


# Regressor tools
def reorder_arrays(arrays:list, orders:list, arrangement:np.array):
    """
    Inputs:
        arrays: The arrays to be reordered
        orders: The current ordering of the arrays
        arrangements: The desired orderings
    Outputs: 
        arrays: a list of the original arrays ordering according to arrangement
    """

    for k in range(len(arrays)):
        assert len(orders[k]) == len (arrangement), f"Length of arrays to reorder {len(orders[k])} doesn't match the reordering indices {len(arrangement)}"
        reorder = np.array([np.where(orders[k] == i) for i in arrangement]).flatten()
        # remember to comment the line below out
        arrays[k] = arrays[k][reorder]
    
    return arrays

def retrain_ensemble(    
    new_dataset: ITGDatasetDF,
    val_dataset: ITGDatasetDF,
    ensemble: EnsembleRegressor,
    learning_rate: int = 1e-4,
    epochs: int = 10,
    validation_step: bool = True,
    lam: Union[float, int] = 1,
    loc: float = 0.0,
    scale: float = 0.01,
    patience: Union[None, int] = None,
    batch_size: int = 1024,
    disable_tqdm: bool = True):
    for regressor in ensemble.regressors:
        retrain_regressor(new_dataset,
                val_dataset,
                regressor,
                learning_rate=learning_rate,
                epochs=epochs,
                validation_step=validation_step,
                lam=lam,
                patience=patience,
                batch_size=batch_size)
        

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
) -> (list, list, list, list):
    """
    Retrain the regressor on the most uncertain points.
    Data for retraining is taken from the combined training and uncertain datasets.
    Returns the losses of the training and validation steps.
    """


    if model.type=='ensemble':
        retrain_ensemble(
            new_dataset,
            val_dataset,
            model,
            learning_rate=learning_rate,
            epochs=epochs,
            validation_step=True,
            lam=lam,
            patience=patience,
            batch_size=batch_size,
            )        
        return

    logging.info(f"Retraining {model.flux} regressor...\n")
    logging.log(15, f"Training on {len(new_dataset)} points")
    # variable the regressor is trained on

    # are we training on any NaNs
    logging.debug(f"NaNs in training: {new_dataset.data[model.flux].isna().sum()}")
    logging.debug(f"NaNs in valid: {val_dataset.data[model.flux].isna().sum()}")
    #TODO: this is a poor fix
    
    new_copy = copy.deepcopy(new_dataset)
    val_copy = copy.deepcopy(val_dataset)

    new_copy.data.dropna(subset=[model.flux], inplace=True)
    val_copy.data.dropna(subset=[model.flux], inplace=True)

    logging.debug(f"NaNs in training: {new_copy.data[model.flux].isna().sum()}")
    logging.debug(f"NaNs in vald: {val_copy.data[model.flux].isna().sum()}")

    new_loader = pandas_to_numpy_data(
        new_copy, regressor_var=model.flux, batch_size=batch_size, shuffle=True
    )
    val_loader = pandas_to_numpy_data(
        val_copy, regressor_var=model.flux, batch_size=batch_size, shuffle=False
    )

    # By default passing lambda = 1 corresponds to a warm start (loc and scale are ignored in this case)
    model.shrink_perturb(lam, loc, scale)

    if validation_step:
        val_loss, val_loss_unscaled = model.validation_step(val_loader)
        logging.log(15, f"Initial validation loss: {val_loss.item():.4f}")
        logging.log(15, f"Initial validation loss (unscaled): {val_loss_unscaled.item():.4f}")

    if not patience:
        patience = epochs

    model.train()

    # instantiate optimiser
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=0.25 * patience,
        min_lr=(1 / 32) * learning_rate,
    )
    train_loss = []
    train_loss_unscaled = []
    val_loss = []
    val_loss_unscaled = []

    for epoch in range(epochs):
        logging.log(15, f"Epoch: {epoch}")
        loss, unscaled_loss = model.train_step(new_loader, opt, epoch=epoch, disable_tqdm=disable_tqdm)

        train_loss.append(loss.item())
        train_loss_unscaled.append(unscaled_loss.item())

        if validation_step:
            loss, loss_unscaled = model.validation_step(val_loader, scheduler)
            val_loss.append(loss)
            val_loss_unscaled.append(loss_unscaled)

        if len(val_loss) > patience:
            if np.mean(val_loss[-patience:]) < val_loss[-1]:
                logging.log(15, "Early stopping criterion reached")
                break

    return #train_loss, val_loss, train_loss_unscaled, val_loss_unscaled


def plot_TSNE(
    uncert: ITGDatasetDF,
    train: ITGDatasetDF,
    iter_num: int,
) -> None:
    temp_train = copy.deepcopy(train)
    start_of_uncert_index = len(temp_train)
    temp_uncert = copy.deepcopy(uncert)
    temp_train.add(temp_uncert)
    temp_train.index = np.arange(len(temp_train))

    print('RUNNING TSNE')
    temp_train =  temp_train.data.drop(["stable_label","index"], axis=1)
   # tsne = TSNE(random_state=42, perplexity=50,learning_rate='auto' )
   # X = tsne.fit_transform(temp_train.values)

#    fig, ax = plt.subplots(4,4, figsize=(24,24))
#    for h,(a,c) in enumerate(zip(ax.ravel(),temp_train.columns)):
#        im = a.scatter(X[0:start_of_uncert_index,0],X[0:start_of_uncert_index,1], s=5, c=temp_train.iloc[0:start_of_uncert_index,h].values)
#        a.scatter(X[start_of_uncert_index:,0],X[start_of_uncert_index:,1], s=5, color='red',label='selection')
#        divider = make_axes_locatable(a)
#        cax = divider.append_axes('right', size='5%', pad=0.05)
#        cbar = fig.colorbar(im, cax=cax, orientation='vertical')        
#        cbar.set_label(c)
#        #plt.legend()
#        a.set_xlabel('x1')
#        a.set_ylabel('x2')
#    fig.subplots_adjust(wspace=0.3)
#    fig.savefig(f'./debug/uncert/tsne{iter_num}.png')
#    fig.clf()

    fig, ax = plt.subplots(4,4, figsize=(24,24))
    bins = np.arange(-5,5,0.5)
    for h,(a,c) in enumerate(zip(ax.ravel(),temp_train.columns)):
        a.hist(temp_train.iloc[0:start_of_uncert_index,h].values,bins=bins,color='blue',lw=3, histtype='step', density=True)
        a.axvline(np.median(temp_train.iloc[0:start_of_uncert_index,h].values), color='blue', ls=':', lw=3)
        a.hist(temp_train.iloc[start_of_uncert_index:,h].values,bins=bins,color='red', histtype='step',lw=3, density=True)
        a.axvline(np.median(temp_train.iloc[start_of_uncert_index:,h].values), color='red', ls='--', lw=3)
        a.set_xlabel(c,fontsize=40)
    fig.subplots_adjust(wspace=0.3)
    ax[0][0].text(-4,0.4,f"iter: {iter_num}", fontsize=25)
    fig.tight_layout()
    fig.savefig(f'./debug/hist/hist{iter_num}.png')
    
    fig.clf()        
    return 



def get_uncertainty(
    dataset: ITGDatasetDF,
    regressor: Regressor,
    n_runs: int = 25,
    order_idx: Union[None, list, np.array] = None,
    train_data: bool = False,
    plot: bool = False,
    iter_num: int = None,
    device: torch.device = None,
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
    dataloader = pandas_to_numpy_data(
        data_copy, regressor_var=regressor.flux, batch_size=batch_size, shuffle=False
    )

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data n_runs times and return points with largest uncertainty
    runs = []

    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):
            x = x.to(device)
            z = z.to(device)
            predictions = regressor(x.float()).detach().cpu().numpy()
            
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
        idx_list.append(idx.detach().cpu().numpy())

    # flatten the list of indices
    flat_list = [item for sublist in idx_list for item in sublist]
    idx_array = np.asarray(flat_list, dtype=object).flatten()

    if plot:
        if order_idx == None:
            tag = "Initial"
        else:
            tag = "Final"
        keep = 0.25
        plot_uncertainties(out_std, keep, tag,i=iter_num)

    if order_idx is not None:
        # matching the real indices to the array position
        reorder = np.array([np.where(idx_array == i) for i in order_idx]).flatten()
        uncertain_data_idx = idx_array[reorder]

        # selecting the corresposing std ordered according to order_idx
        uncertain_list_indices = reorder

        # Make sure the real indices match
        assert list(np.unique(uncertain_data_idx)) == list(np.unique(order_idx))

        # Make sure they are in the same order
        assert uncertain_data_idx.tolist() == order_idx.tolist(), logging.error(
            "Ordering error"
        )
        
        return out_std[reorder], uncertain_data_idx
    else:
        return out_std, idx_array


def get_most_uncertain(
    dataset: ITGDatasetDF,
    out_stds: Union[list, np.array],
    idx_arrays: Union[list, np.array],
    model: Regressor,
    keep: float = 0.25,
    unlabelled_pool: Union[None, ITGDataset] = None,
    plot: bool = True,
    acquisition: str = "add_uncertainties",
    alpha: float = 1,
):

    """
    Inputs:

        dataset: dataset of points that MC drop out was ran on
        out_stds: standard deviations from MC dropout from regressors
        idx_arrays: order of the datapoints from dataloaders
        keep: percentage of most uncertain points to keep
        plot: Whether to plot the distribution of the uncertainties

    Outputs:
        data_copy: a dataset object containing only the most uncertain points
        out_std: standard deviation of the most ucnertain points
        idx_array: idx_array used for ordering  (which one have we followed)

    """
    logging.debug(f"Getting most uncertain for: {model.flux}")
    data_copy = copy.deepcopy(dataset)
    n_candidates = out_stds[0].shape[0]
    #keep = int(n_candidates*keep)
    keep = 500

    if len(out_stds) > 1:
        print('BAZINGA')
        assert len(idx_arrays) == len(out_stds), "N indices doesn't match N stds"
        pred_list = []
        # reorder idx_arrays to match the order of idx_arrays[0]
        for i in range(len(idx_arrays)):
            reorder = np.array(
                [np.where(idx_arrays[i] == j) for j in idx_arrays[0]]
            ).flatten()
            out_stds[i] = out_stds[i][reorder]

            # run predict method on dataset using idx_arrays ordering
            data_copy.data = data_copy.data.loc[idx_arrays[0]]
            preds, _ = model.predict(data_copy)
            preds = np.hstack(preds)
            pred_list.append(preds)

        out_stds = np.array(out_stds)

        if acquisition == "leading_flux_uncertainty":
            total_std = out_stds[0, :]

        else:
            total_std = np.sum(out_stds, axis=0)

        pred_array = np.stack(pred_list, axis=0)

    else:
        total_std = out_stds[0]
        data_copy.data = data_copy.data.loc[idx_arrays[0]]
    #    pred_array, _ = model.predict(data_copy)    

    #TODO: how to best choose alpha?
    if acquisition == "distance_penalty":
        logging.info("Using distance penalty acquisition")
        # cdist returns a matrix of distances between each pair of points

        cdists = cdist(data_copy.data[train_keys].values, data_copy.data[train_keys].values, metric = "euclidean")
        cdists.sort()
        median_dist = np.median(cdists, axis=1)

        total_std = total_std + alpha * median_dist# nearest

    if acquisition == "random":
        logging.info("Using random acquisition")
        # choose random indices
        #uncertain_list_indices = np.random.choice(n_candidates, int(keep * n_candidates), replace=False)
        uncertain_list_indices = np.random.choice(n_candidates, keep, replace=False)
        #random_certain is all the indices not in random_idx
        certain_list_indices = np.array(list(set(range(n_candidates)) - set(uncertain_list_indices)))
        
    else:
        #uncertain_list_indices = np.argsort(total_std)[-int(n_candidates*keep):]
        uncertain_list_indices = np.argsort(total_std)[-keep:]
        #certain_list_indices = np.argsort(total_std)[:n_candidates-int(n_candidates*keep)]
        certain_list_indices = np.argsort(total_std)[:n_candidates-keep)]



    certain_data_idx = idx_arrays[0][certain_list_indices]
    uncertain_data_idx = idx_arrays[0][uncertain_list_indices]

    # Take the points that are not in the most uncertain points and add back into the validation set
    temp_dataset = copy.deepcopy(dataset)
    temp_dataset.remove(indices=uncertain_data_idx)
    unlabelled_pool.add(temp_dataset)

    logging.log(15, f"no valid after : {len(unlabelled_pool)}")

    del temp_dataset

    # Remove them from the sample
    data_copy.remove(certain_data_idx)

    out_idx = idx_arrays[0][uncertain_list_indices]
    nans = dataset.data[model.flux].loc[out_idx]

    logging.debug(f"NaN inputs selected:{nans.isna().sum()}")
    logging.debug(f"Number of selected points {len(out_idx)} ")

    return data_copy, total_std[uncertain_list_indices], uncertain_data_idx, unlabelled_pool


def regressor_uncertainty(
    dataset: ITGDatasetDF,
    regressor: Regressor,
    keep: float = 0.25,
    n_runs: int = 25,
    train_data: bool = False,
    plot: bool = False,
    order_idx: Union[None, list, np.array] = None,
    unlabelled_pool: Union[None, ITGDataset] = None,
    device: torch.device = None,
    iter_num: int=None
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
    dataloader = pandas_to_numpy_data(
        data_copy, regressor_var=regressor.flux, batch_size=batch_size, shuffle=False
    )

    regressor.eval()
    regressor.enable_dropout()

    # evaluate model on training data n_runs times and return points with largest uncertainty
    runs = []

    for i in tqdm(range(n_runs)):
        step_list = []
        for step, (x, y, z, idx) in enumerate(dataloader):
            x = x.to(device)
            predictions = regressor(x.float()).detach().cpu().numpy()
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
        idx_list.append(idx.detach().cpu().numpy())

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

        plot_uncertainties(out_std, keep,iter_num)

    if order_idx is not None:
        # matching the real indices to the array position
        reorder = np.array([np.where(idx_array == i) for i in order_idx]).flatten()
        uncertain_data_idx = idx_array[reorder]

        # selecting the corresposing std ordered according to order_idx
        uncertain_list_indices = reorder

        # Make sure the real indices match
        assert list(np.unique(uncertain_data_idx)) == list(np.unique(order_idx))

        # Make sure they are in the same order
        assert uncertain_data_idx.tolist() == order_idx.tolist(), logging.error(
            "Ordering error"
        )

    if not train_data:
        return (
            data_copy,
            out_std[uncertain_list_indices],
            uncertain_data_idx,
            unlabelled_pool,
        )

    else:
        return data_copy, out_std, idx_array


def pandas_to_numpy_data(
    dataset: ITGDatasetDF,
    regressor_var: str = None,
    batch_size: int = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Helper function to convert pandas dataframe to numpy array and create a dataloader.
    Dataloaders created from numpy arrays are much faster than pandas dataframes.
    """
    if regressor_var == None:
        logging.debug(f"{dataset.target}")
        regressor_var = dataset.target
        logging.info(f"No regressor value chosen, setting z to {regressor_var}")


    x_array = dataset.data[train_keys].values
    y_array = dataset.data[dataset.label].values
    z_array = dataset.data[regressor_var].values
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
    save_path: str = None,
    iteration: int = None,
    lam: float = 1.0,
) -> (float, float, float):
    """
    Calculates the change in MSE between the before and after training.
    """

    if data == "train":
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
            target=uncertain_dataset.target,
        )

    return mse_before, mse_after, perc_change


def uncertainty_change(
    x: Union[list, np.array],
    y: Union[np.array, list],
    plot: bool = False,
    plot_title: str = None,
    iteration: int = None,
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

    #TODO: fix! strange this occuring with the shapes
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


def plot_uncertainties(out_std: np.ndarray, keep: float,path:str=None, tag=None,i:int=None,) -> None:
    """
    Plot the histogram of standard deviations of the the predictions,
    plotting the most uncertain points in a separate plot.
    """

    print("plotting...")
    plt.figure(figsize=(8,8))

#    name_uncertain = "standard_deviation_histogram_most_uncertain"
#    if tag is not None:
#        name_uncertain = f"{name_uncertain}_{tag}"
#    plt.savefig(f"{name_uncertain}.png")
#    plt.clf()

    plt.hist(out_std,bins=np.arange(0,0.5,0.01), color='blue', histtype='step', lw=5)
    plt.axvline(np.median(out_std), color='blue')
    plt.hist(out_std[np.argsort(out_std)[-int(len(out_std) * keep) :]], bins=np.arange(0,0.5,0.01),ls=':' , lw=5, color='red',histtype='step')
    plt.axvline(np.median(out_std[np.argsort(out_std)[-int(len(out_std) * keep) :]]), color='red')

    plt.title(f"iteration {i}")
    name = "standard_deviation_histogram"
    if tag is not None:
        name = f"{name}_{tag}"
    plt.savefig(f"./debug/uncert/{name}_{i}.png")
    plt.clf()


def plot_scatter(
    initial_std: np.ndarray, final_std: np.ndarray, title: str, it: int, save_dest: str
) -> None:
    """
    Plot the scatter plot of the initial and final standard deviations of the predictions.
    """
    logging.debug(f" plot x shape: {initial_std.shape}")
    logging.debug(f" plot y shape: {final_std.shape}")
    
    #plot = sns.jointplot(initial_std, final_std, kind="reg")
    plt.figure(figsize=(8,8))
    plt.scatter(initial_std, final_std)
    plt.xlabel("before")
    plt.ylabel("after")    
    plt.xlim(0,0.5)
    plt.ylim(0,0.5)
    xx = np.arange(0,0.5,0.1)
    plt.plot(xx,xx, ls=':', color='red',lw=2)

#    plt.plot(
#        [initial_std.min(), final_std.max()],
#        [initial_std.min(), final_std.max()],
#        "k--",
#        lw=2,
#    )

    plt.title(f"iteration {it}")
    save_path = os.path.join(save_dest, f"{it}_scatter_plot.png")
    plt.savefig(save_path)
    plt.close()


def plot_mse_change(
    ground_truth: np.array,
    intial_prediction: np.array,
    final_prediction: np.array,
    uncertainties: list,
    target="efiitg_gb",
    data: str = "novel",
    save_plots: bool = False,
    save_path=None,
    iteration=None,
    lam=1.0,
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
        save_dest = os.path.join(save_path, target)

        if not os.path.exists(save_dest):
            os.mkdir(save_dest)

        save_dest = os.path.join(save_dest, f"{lam}")

        if not os.path.exists(save_dest):
            os.mkdir(save_dest)

        save_dest = os.path.join(save_dest, filename)

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
        save_dest = os.path.join(save_path, target)

        if not os.path.exists(save_dest):
            os.mkdir(save_dest)

        save_dest = os.path.join(save_dest, f"{lam}")

        if not os.path.exists(save_dest):
            os.mkdir(save_dest)

        save_dest = os.path.join(save_dest, filename)

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
