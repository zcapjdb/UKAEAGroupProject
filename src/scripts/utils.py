import pandas as pd
import numpy as np
import os
import comet_ml

from pytorch_lightning.loggers import CometLogger
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch.utils.data import Dataset


def ScaleData(data: pd.DataFrame, scaler: object = None) -> pd.DataFrame:
    """
    Scale the data using the StandardScaler. If given a training set, fit the scaler to the training data.
    If given a validation or test set, use the fitted scaler to scale the test data.

    Inputs:
        data: a pandas dataframe containing the data to be scaled
        scaler: a sklearn StandardScaler object

    Outputs:
        data: a pandas dataframe containing the scaled data
        scaler: a StandardScaler object containing the fitted scaler if validating or testing
    """
    if scaler is None:
        scaler = StandardScaler()
        columns = data.columns
        data = scaler.fit_transform(data)  # scaler converts dataframe to numpy array!
        data = pd.DataFrame(data, columns=columns)

        return data, scaler

    else:
        columns = data.columns
        data = scaler.transform(data)
        data = pd.DataFrame(data, columns=columns)

        return data, scaler


def prepare_model(
    train_path: str,
    val_path: str,
    test_path: str,
    CustomDataset: Dataset,
    keys: list,
    comet_project_name: str,
    experiment_name: str,
    save_dir: str = "/share/rcifdata/jbarr/UKAEAGroupProject/logs",
):
    """
    Prepare the data and logging for training.

    Inputs:
        train_path: the path to the training data
        val_path: the path to the validation data
        test_path: the path to the test data
        CustomDataset: the dataset class to use for data loading
        keys: the dataframe columns to be used for training
        comet_project_name: the name of the comet project
        experiment_name: the name of the experiment
        save_dir: the directory to save the model

    Outputs:
        train_data: a Dataset object containing the training data
    """
    train_data = CustomDataset(train_path, columns=keys, train=True)
    train_data.scale()

    val_data = CustomDataset(val_path, columns=keys)
    val_data.scale()

    test_data = CustomDataset(test_path, columns=keys)
    test_data.scale()

    comet_api_key = os.environ["COMET_API_KEY"]
    comet_workspace = os.environ["COMET_WORKSPACE"]

    comet_logger = CometLogger(
        api_key=comet_api_key,
        project_name=comet_project_name,
        workspace=comet_workspace,
        save_dir=save_dir,  # TODO: figure out how this works so this can be more general
        experiment_name=experiment_name,
    )

    # can have memory issues if too many data points TODO: find out is there a way round this
    # comet_logger.experiment.log_dataframe_profile(train_data.data, name = 'train_data', minimal = True)

    return comet_logger, train_data, val_data, test_data


def callbacks(
    directory: str,
    run: str,
    experiment_name: str,
    top_k: int = 1,
    patience=25,
    swa_epoch=75,
) -> list:
    """
    Prepare the callbacks for training.

    Inputs:
        directory: the directory to save the model
        run: the run number for the model
        experiment_name: the name of the experiment

    Outputs:
        callbacks: a list of callbacks to be used for training
    """

    log_dir = (
        f"/share/rcifdata/jbarr/UKAEAGroupProject/logs/{directory}/{experiment_name}"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=patience
    )
    progress = TQDMProgressBar(refresh_rate=100)

    SWA = StochasticWeightAveraging(
        swa_epoch_start=swa_epoch
    )  # TODO base this off max epochs

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir,
        filename="{experiment_name}-{epoch:02d}-{val_loss:.2f}",
        save_top_k=top_k,
        mode="min",
    )

    return [early_stop_callback, progress, checkpoint_callback, SWA]


train_keys = [
    "ane",
    "ate",
    "autor",
    "machtor",
    "x",
    "zeff",
    "gammae",
    "q",
    "smag",
    "alpha",
    "ani1",
    "ati0",
    "normni1",
    "ti_te0",
    "lognustar",
]

target_keys = [
    "dfeitg_gb_div_efiitg_gb",
    "dfetem_gb_div_efetem_gb",
    "dfiitg_gb_div_efiitg_gb",
    "dfitem_gb_div_efetem_gb",
    "efeetg_gb",
    "efeitg_gb_div_efiitg_gb",
    "efetem_gb",
    "efiitg_gb",
    "efitem_gb_div_efetem_gb",
    "pfeitg_gb_div_efiitg_gb",
    "pfetem_gb_div_efetem_gb",
    "pfiitg_gb_div_efiitg_gb",
    "pfitem_gb_div_efetem_gb",
    "vceitg_gb_div_efiitg_gb",
    "vcetem_gb_div_efetem_gb",
    "vciitg_gb_div_efiitg_gb",
    "vcitem_gb_div_efetem_gb",
    "vfiitg_gb_div_efiitg_gb",
    "vfitem_gb_div_efetem_gb",
    "vriitg_gb_div_efiitg_gb",
    "vritem_gb_div_efetem_gb",
    "vteitg_gb_div_efiitg_gb",
    "vtiitg_gb_div_efiitg_gb",
]
