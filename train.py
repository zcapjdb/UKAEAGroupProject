import pathlib
import numpy as np
import pandas as pd
import os

import comet_ml
from pytorch_lightning.loggers import CometLogger

import torch
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from sklearn.preprocessing import StandardScaler

from QLKNN import QLKNN, QLKNN_Dataset, train_keys, target_keys

hyper_parameters = {
    'batch_size': 2048,
    'epochs': 25,
    'learning_rate': 0.001,
}

num_gpu = 3 # Make sure to request this in the batch script
accelerator = 'gpu'

comet_api_key = os.environ['COMET_API_KEY']
comet_workspace = os.environ['COMET_WORKSPACE']
comet_project_name = 'QLKNN-Regressor'

train_data_path = "data/QLKNN_train_data.pkl"
val_data_path = "data/QLKNN_validation_data.pkl"
test_data_path = "data/QLKNN_test_data.pkl"


def prepare_model(target):
    train_data = QLKNN_Dataset(train_data_path, columns = train_keys + [target])
    val_data = QLKNN_Dataset(val_data_path, columns = train_keys + [target])
    test_data = QLKNN_Dataset(test_data_path, columns = train_keys + [target])

    # maybe cleaner to do this in the dataset class?
    scaler = StandardScaler()
    scaler.fit(train_data.data)

    train_data.data = scaler.transform(train_data.data) # scaler converts dataframe to numpy array!
    train_data.data = pd.DataFrame(train_data.data, columns = train_keys + [target])

    val_data.data = scaler.transform(val_data.data)
    val_data.data = pd.DataFrame(val_data.data, columns = train_keys + [target])

    test_data.data = scaler.transform(test_data.data)
    test_data.data = pd.DataFrame(test_data.data, columns = train_keys + [target])

    comet_logger = CometLogger(api_key = comet_api_key, project_name = comet_project_name,
                        workspace = comet_workspace, save_dir = './logs', experiment_name = target)

    comet_logger.log_hyperparams(hyper_parameters)

    return comet_logger, train_data, val_data, test_data


def main():

    for target in target_keys:
        print(f"Training model for {target}")
        comet_logger, train_data, val_data, test_data = prepare_model(target)

        model = QLKNN(n_input = 15, **hyper_parameters)
        print(model)

        train_loader = DataLoader(train_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)
        val_loader = DataLoader(val_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)
        test_loader = DataLoader(test_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)

        early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = 4)
        progress = TQDMProgressBar(refresh_rate = 50)

        log_dir = "logs/" + target
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        checkpoint_callback = ModelCheckpoint(
            monitor = "val_loss",
            dirpath = log_dir,
            filename = "{target}-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )

        trainer = Trainer(max_epochs = hyper_parameters['epochs'],
            logger = comet_logger,
            accelerator = accelerator,
            strategy = DDPPlugin(find_unused_parameters=False),
            devices = num_gpu,
            callbacks = [early_stop_callback, progress, checkpoint_callback],
            log_every_n_steps=50)

        
        trainer.fit(model = model, train_dataloaders = train_loader, val_dataloaders = val_loader)
        comet_logger.log_graph(model)

        trainer.test(dataloaders = test_loader)


if __name__ == '__main__':
    main()