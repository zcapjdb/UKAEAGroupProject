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

from QLKNN import QLKNN, QLKNN_Dataset
from utils import train_keys, target_keys, prepare_model 

hyper_parameters = {
    'batch_size': 4096,
    'epochs': 75,
    'learning_rate': 0.001,
}

num_gpu = 3 # Make sure to request this in the batch script
accelerator = 'gpu'

run = "4"

train_data_path = "data/QLKNN_train_data.pkl"
val_data_path = "data/QLKNN_validation_data.pkl"
test_data_path = "data/QLKNN_test_data.pkl"

comet_api_key = os.environ['COMET_API_KEY']
comet_workspace = os.environ['COMET_WORKSPACE']
comet_project_name = 'QLKNN-Regressor'

def main():

    comet_logger_main = CometLogger(api_key = comet_api_key, 
        project_name = comet_project_name,
        workspace = comet_workspace,
        save_dir = './logs',
        experiment_name = f'Run-{run}-main')

    for target in target_keys:
        print(f"Training model for {target}")
        experiment_name = f"Run-{run}-{target}"
        keys = train_keys + [target]

        comet_logger, train_data, val_data, test_data = prepare_model(train_data_path, val_data_path,
        test_data_path, QLKNN_Dataset, keys, comet_project_name, experiment_name)

        model = QLKNN(n_input = 15, **hyper_parameters)
        print(model)

        comet_logger.log_hyperparams(hyper_parameters)

        train_loader = DataLoader(train_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)
        val_loader = DataLoader(val_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)
        test_loader = DataLoader(test_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 20)

        early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = 10)
        progress = TQDMProgressBar(refresh_rate = 50)

        log_dir = "logs/" + f"Run-{run}/"+ experiment_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        checkpoint_callback = ModelCheckpoint(
            monitor = "val_loss",
            dirpath = log_dir,
            filename = "{target}-{epoch:02d}-{val_loss:.2f}",
            save_top_k = 1,
            mode="min",
        )

        trainer = Trainer(max_epochs = hyper_parameters['epochs'],
            logger = comet_logger,
            accelerator = accelerator,
            strategy = DDPPlugin(find_unused_parameters = False),
            devices = num_gpu,
            callbacks = [early_stop_callback, progress, checkpoint_callback],
            log_every_n_steps = 50)

        
        trainer.fit(model = model, train_dataloaders = train_loader, val_dataloaders = val_loader)
        comet_logger.log_graph(model)

        trainer.test(dataloaders = test_loader)

        comet_logger_main.log_metrics(model.metrics, step = target)

if __name__ == '__main__':
    main()