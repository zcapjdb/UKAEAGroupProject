import comet_ml
import torch
import sys
path = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/'
path2 = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/scripts'
sys.path.append(path)
sys.path.append(path2)
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

from QLKNN import BaseQLKNN, QLKNN, QLKNNDataset
from scripts.utils import train_keys, target_keys, prepare_model, callbacks



patience = 20
swa_epoch = 75

num_gpu = 1  # Make sure to request this in the batch script
accelerator = "gpu"

run = "1"
datapath = "/lustre/home/pr5739/qualikiz/UKAEAGroupProject"
train_data_path = f"{datapath}/data/train_data_clipped.pkl"
val_data_path = f"{datapath}/data/valid_data_clipped.pkl"
test_data_path = f"{datapath}/data/test_data_clipped.pkl"

comet_project_name = "QLKNN-Regressor"


def main(hyper_parameters):

    # comet_logger_main = CometLogger(api_key = comet_api_key,
    #     project_name = comet_project_name,
    #     workspace = comet_workspace,
    #     save_dir = './logs',
    #     experiment_name = f'Run-{run}-main')
    save_dir = f"{path}/logs/run-{run}"
    for target in target_keys:
        print(f"Training model for {target}")
        experiment_name = f"Run-{run}-{target}-physloss_{hyper_parameters['phys_loss']}"
        keys = train_keys + [target]

        comet_logger, train_data, val_data, test_data = prepare_model(
            train_data_path,
            val_data_path,
            test_data_path,
            QLKNNDataset,
            keys,
            comet_project_name,
            experiment_name,
            hyper_parameters['phys_loss']
        )

        scaler = QLKNNDataset.scaler
        model = BaseQLKNN(n_input=15, **hyper_parameters, scaler=scaler)
        #model = BaseQLKNN(learning_rate=hyper_parameters['learning_rate'], scaler=QLKNNDataset.scaler) #QLKNNDataset.scaler persists after initial on train_data
        print(model)

        comet_logger.log_hyperparams(hyper_parameters)

        train_loader = DataLoader(
            train_data,
            batch_size=hyper_parameters["batch_size"],
            shuffle=True,
            num_workers=20,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=hyper_parameters["batch_size"],
            shuffle=False,
            num_workers=20,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=hyper_parameters["batch_size"],
            shuffle=False,
            num_workers=20,
        )

        # Create callbacks
        callback_list = callbacks(
            directory=comet_project_name,
            run=run,
            experiment_name=experiment_name,
            top_k=3,
            patience=patience,
            swa_epoch=swa_epoch,
        )

        trainer = Trainer(
            max_epochs=hyper_parameters["epochs"],
            logger=comet_logger,
            accelerator=accelerator,
            strategy=DDPPlugin(find_unused_parameters=False),
            devices=num_gpu,
            callbacks=callback_list,
            log_every_n_steps=1,
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        comet_logger.log_graph(model)

        trainer.test(dataloaders=test_loader)

        # log validation loss for each target TODO: model.metrics doesn't exist
        # comet_logger_main.log_metrics(model.metrics, step = target)


if __name__ == "__main__":
    hyperparameters = {
    "batch_size": 4096,
    "epochs": 100,
    "learning_rate": 0.002,
    "phys_loss" : False
    }
    main(hyperparameters)
    hyperparameters = {
    "batch_size": 4096,
    "epochs": 100,
    "learning_rate": 0.002,
    "phys_loss" : True
    }
    main(hyperparameters)