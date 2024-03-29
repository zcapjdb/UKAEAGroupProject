import comet_ml
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

from scripts.AutoEncoder import VariationalAutoencoder, AutoEncoderDataset
from scripts.utils import train_keys, target_keys, prepare_model, callbacks

hyper_parameters = {
    "batch_size": 1024,
    "epochs": 250,
    "learning_rate": 1e-3,
    "latent_dims": 5,
}

patience = 25
# swa_epoch = 100

num_gpu = 1  # Make sure to request this in the batch script
accelerator = "gpu"

run = "0"

train_data_path = "/home/tmadula/data/UKAEA/train_data_clipped.pkl"
val_data_path = "/home/tmadula/data/UKAEA/valid_data_clipped.pkl"
test_data_path = "/home/tmadula/data/UKAEA/test_data_clipped.pkl"

comet_project_name = "vae"


def main():
    # Load data and create logger for training
    experiment_name = f"Run-{run}"
    keys = train_keys
    comet_logger, train_data, val_data, test_data = prepare_model(
        train_data_path,
        val_data_path,
        test_data_path,
        AutoEncoderDataset,
        keys,
        comet_project_name,
        experiment_name,
    )

    # Create model
    model = VariationalAutoencoder(
        latent_dims=hyper_parameters["latent_dims"],
        learning_rate=hyper_parameters["learning_rate"],
    )
    print(model)

    # Log hyperparameters
    comet_logger.log_hyperparams(hyper_parameters)

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=10,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=10,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=10,
    )

    # Create callbacks
    callback_list = callbacks(
        directory=comet_project_name,
        run=run,
        experiment_name=experiment_name,
        top_k=3,
        patience=patience,
        # swa_epoch=swa_epoch,
    )

    # TODO: make this only call in a debug mode
    # callback_list.append(ModuleDataMonitor(submodules = True, log_every_n_steps = 2500))

    # if hyper_parameters["latent_dims"] == 2 or hyper_parameters["latent_dims"] == 3:
    #     callback_list.append(LatentSpace())
    #     callback_list.append(LatentTrajectory())

    # Create trainer
    trainer = Trainer(
        max_epochs=hyper_parameters["epochs"],
        logger=comet_logger,
        accelerator=accelerator,
        strategy=DDPPlugin(find_unused_parameters=False),
        devices=num_gpu,
        callbacks=callback_list,
        log_every_n_steps=100,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate model
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()
