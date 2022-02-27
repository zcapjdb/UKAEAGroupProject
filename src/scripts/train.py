import comet_ml
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

from QLKNN import QLKNN, QLKNNDataset
from scripts.utils import train_keys, target_keys, prepare_model, callbacks

hyper_parameters = {
    "batch_size": 2048,
    "epochs": 150,
    "learning_rate": 0.001,
}

patience = 10
swa_epoch = 100

num_gpu = 4  # Make sure to request this in the batch script
accelerator = "gpu"

run = "7"

train_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
val_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
test_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/test_data_clipped.pkl"

comet_project_name = "QLKNN-Regressor"


def main():

    save_dir = f"/share/rcifdata/jbarr/UKAEAGroupProject/logs/run-{run}"
    for target in target_keys:
        print(f"Training model for {target}")
        experiment_name = f"Run-{run}-{target}"
        keys = train_keys + [target]

        comet_logger, train_data, val_data, test_data = prepare_model(
            train_data_path,
            val_data_path,
            test_data_path,
            QLKNNDataset,
            keys,
            comet_project_name,
            experiment_name,
        )

        model = QLKNN(n_input=15, **hyper_parameters)
        print(model)

        comet_logger.log_hyperparams(hyper_parameters)

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
            swa_epoch=swa_epoch,
        )

        trainer = Trainer(
            max_epochs=hyper_parameters["epochs"],
            logger=comet_logger,
            accelerator=accelerator,
            strategy=DDPPlugin(find_unused_parameters=False),
            devices=num_gpu,
            callbacks=callback_list,
            log_every_n_steps=250,
            benchmark=True,
            check_val_every_n_epoch=5
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        comet_logger.log_graph(model)

        trainer.test(dataloaders=test_loader)

        # log validation loss for each target TODO: model.metrics doesn't exist
        # comet_logger_main.log_metrics(model.metrics, step = target)


if __name__ == "__main__":
    main()
