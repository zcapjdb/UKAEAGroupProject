#!/usr/bin/env python
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

import sys
import pickle
from utils import train_keys, target_keys, prepare_model, callbacks
from Classifier import Classifier, ClassifierDataset
from scripts.utils import train_keys, target_keys, prepare_model
from temperature_scaling import ModelWithTemperature

num_gpu = 3  # Make sure to request this in the batch script
accelerator = "gpu"

run = "1"

train_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
val_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
test_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/test_data_clipped.pkl"

parameters = {"nodes": [128, 256, 512], "layers": [3, 4, 5]}

hyper_parameters = {
    "batch_size": 4096,
    "epochs": 1,
    "learning_rate": 0.002,
}


def grid_search(parameters, train_loader, val_loader, test_loader, inshape=15):
    """
    Inputs:
        build_fn: a function that will be used to build the neural network
        parameters: a dictionary of model parameters
        train_data:
        val_data
    """

    results_dict = {}

    counter = 0

    best_val_loss = sys.float_info.max

    for i in parameters["layers"]:

        # List of possible node combinations
        n = i

        combs = [[nodesize] * i for nodesize in parameters["nodes"]]

        for node in combs:

            # build model
            model = Classifier()
            early_stopping = EarlyStopping(
                monitor="val_loss", min_delta=0.0, patience=10
            )
            progress_bar = TQDMProgressBar(refresh_rate=250)

            model.build_classifier(i, node, inshape)
            print(model)

            trainer = Trainer(
                max_epochs=hyper_parameters["epochs"],
                accelerator=accelerator,
                strategy=DDPPlugin(find_unused_parameters=False),
                devices=num_gpu,
                callbacks=[early_stopping, progress_bar],
                log_every_n_steps=250,
                precision=32,
                amp_backend="native",
            )

            trainer.fit(model, train_loader, val_loader)
            result_1 = trainer.test(dataloaders=test_loader)
            print(result_1)

            scaled_model = ModelWithTemperature(model)
            scaled_model.set_temperature(val_loader)

            trainer_test = Trainer(precision=32, amp_backend="native")
            result = trainer_test.test(model=scaled_model, dataloaders=test_loader)
            result = trainer.test(dataloaders=test_loader)
            print(result)

            trial_dict = {
                "layers": i,
                "nodes": node,
                "perfomance": result,
            }

            results_dict["trial_" + str(counter)] = trial_dict

            # file_name = f"/home/tmadula/grid_search/trial_test{str(counter)}.pkl"
            file_name = f"/share/rcifdata/jbarr/UKAEAGroupProject/grid_search_torch/trial_test{str(counter)}.pkl"
            with open(file_name, "wb") as file:
                pickle.dump(trial_dict, file)

            counter += 1
    return results_dict


def main():

    keys = train_keys + ["target"]
    train_data, val_data, test_data = prepare_model(
        train_data_path,
        val_data_path,
        test_data_path,
        ClassifierDataset,
        keys,
        categorical_keys=["target"],
    )

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

    grid_dict = grid_search(parameters, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
