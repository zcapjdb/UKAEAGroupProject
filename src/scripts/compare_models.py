import scripts.Models as models
import scripts.pipeline_tools as tools
import scripts.utils as utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging, verboselogs, coloredlogs

level = "DEBUG"
# Create logger object for use in pipeline
verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level=level)

training_sample_sizes = [1_000, 5_000, 10_000, 15_000, 20_000, 25_000]

for size in training_sample_sizes:
    print(f"Training on {size} samples")
    train_path = f"/unix/atlastracking/jbarr/sampled_data_{size}.pkl"
    valid_path = "/unix/atlastracking/jbarr/valid_data_clipped.pkl"

    print("Loading data")
    train, val = tools.prepare_data(
        train_path,
        valid_path,
        target_column="efiitg_gb",
        target_var="itg",
        valid_size=1_000_000,
    )

    scaler = StandardScaler()
    scaler.fit_transform(train.drop(["itg"], axis=1))

    train_dataset = models.ITGDatasetDF(train, target_column="efiitg_gb", target_var="itg")
    valid_dataset = models.ITGDatasetDF(val, target_column="efiitg_gb", target_var="itg")

    train_dataset.scale(scaler)
    valid_dataset.scale(scaler)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)

    x_array = valid_dataset.data[utils.train_keys].values
    y_array = valid_dataset.data["itg"].values
    z_array = valid_dataset.data["efiitg_gb"].values
    dataset_numpy = models.ITGDataset(x_array, y_array, z_array)

    valid_loader = DataLoader(
        dataset_numpy, batch_size=int(0.1 * len(y_array)), shuffle=True
    )


    classifier = models.ITG_Classifier()

    train_loss, train_acc, val_loss, val_acc = models.train_model(classifier, 
        train_loader, 
        valid_loader, 
        epochs=100,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=50,
        checkpoint_path=f"./checkpoints/size_{size}",
        checkpoint = 10
        )
    
    # Plot the training loss and validation loss
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.savefig(f"./checkpoints/{size}_loss.png")
    plt.clf()

    # Plot the training accuracy and validation accuracy
    plt.figure()
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.savefig(f"./checkpoints/{size}_acc.png")
    plt.clf()
