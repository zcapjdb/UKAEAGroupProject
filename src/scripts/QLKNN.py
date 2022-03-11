import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from scripts.utils import ScaleData


class QLKNN(pl.LightningModule):
    """
    Class that implements QLKNN model as defined in the paper:
    Fast modeling of turbulent transport in fusion plasmas using neural networks
    """

    def __init__(
        self,
        n_input: int = 15,
        batch_size: int = 2048,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.lr = learning_rate

    def forward(self, x):
        X = self.model(x.float())
        return X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X).squeeze()
        loss = self.loss_function

        return loss(pred.float(), y.float())

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # tensorboad_logs = {'test_loss': loss}
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )

    def loss_function(self, y, y_hat):
        # Loss function missing regularization term (to be added using Adam optimizer)
        lambda_stab = 1e-3
        k_stab = 5
        if y.sum() == 0:
            c_good = 0
            c_stab = torch.mean(y_hat - k_stab)

        else:
            c_good = torch.mean(torch.square(y - y_hat))
            c_stab = 0
        return c_good + lambda_stab * k_stab

class QLKNN_Big(pl.LightningModule):
    """
    Class that implements QLKNN model as defined in the paper:
    Fast modeling of turbulent transport in fusion plasmas using neural networks
    """

    def __init__(
        self,
        n_input: int = 15,
        batch_size: int = 2048,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.lr = learning_rate

    def forward(self, x):
        X = self.model(x.float())
        return X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X).squeeze()
        loss = self.loss_function

        return loss(pred.float(), y.float())

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # tensorboad_logs = {'test_loss': loss}
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )

    def loss_function(self, y, y_hat):
        # Loss function missing regularization term (to be added using Adam optimizer)
        lambda_stab = 1e-3
        k_stab = 5
        if y.sum() == 0:
            c_good = 0
            c_stab = torch.mean(y_hat - k_stab)

        else:
            c_good = torch.mean(torch.square(y - y_hat))
            c_stab = 0
        return c_good + lambda_stab * k_stab
        
class QLKNNDataset(Dataset):
    """
    Class that implements a PyTorch Dataset object for the QLKNN model
    """

    scaler = None  # create scaler class instance

    def __init__(self, file_path: str, columns: list = None, train: bool = False):
        self.data = pd.read_pickle(file_path)

        if columns is not None:
            self.data = self.data[columns]

        self.data = self.data.dropna()

        if train:  # ensures the class attribute is reset for every new training run
            QLKNNDataset.scaler, self.scaler = None, None

    def scale(self, own_scaler: object = None, categorical_keys: list = None):
        if own_scaler is not None:
            self.data = ScaleData(self.data, own_scaler)

        self.data, QLKNNDataset.scaler = ScaleData(self.data, self.scaler)

    def __len__(self):
        # data is numpy array
        return len(self.data.index)

    def __getitem__(self, idx):
        X = self.data.iloc[idx, :-1].to_numpy()
        y = self.data.iloc[idx, -1]
        return X.astype(float), y.astype(float)


if __name__ == "__main__":
    pass
