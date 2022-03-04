import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset
from scripts.utils import ScaleData


    

    
class BaseQLKNN(pl.LightningModule):
    def __init__(
        self,
        input_dim : int = 15,
        features : list = [150,70,30], #QLKNN architecture
        num_outputs: int = 1,
        activation: str ="tanh",
        dropout_rate: float = 0.0,
        phys_loss : bool = True,
        batch_size: int = 2048,
        epochs: int = 50,
        learning_rate: float = 0.001,
        scaler : object = None
        
    ):
        super().__init__()
        self.first = nn.Linear(input_dim, features[0])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features[i], features[i+1]) for i in range(len(features)-1)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.phys_loss = phys_loss
        self.num_outputs = num_outputs
        self.lr = learning_rate
        self.scaler = scaler
        if num_outputs is not None:
            self.last = nn.Linear(features[-1], num_outputs)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "tanh": #added because QLKNN uses tanh
            self.activation = torch.tanh
        else:
            raise ValueError("That activation is unknown")

    def forward(self, x):
        x = self.first(x)
        for layer in self.linear_layers:
            x = self.dropout(self.activation(layer(x)))
        if self.num_outputs is not None:
            x = self.last(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X).squeeze()
        loss = self.loss_function 

        return loss(pred.float(), y.float())
    
    def RMSE(self,batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X).squeeze()        
        y_pred = self.inverse_transform(y_pred)
        y = self.inverse_transform(y)
        mse = F.mse_loss(y_pred, y)#.item()
        rmse = torch.sqrt(mse)
       # rrmse = torch.sqrt(mse/torch.mean(y))
        return rmse
    
    def inverse_transform(self,z):
        znumpy = z.detach().numpy()
        scaled = znumpy*self.scaler.scale_[-1]+self.scaler.mean_[-1]
        return torch.Tensor(scaled)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # tensorboard_logs = {'train_loss': loss}

        # self.log_gradients()
        # self.log_weights_and_biases()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # return {'loss': loss, 'log': tensorboard_logs}
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )
        RMSE = self.RMSE(batch, batch_idx)
        self.log(
            "val_rmse", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # tensorboad_logs = {'test_loss': loss}
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
        )
        RMSE = self.RMSE(batch, batch_idx)
        self.log(
            "test_rmse", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True
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
        if self.phys_loss:
            return c_good + lambda_stab * k_stab
        else:
            return c_good

    
    
    
    
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
            nn.ReLU(),
            nn.Linear(128, 128),
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

    # TODO: make this work, currently param.data gives an error - 'NoneType' object has no attribute 'data'
    # def log_weights_and_biases(self):
    #     for name, param in self.named_parameters():
    #         self.log(name, param.data.cpu().numpy(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

    # def log_gradients(self):
    #     for name, param in self.named_parameters():
    #         self.log(name, param.grad.data.cpu().numpy(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X).squeeze()
        loss = self.loss_function

        return loss(pred.float(), y.float())

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # tensorboard_logs = {'train_loss': loss}

        # self.log_gradients()
        # self.log_weights_and_biases()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # return {'loss': loss, 'log': tensorboard_logs}
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

    def __init__(self, file_path: str, columns: list = None, train: bool = False, phys_loss : bool = True):
        self.data = pd.read_pickle(file_path)
        if columns is not None:
            self.data = self.data[columns]
            if not phys_loss:
                self.data = self.data.query(f'{columns[-1]}>0')

        self.data = self.data.dropna()

        if train:  # ensures the class attribute is reset for every new training run
            QLKNNDataset.scaler, self.scaler = None, None  #QLKNNDataset.scaler-->self.scaler

    def scale(self, own_scaler: object = None):
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
