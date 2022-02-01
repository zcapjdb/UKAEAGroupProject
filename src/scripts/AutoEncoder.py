import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from utils import ScaleData

class Encoder(nn.Module): 
    def __init__(self, latent_dims: int = 3, n_input: int = 15):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 10),
            nn.ReLU(), 
            nn.Linear(10,5), 
            nn.ReLU(),
            nn.Linear(5, latent_dims)
        )
    
    def forward(self, x): 
        encoded = self.encoder(x.float())

        return encoded

class Decoder(nn.Module):
        def __init__(self, latent_dims: int = 3, n_output: int = 15):
            super().__init__()

            self.decoder = nn.Sequential(
                nn.Linear(latent_dims,5),
                nn.ReLU(), 
                nn.Linear(5, 10), 
                nn.ReLU(), 
                nn.Linear(10, n_output)
            )

        def forward(self, encoded):
            decoded = self.decoder(encoded.float())

            return decoded


class AutoEncoder(pl.LightningModule):

    def __init__(
        self,
        encoder: nn.Module = Encoder,
        decoder: nn.Module = Decoder,
        latent_dims: int = 3,
        n_input: int = 15,
        batch_size: int = 2048,
        epochs: int = 100,
        learning_rate: float = 0.001,
        ):

        super().__init__()
        self.encoder = encoder(latent_dims, n_input)
        self.decoder = decoder(latent_dims, n_input)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def configure_optimizers(self, lr = 0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        return optimizer

    def step(self, batch, batch_idx):
        X = batch
        pred = self.forward(X).squeeze()

        MSE_loss = nn.MSELoss()
        loss = MSE_loss(X.float(), pred.float())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)



class AutoEncoderDataset(Dataset):
    """
    Class that implements a PyTorch Dataset object for the autoencoder
    """
    scaler = None # create scaler class instance

    def __init__(self, file_path: str, columns = None, train: bool = False):
        self.data = pd.read_pickle(file_path)

        if columns is not None:
            self.data = self.data[columns]
        
        if train: # ensures the class attribute is reset for every new training run
            AutoEncoderDataset.scaler, self.scaler = None, None

    def scale(self, own_scaler: object = None):
        if own_scaler is not None:
            self.data = ScaleData(self.data, own_scaler)

        self.data, AutoEncoderDataset.scaler = ScaleData(self.data, self.scaler)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        # no y label for auto encoder
        X = self.data.iloc[idx, :].to_numpy() 
        return X.astype(float)