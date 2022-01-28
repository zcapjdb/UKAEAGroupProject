import numpy as np 
import pandas as pd 
import torch.nn as nn

import pytorch_lightning as pl


class Encoder(nn.Model): 
    def __init__(self, latent_dims: int = 3, n_input: int = 15):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLu(),
            nn.Linear(20, 10),
            nn.ReLu(), 
            nn.Linear(10,5), 
            nn.ReLu(),
            nn.Linear(5, latent_dims)
        )
    
    def forward(self, x): 
        encoded = self.encoder(x)

        return encoded

class Decoder(nn.Model):
        def __init__(self, latent_dims: int = 3, n_output: int = 15):
            super().__init__()

            self.decoder = nn.Sequential(
                nn.Linear(latent_dims,5),
                nn.ReLu(), 
                nn.Linear(5, 10), 
                nn.ReLu(), 
                nn.Linear(10, 20), 
                nn.ReLu(), 
                nn.Linear(20, n_output)
            )

        def forward(self, encoded):
            decoded = self.decoder(encoded)

            return decoded



    

class AutoEncoder(pl.LightningModule):

    def __init__(
        self,
        latent_dims: int = 3,
        encoder: nn.Model = Encoder,
        decoder: nn.Model = Decoder,
        n_input: int = 15,
        n_output: int = 15,
        batch_size: int,
        epochs: int,
        learning_rate: float
        ):

        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


    def configure_optimizers(self, lr = 0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        return optimizer

    def step(self, batch, batch_idx):
        X = batch
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)

        MSE_loss = nn.MSELoss()
        loss = MSE_loss(decoded, X)

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



class AE_Dataset(Dataset):
    def __init__(self, file_path: str, columns = None, scale: object = None):
        self.data = pd.read_pickle(file_path)

        if columns is not None:
            self.data = self.data[columns]

        if scale is None:
            self.data, self.scaler = ScaleData(self.data)

        if scale is not None:
            self.data = ScaleData(self.data, scale)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        # no y label for auto encoder
        X = self.data.iloc[idx, :].to_numpy() 
        return X.astype(float)