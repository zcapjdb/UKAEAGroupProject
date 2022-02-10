import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch
import copy
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torch.utils.data import Dataset
from scripts.utils import ScaleData

class Encoder(nn.Module): 
    def __init__(self, latent_dims: int = 3, n_input: int = 15):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 12),
            nn.ReLU(), 
            nn.Linear(12, 8), 
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, latent_dims)
        )
    
    def forward(self, x): 
        encoded = self.encoder(x.float())

        return encoded

class Decoder(nn.Module):
        def __init__(self, latent_dims: int = 3, n_output: int = 15):
            super().__init__()

            self.decoder = nn.Sequential(
                nn.Linear(latent_dims, 4),
                nn.ReLU(), 
                nn.Linear(4, 8), 
                nn.ReLU(), 
                nn.Linear(8, 12),
                nn.ReLU(),
                nn.Linear(12, n_output)
            )

        def forward(self, encoded):
            decoded = self.decoder(encoded.float())

            return decoded


class AutoEncoder(LightningModule):

    def __init__(
        self,
        encoder: nn.Module = Encoder,
        decoder: nn.Module = Decoder,
        latent_dims: int = 3,
        n_input: int = 15,
        batch_size: int = 2048,
        epochs: int = 100,
        learning_rate: float = 0.0025,
        ):

        super().__init__()
        self.encoder = encoder(latent_dims, n_input)
        self.decoder = decoder(latent_dims, n_input)

        self.latent_dims = latent_dims
        self.n_input = n_input
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 1e-4)
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



class LatentSpace(Callback):

    # callback to plot a scatter plot of the latent space every n epochs
    def __init__(self, n_epochs: int = 10, n_samples: int = 1000, n_latent: int = 3, n_input: int = 15):
        super().__init__()
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.n_latent = n_latent
        self.n_input = n_input

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.n_epochs == 0:
            latent_space = self.get_latent_space(pl_module)
            latent_space_plot = self.plot_latent_space(latent_space)

            trainer.logger.experiment.log_figure(figure_name = f"latent_space_{trainer.current_epoch}", figure = latent_space_plot)
    
    # Only use this with at most 1 GPU for now
    def get_latent_space(self, pl_module: LightningModule) -> np.ndarray:

        # create temp model with pl_module weights (don't have to worry about changing devices)
        encoder_state = copy.deepcopy(pl_module.encoder.state_dict())
        temp_model = Encoder(latent_dims = self.n_latent, n_input = self.n_input)
        temp_model.load_state_dict(encoder_state)
        temp_model.to("cpu")

        samples = torch.randn(self.n_samples, self.n_input).to("cpu")
        latent_space = temp_model(samples).detach().numpy()
        return latent_space

    def plot_latent_space(self, latent_space: np.ndarray):
        if self.n_latent == 2:
            fig, ax = plt.subplots(figsize = (10, 10))
            ax.scatter(latent_space[:, 0], latent_space[:, 1])

            return fig

        elif self.n_latent == 3:
            # 2D scatter where each point is colored by the third dimension
            fig = plt.figure(figsize = (10, 10))
            sc = plt.scatter(latent_space[:, 0], latent_space[:, 1], c = latent_space[:, 2])
            plt.colorbar(sc)

            return fig

        else:
            raise ValueError("n_latent must be either 2 or 3")



class LatentTrajectory(Callback):
    # callback to recored trajectories of test points in the latent space
    # when training is complete plots the trajectories

    def __init__(self, n_epochs: int = 10, n_samples: int = 10, n_latent: int = 3, n_input: int = 15):
        super().__init__()
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.n_latent = n_latent
        self.n_input = n_input
        self.test_points = None
        self.trajectories = {}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == 0:
            self.test_points = torch.randn(self.n_samples, self.n_input)

        if trainer.current_epoch % self.n_epochs == 0:
            latent_space = self.get_latent_space(pl_module, self.test_points)
            # store trajectories in a dictionary
            self.trajectories[f"epoch_{trainer.current_epoch}"] = latent_space

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # plot the trajectories
        latent_space_plot = self.plot_trajectories()

        trainer.logger.experiment.log_figure(figure_name = f"latent_trajectory", figure = latent_space_plot)

    def get_latent_space(self, pl_module: LightningModule, test_points: torch.Tensor) -> np.ndarray:
        # create temp model with pl_module weights (don't have to worry about changing devices)
        encoder_state = copy.deepcopy(pl_module.encoder.state_dict())
        temp_model = Encoder(latent_dims = self.n_latent, n_input = self.n_input)
        temp_model.load_state_dict(encoder_state)
        temp_model.to("cpu")

        test_points.to("cpu")
        latent_space = temp_model(test_points).detach().numpy()
        return latent_space

    def plot_trajectories(self):
        if self.n_latent == 2:
            fig, ax = plt.subplots(figsize = (10, 10))
            # plot trajectories with increasing opacity
            # loop through the trajectories
            for i, key in enumerate(self.trajectories.keys()):
                # get the trajectory
                trajectory = self.trajectories[key]
                # plot the trajectory
                ax.plot(trajectory[:, 0], trajectory[:, 1], alpha = i / len(self.trajectories))

            return fig
        
        elif self.n_latent == 3:
            # 3D scatter plot where each point is colored by the third dimension
            fig = plt.figure(figsize = (10, 10))
            for i, key in enumerate(self.trajectories.keys()):
                trajectory = self.trajectories[key]
                sc = plt.scatter(trajectory[:, 0], trajectory[:, 1],
                c = trajectory[:, 2], alpha = i/len(self.trajectories))
            
            #plt.colorbar(sc)

            return fig

        else:
            raise ValueError("n_latent must be either 2 or 3")