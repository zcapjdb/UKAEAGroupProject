import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch
import pytorch_lightning as pl

from scripts.utils import ScaleData

class Classifier(pl.LightningModule): 
    def __init__(self): 
    
    super().__init__()
    self.model = nn.Sequential()

    def build_classifier(self, n_layers, nodes, inshape):
        layers = [] 
        for i in range(n_layers):
            if i == 0:  
                layers.append(nn.Linear(inshape, nodes[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else: 
                layers.append(nn.Linear(nodes[i-1], nodes[i]))
                layers.append(nn.ReLU())
                layer.append(nn.Dropout(0.1))

        layers.append(nn.Linear(nodes[-1], 1))
        layers.append(nn.sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self):
        output = self.model(x)

        return output
    def configure_optimizers(self, lr = 0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        return optimizer
    
    def step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X).squeeze()

        BC_loss = nn.CrossEntropyLoss()
        loss = BC_loss(y_hat, y)

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

class ClassifierDataset(Dataset):
    """
    Class that implements a PyTorch Dataset object for the autoencoder
    """
    scaler = None # create scaler class instance

    def __init__(self, file_path: str, xcolumns = None,ycolumns = None,  train: bool = False):
        self.xdata = pd.read_pickle(file_path) 

        if xcolumns and ycolumns is not None:
            self.ydata = self.xdata[ycolumns]
            self.xdata = self.xdata[xcolumns]
        
        if train: # ensures the class attribute is reset for every new training run
            ClassifierDataset.scaler, self.scaler = None, None

    def scale(self, own_scaler: object = None):
        if own_scaler is not None:
            self.xdata = ScaleData(self.xdata, own_scaler)

        self.xdata, ClassifierDataset.scaler = ScaleData(self.xdata, self.scaler)

    def __len__(self):
        return len(self.xdata.index)

    def __getitem__(self, idx):
        # no y label for auto encoder
        X = self.xdata.[xcolumns].to_numpy()

        if ycolumns is not None:  
            Y = self.ydata.[ycolumns].to_numpy()

            return X.astype(float), Y.astype(float)
        else: 
            return X.astype(float)
