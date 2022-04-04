#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torchmetrics
from utils import ScaleData


class Classifier(pl.LightningModule):
    def __init__(self,n_layers, nodes, inshape, multiclass=False, outshape=1, weights=None):

        super().__init__()

        self.save_hyperparameters()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(inshape, nodes[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else:
                layers.append(nn.Linear(nodes[i - 1], nodes[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(nodes[-1], outshape))
        self.multiclass = multiclass
        self.weights = weights
        if self.multiclass:
            # don't use probabilities, cross_entropy takes care of that
            self.loss = F.cross_entropy
        else:
            # use probabilities
            layers.append(nn.Sigmoid())
            self.loss = F.binary_cross_entropy
            
        self.model = nn.Sequential(*layers)
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
    def forward(self, X):
        output = self.model(X.float())

        return output

    def configure_optimizers(self, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        return optimizer

    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X.float()).squeeze()
        return y,pred

    def training_step(self, batch, batch_idx):
        y,pred = self.step(batch, batch_idx)
        #if not self.multiclass:
        #    y = y.int()
        #else:
        #    pred = pred.float()
        pred = pred.float()
        if not self.multiclass:
            loss = self.loss(pred, y.float(), self.weights)
        else:
            loss = self.loss(pred, y.long(), self.weights)
        self.train_accuracy(pred,y.int())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )        
        return {"loss": loss}

    def train_epoch_end(self, outs):
        self.log(
            "train_acc", self.train_accuracy.compute())
    
    def validation_step(self, batch, batch_idx):
        y,pred = self.step(batch, batch_idx)
        pred = pred.float()
        if not self.multiclass:
            loss = self.loss(pred, y.float(), self.weights)
        else:
            loss = self.loss(pred, y.long(), self.weights)    
        self.val_accuracy(pred,y.int())
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return

    def validation_epoch_end(self,outs):
        self.log(
            "val_acc", self.val_accuracy.compute())
            

    def test_step(self, batch, batch_idx):
        y,pred = self.step(batch, batch_idx)
        loss = self.loss(pred, y.float())
        self.test_accuracy(pred,y.int())
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        
    def test_epoch_end(self,outs):
        self.log(
            "test_acc", self.test_accuracy.compute())

class ClassifierDataset(Dataset):
    """
    Class that implements a PyTorch Dataset object for the classifier.
    """

    scaler = None  # create scaler class instance

    def __init__(self, file_path: str, columns=None, train: bool = False):
        self.data = pd.read_pickle(file_path)

        if columns is not None:
            self.data = self.data[columns]

        self.data = self.data.dropna()

        if train:  # ensures the class attribute is reset for every new training run
            ClassifierDataset.scaler, self.scaler = None, None

    def scale(self, own_scaler: object = None, categorical_keys: list = None):
        if own_scaler is not None:
            self.data = ScaleData(self.data, own_scaler, categorical_keys)

        self.data, ClassifierDataset.scaler = ScaleData(
            self.data, self.scaler, categorical_keys
        )

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        X = self.data.iloc[idx, :-1].to_numpy()
        y = self.data.iloc[idx, -1]
        return X.astype(float), y.astype(float)
