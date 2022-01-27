<<<<<<< HEAD
=======
import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch
<<<<<<< HEAD
import pytorch_lightning as pl

from torch.utils.data import Dataset


class QLKNN(pl.LightningModule):
    def __init__(self, n_input, batch_size, epochs, learning_rate):
        super().__init__()
        self.model = nn.Sequential(
           nn.Linear(n_input,128), 
           nn.ReLU(),
           nn.Linear(128,128),
           nn.ReLU(),
           nn.Linear(128,1)
        )
        self.lr = learning_rate
    
    def forward(self, x):
        X = self.model(x.float())
        return X

    def configure_optimizers(self, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = 1e-5)
        return optimizer

    
    def step(self, batch, batch_idx):
        X, y = batch
        pred = self.forward(X).squeeze()

        loss = self.loss_function
        #loss = nn.MSELoss()
        return loss(pred.float(), y.float())
        #return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        #tensorboard_logs = {'train_loss': loss}
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #return {'loss': loss, 'log': tensorboard_logs}
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger = True)


    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        #tensorboad_logs = {'test_loss': loss}
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, logger = True)

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
        return c_good + lambda_stab*k_stab


class QLKNN_Dataset(Dataset):
    def __init__(self, file_path, columns = None, train = True):
        self.data = pd.read_pickle(file_path)

        if columns is not None:
            self.data = self.data[columns]

        self.data = self.data.dropna()
        # convert all columns to float
        self.data = self.data.astype(float)

    def __len__(self):
        # data is numpy array
        return len(self.data.index)

    def __getitem__(self, idx):
        X = self.data.iloc[idx, :-1].to_numpy()
        y = self.data.iloc[idx, -1]
        return X.astype(float), y.astype(float)


train_keys = ['Ane', 'Ate', 'Autor', 'Machtor', 'x', 'Zeff', 'gammaE', 
              'q', 'smag', 'alpha', 'Ani1', 'Ati0', 'normni1', 'Ti_Te0', 'logNustar']

target_keys = ['dfeitg_gb_div_efiitg_gb', 'dfetem_gb_div_efetem_gb',
       'dfiitg_gb_div_efiitg_gb', 'dfitem_gb_div_efetem_gb', 'efeetg_gb',
       'efeitg_gb_div_efiitg_gb', 'efetem_gb', 'efiitg_gb',
       'efitem_gb_div_efetem_gb', 'pfeitg_gb_div_efiitg_gb',
       'pfetem_gb_div_efetem_gb', 'pfiitg_gb_div_efiitg_gb',
       'pfitem_gb_div_efetem_gb', 'vceitg_gb_div_efiitg_gb',
       'vcetem_gb_div_efetem_gb', 'vciitg_gb_div_efiitg_gb',
       'vcitem_gb_div_efetem_gb', 'vfiitg_gb_div_efiitg_gb',
       'vfitem_gb_div_efetem_gb', 'vriitg_gb_div_efiitg_gb',
       'vritem_gb_div_efetem_gb', 'vteitg_gb_div_efiitg_gb',
        'vtiitg_gb_div_efiitg_gb',]
=======

