import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

import numpy as np

# Class definitions
class ITG_Classifier(nn.Module): 
    def __init__(self):
        super.__init__()
        self.type = 'classifier'
        self.model = self.model = nn.Sequential(
            nn.Linear(15, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def train_step(self, dataloader, optimizer):
        # Initalise loss function
        BCE = nn.BCELoss()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        for batch, (X, y) in enumerate(dataloader):

            y_hat = self.forward(X.float())
            loss = BCE(y_hat, y.unsqueeze(-1).float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch == num_batches - 1:
                loss = loss.item()
                print(f"loss: {loss:>7f}")
    
    def validation_step(self, dataloader):
        size = len(dataloader.dataset)
        # Initalise loss function
        BCE = nn.BCELoss()

        test_loss = []
        correct = 0
         
        with torch.no_grad():
            for X, y in dataloader:
                y_hat = self.forward(X.float())
                test_loss.append(BCE(y.unsqueeze(-1).float()).item(), y_hat)

                # calculate test accuracy
                pred_class = torch.round(y_hat.squeeze())
                correct += torch.sum(pred_class == y.float()).item()

        correct /= size
        return np.mean(test_loss), correct

class ITG_Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = 'regressor'
        self.model = nn.Sequential(
            nn.Linear(15, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 1),

        )

    def forward(self, x):
        y_hat = self.model(x.float())
        return y_hat
    
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
    
    def train_step(self, dataloader, optimizer):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        for batch, (X, y) in enumerate(dataloader):

            y_hat = self.forward(X.float())
            loss = self.loss_function(y.unsqueeze(-1).float(), y_hat)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if batch == num_batches - 1:
                loss = loss.item()
                print(f"loss: {loss:>7f}")

        return np.mean(losses)
    
    def validation_step(self, dataloader):
        test_loss = []
         
        with torch.no_grad():
            for X, y in dataloader:
                y_hat = self.forward(X.float())
                test_loss.append(self.loss_fn(y.unsqueeze(-1).float()).item(), y_hat)
        
        return np.mean(test_loss)

class ITGDataset(Dataset):
    def __init__(self, X, y, z = None):
        self.X = X
        self.y = y
        self.z = z

    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        if self.z is not None:
            return[self.X[idx], self.y[idx], self.z[idx]]
        else:
            return [self.X[idx], self.y[idx]]

    # add method to add a new row to the dataset
    def add(self, x, y, z = None):
        self.X = np.append(self.X, x, axis = 0)
        self.y = np.append(self.y, y, axis = 0)
        
        if z is not None:
            self.z = np.append(self.z, z, axis = 0)

# General Model functions

def train_model(model, train_loader,val_loader, epochs, learning_rate, weight_decay=None):

    # Initialise the optimiser
    if weight_decay: 
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else: 
       opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    validation_losses = []

    if model.type == 'classifer':
        val_acc= []
        
        for epoch in range(epochs):
            loss = model.train_step(train_loader, opt)
            losses.append(loss)

            val_loss, acc = model.validation_step(val_loader)
            validation_losses.append(validation_losses)
            val_acc.append(acc)
        return losses, validation_losses, val_acc
    
    elif model.type =='regressor': 

        for epoch in range(epochs):
            loss = model.train_step(train_loader, opt)
            losses.append(loss)

            val_loss = model.validation_step(val_loader)
            validation_losses.append(validation_losses)

        return losses, validation_losses







