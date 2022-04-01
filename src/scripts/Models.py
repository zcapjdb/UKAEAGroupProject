from operator import index
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scripts.utils import train_keys
from tqdm.auto import tqdm 

# Class definitions
class ITG_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = "classifier"
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

        for batch, (X, y, idx) in enumerate(dataloader):

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
            for X, y, idx in dataloader:
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
        self.type = "regressor"
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

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    # def loss_function(self, y, y_hat):
    #     # Loss function missing regularization term (to be added using Adam optimizer)
    #     lambda_stab = 1e-3
    #     k_stab = 5
    #     if y.sum() == 0:
    #         c_good = 0
    #         c_stab = torch.mean(y_hat - k_stab)

    #     else:
    #         c_good = torch.mean(torch.square(y - y_hat))
    #         c_stab = 0
    #     return c_good + lambda_stab * k_stab
    def loss_function(self, y, y_hat):
        MSE_loss = nn.MSELoss()
        return MSE_loss(y_hat, y.float())

    def train_step(self, dataloader, optimizer):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        for batch, (X, y, idx) in enumerate(dataloader):
            batch_size = len(X)
            y_hat = self.forward(X.float())
            loss = self.loss_function(y.unsqueeze(-1).float(), y_hat) 

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

            loss = loss.item()
            losses.append(loss)

            if batch == num_batches - 1:
                print(f"loss: {loss:>7f}")

        return np.mean(losses)

    def validation_step(self, dataloader):
        test_loss = []

        with torch.no_grad():
            for X, y, idx in tqdm(dataloader):
                y_hat = self.forward(X.float())
                test_loss.append(self.loss_function(y.unsqueeze(-1).float(), y_hat).item())

        return np.mean(test_loss)
    def predict(self, dataloader):
        pred = [] 
        for (x,y,idx) in dataloader: 
            y_hat = self.forward(x.float())
            pred.append(y_hat.squeeze().detach().numpy())

        return np.array(pred)


class ITGDataset(Dataset):
    def __init__(self, X, y, z=None):
        self.X = X
        self.y = y
        self.z = z

        # add indices to all the data points
        self.indices = np.arange(len(self.X))

    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        if self.z is not None:
            return [self.X[idx], self.y[idx], self.z[idx]]
        else:
            return [self.X[idx], self.y[idx]]

    # method to add a new row to the dataset
    def add(self, x, y, z=None):
        self.X = np.append(self.X, x, axis=0)
        self.y = np.append(self.y, y, axis=0)

        if z is not None:
            self.z = np.append(self.z, z, axis=0)

        # update indices from max index
        max_index = np.max(self.indices)
        new_indices = np.arange(max_index + 1, max_index + len(x) + 1)
        self.indices = np.append(self.indices, new_indices)

    # get index of a row
    def get_index(self, idx):
        return self.indices[idx]

    # method to remove rows from the dataset using indices
    def remove(self, idx):
        # get indices of rows to be removed
        indices = self.indices[idx]

        # remove rows from dataset
        self.X = np.delete(self.X, indices, axis=0)
        self.y = np.delete(self.y, indices, axis=0)
        if self.z is not None:
            self.z = np.delete(self.z, indices, axis=0)

    # method to sample a batch of rows from the dataset - not inplace!
    # TODO: carry over indices into sample dataset
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.y), batch_size)
        if self.z is not None:
            return ITGDataset(self.X[indices], self.y[indices], self.z[indices])
        else:
            return ITGDataset(self.X[indices], self.y[indices])


class ITGDatasetDF(Dataset):
    # DataFrame version of ITGDataset
    def __init__(self, df: pd.DataFrame, target_column: str, target_var: str):
        self.data = df
        self.target = target_column
        self.label = target_var 

        # make sure the dataframe contains the variable information we need
        assert target_column and target_var in list(self.data.columns)

        self.data["index"] = self.data.index

    def scale(self, scaler):
        # Scale features in the scaler object and leave the rest as is
        scaled = scaler.transform(self.data.drop([self.label, 'index'], axis = 1))

        cols = scaler.feature_names_in_

        temp_df = pd.DataFrame(scaled, index=self.data.index,columns=cols )

        assert set(list(temp_df.index)) == set(list(self.data.index))

        temp_df['index'] = self.data['index']
        temp_df[self.label] = self.data[self.label]

        self.data = temp_df

        del temp_df

    def sample(self, batch_size):
        return ITGDatasetDF(
            self.data.sample(batch_size), self.target, self.label
        )

    def add(self, rows):
        rows["index"] = np.arange(len(self.data), len(self.data) + len(rows))
        self.data = pd.concat([self.data, rows], axis=0)

    def remove(self, indices):
        self.data.drop(index = indices, inplace = True)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):

        x = self.data[train_keys].iloc[idx].to_numpy()
        y = self.data['itg'].iloc[idx]
        idx = self.data['index'].iloc[idx]
        return x,y,idx


# General Model functions


def train_model(
    model, train_loader, val_loader, epochs, learning_rate, weight_decay=None
):

    # Initialise the optimiser
    if weight_decay:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    validation_losses = []

    if model.type == "classifer":
        val_acc = []

        for epoch in range(epochs):
            loss = model.train_step(train_loader, opt)
            losses.append(loss)

            val_loss, acc = model.validation_step(val_loader)
            validation_losses.append(validation_losses)
            val_acc.append(acc)
        return losses, validation_losses, val_acc

    elif model.type == "regressor":

        for epoch in range(epochs):
            loss = model.train_step(train_loader, opt)
            losses.append(loss)

            val_loss = model.validation_step(val_loader)
            validation_losses.append(validation_losses)

        return losses, validation_losses


def load_model(model, save_path):
    print(model)
    if model == "ITG_class":
        classifier = ITG_Classifier()
        classifier.load_state_dict(torch.load(save_path))
        return classifier

    elif model == "ITG_reg":
        regressor = ITG_Regressor()
        regressor.load_state_dict(torch.load(save_path))
        return regressor
