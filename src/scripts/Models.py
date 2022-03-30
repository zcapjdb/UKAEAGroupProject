import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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
    def __init__(
        self, df: pd.DataFrame, train_columns: list = None, target_columns: list = None
    ):
        self.data = df
        self.train_columns = train_columns
        self.target_columns = target_columns

        self.columns = self.train_columns + self.target_columns
        if self.columns is not None:
            self.data = self.data[self.columns]

        try:
            self.data["itg"] = np.where(train_data["efiitg_gb"] != 0, 1, 0)
        except:
            raise ValueError("train_data does not contain efiitg_gb column")

        self.data = self.data.dropna()

        self.data["index"] = np.arange(len(self.data))

    def scale(self, scaler):
        # Scale features in the scaler object and leave the rest as is
        scaled_features = scaler.feature_names_in_

        column_transformer = ColumnTransformer(
            [("scaler", scaler, scaled_features)], remainder="passthrough"
        )

        self.data = column_transformer.fit_transform(self.data)

    def sample(self, batch_size):
        return ITGDataset(
            self.data.sample(batch_size), self.train_columns, self.target_columns
        )

    def add(self, rows):
        rows["index"] = np.arange(len(self.data), len(self.data) + len(rows))
        self.data = pd.concat([self.data, rows], axis=0)

    def remove(self, indices):
        self.data = self.data[~self.data["index"].isin(indices)]

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


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
