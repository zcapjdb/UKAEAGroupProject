import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scripts.utils import train_keys
from tqdm.auto import tqdm
import logging

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

    def shrink_perturb(self, lam, loc, scale):
        if lam != 1:
            noise_dist = torch.distributions.Normal(
                torch.Tensor([loc]), torch.Tensor([scale])
            )
            noise = noise_dist.sample()

            with torch.no_grad():
                for param in self.model.parameters():
                    param_update = (param * lam) + noise
                    param.copy_(param_update)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def train_step(self, dataloader, optimizer):
        # Initalise loss function
        BCE = nn.BCELoss()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        correct = 0
        for batch, (X, y, z, idx) in enumerate(dataloader):

            y_hat = self.forward(X.float())
            loss = BCE(y_hat, y.unsqueeze(-1).float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # calculate train accuracy
            pred_class = torch.round(y_hat.squeeze())
            correct += torch.sum(pred_class == y.float()).item()

        correct /= size
        average_loss = np.mean(losses)
        logging.debug(f"Train accuracy: {correct:>7f}, loss: {average_loss:>7f}")
        return average_loss, correct

    def validation_step(self, dataloader):
        size = len(dataloader.dataset)
        # Initalise loss function
        BCE = nn.BCELoss()

        test_loss = []
        correct = 0

        with torch.no_grad():
            for X, y, z in dataloader:
                y_hat = self.forward(X.float())
                test_loss.append(BCE(y_hat, y.unsqueeze(-1).float()).item())

                # calculate test accuracy
                pred_class = torch.round(y_hat.squeeze())
                correct += torch.sum(pred_class == y.float()).item()

        correct /= size
        average_loss = np.mean(test_loss)
        logging.debug(f"Test accuracy: {correct:>7f}, loss: {average_loss:>7f}")
        return average_loss, correct


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

    def reset_weights(self):
        self.model.apply(weight_reset)

    def shrink_perturb(model, lam, loc, scale):
        if lam != 1:
            with torch.no_grad():
                for param in model.parameters():
                    loc_tensor = loc*torch.ones_like(param)
                    scale_tensor = scale*torch.ones_like(param)
                    noise_dist = torch.distributions.Normal(loc_tensor,scale)
                    noise = noise_dist.sample()

                    param_update = (param * lam) + noise
                    param.copy_(param_update)

    def loss_function(self, y, y_hat):
        MSE_loss = nn.MSELoss(reduction="sum")
        return MSE_loss(y_hat, y.float())

    def train_step(self, dataloader, optimizer):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        for batch, (X, y, z, idx) in enumerate(dataloader):
            batch_size = len(X)
            z_hat = self.forward(X.float())
            loss = self.loss_function(z.unsqueeze(-1).float(), z_hat)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            losses.append(loss)

        return np.sum(losses) / size

    def validation_step(self, dataloader):
        size = len(dataloader.dataset)

        test_loss = []
        with torch.no_grad():
            # for X, y, z, idx in tqdm(dataloader):
            for X, y, z in tqdm(dataloader):
                z_hat = self.forward(X.float())
                test_loss.append(
                    self.loss_function(z.unsqueeze(-1).float(), z_hat).item()
                )

        return np.sum(test_loss) / size

    def predict(self, dataloader, order_outputs=None):
        pred = []
        index_ordering = []
        for (x, y, z, idx) in dataloader:
            y_hat = self.forward(x.float())
            pred.append(y_hat.squeeze().detach().numpy())
            index_ordering.append(idx.detach().numpy())

        idx_array = np.asarray(index_ordering, dtype=object).flatten()
        pred = np.asarray(pred).flatten()

        if order_outputs is not None:
            assert len(np.unique(order_outputs)) == len(order_outputs), logging.error(
                "The order_outputs array must be unique - duplicate indices found"
            )

            assert len(order_outputs) == len(idx_array), logging.error(
                "Index ordering passed is a different length to the number of predictions!"
            )
            reorder = [np.where(idx_array == i)[0] for i in order_outputs]
            reorder = np.concatenate(reorder).flatten()

            pred = pred[reorder]
            real_idx = idx_array[reorder]
            # Make sure the reording has worked
            assert real_idx.tolist() == order_outputs.tolist(), logging.error(
                "Ordering error"
            )

        return pred, idx_array


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
        self,
        df: pd.DataFrame,
        target_column: str,
        target_var: str,
        keep_index: bool = False,
    ):
        self.data = df
        self.target = target_column
        self.label = target_var

        # make sure the dataframe contains the variable information we need
        assert target_column and target_var in list(self.data.columns)

        if not keep_index:
            self.data["index"] = self.data.index

    def scale(self, scaler):
        # Scale features in the scaler object and leave the rest as is
        scaled = scaler.transform(self.data.drop([self.label, "index"], axis=1))

        cols = scaler.feature_names_in_

        temp_df = pd.DataFrame(scaled, index=self.data.index, columns=cols)

        assert set(list(temp_df.index)) == set(list(self.data.index))

        temp_df["index"] = self.data["index"]
        temp_df[self.label] = self.data[self.label]

        self.data = temp_df

        del temp_df

    def sample(self, batch_size):
        return ITGDatasetDF(
            self.data.sample(batch_size), self.target, self.label, keep_index=True
        )

    def add(self, dataset):
        # rows["index"] = np.arange(len(self.data), len(self.data) + len(rows))
        # self.data = pd.concat([self.data, rows], axis=0)
        self.data = pd.concat([self.data, dataset.data], axis=0)

    # Not sure if needed yet
    # return a copy of the dataset with only the specified indices
    # def subset(self, indices):
    #    return ITGDatasetDF(self.data.iloc[indices], self.target, self.label)

    def remove(self, indices):
        self.data.drop(
            index=indices, inplace=True
        )  # I'm not sure this does what I want
        # self.data = self.data[~self.data["index"].isin(indices)]

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):

        x = self.data[train_keys].iloc[idx].values
        y = self.data[self.label].iloc[idx]
        z = self.data[self.target].iloc[idx]
        idx = self.data["index"].iloc[idx]
        return x, y, z, idx


# General Model functions
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    weight_decay=None,
    patience=None,
    checkpoint=None,
    checkpoint_path=None,
):

    # Initialise the optimiser
    if weight_decay:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    validation_losses = []
    train_accuracy = []
    val_accuracy = []

    if not patience:
        patience = epochs

    if model.type not in ["classifier", "regressor"]:
        raise ValueError("Model type not recognised")

    for epoch in range(epochs):

        logging.debug(f"Epoch: {epoch}")

        if model.type == "classifier":
            loss, train_acc = model.train_step(train_loader, opt)
            losses.append(loss)
            train_accuracy.append(train_acc)

            val_loss, val_acc = model.validation_step(val_loader)
            validation_losses.append(val_loss)
            val_accuracy.append(val_acc)

            stopping_metric = -val_accuracy

        elif model.type == "regressor":
            logging.debug(f"Epoch: {epoch}")
            loss = model.train_step(train_loader, opt)
            losses.append(loss)

            val_loss = model.validation_step(val_loader)
            validation_losses.append(validation_losses)

            stopping_metric = val_loss

        # if validation metric is not better than the average of the last n losses then stop
        if len(stopping_metric) > patience:
            if np.mean(stopping_metric[-patience:]) < stopping_metric[-1]:
                logging.debug("Early stopping criterion reached")
                break

        if checkpoint_path:
            if checkpoint is None:
                checkpoint = epochs

            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "train_losses": losses,
                "train_acc": train_accuracy,
                "validation_losses": validation_losses,
                "val_acc": val_acc,
            }

            if epoch % checkpoint == 0:
                torch.save(state, f"{checkpoint_path}_epoch_{epoch}.pt")

    if model.type == "classifier":
        return losses, train_accuracy, validation_losses, val_accuracy

    elif model.type == "regressor":
        return losses, validation_losses


def load_model(model, save_path):
    logging.info(f"Model Loaded: {model}")
    if model == "ITG_class":
        classifier = ITG_Classifier()
        classifier.load_state_dict(torch.load(save_path))
        return classifier

    elif model == "ITG_reg":
        regressor = ITG_Regressor()
        regressor.load_state_dict(torch.load(save_path))
        return regressor
