import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pipeline.pipeline_tools as pt

import numpy as np
from sklearn.preprocessing import StandardScaler
from scripts.utils import train_keys
from tqdm.auto import tqdm
import logging
import time
from tqdm.auto import tqdm

cuda0 = torch.device("cuda:0")

# Class definitions
class Classifier(nn.Module):
    def __init__(self, device, dropout=0.1):
        super().__init__()
        self.type = "classifier"
        self.device = device
        self.dropout = dropout
        self.model = self.model = nn.Sequential(
            nn.Linear(15, 512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def shrink_perturb(self, lam, loc, scale):
        if lam != 1:
            with torch.no_grad():
                for param in self.model.parameters():
                    loc_tensor = loc * torch.ones_like(param)
                    scale_tensor = scale * torch.ones_like(param)
                    noise_dist = torch.distributions.Normal(loc_tensor, scale)
                    noise = noise_dist.sample()

                    param_update = (param * lam) + noise
                    param.copy_(param_update)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def train_step(self, dataloader, optimizer, epoch=None, disable_tqdm=False):
        # Initalise loss function
        BCE = nn.BCELoss()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        correct = 0
        for batch, (X, y, z, idx) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}", disable=disable_tqdm)
        ):

            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(X.float())
            loss = BCE(y_hat, y.unsqueeze(-1).float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # calculate train accuracy
            pred_class = torch.round(y_hat.squeeze())  # torch.round(y_hat.squeeze())
            correct += torch.sum(
                pred_class == y.float()
            ).item()  # torch.sum(pred_class == y.float()).item()

        correct /= size
        average_loss = np.mean(losses)
        logging.debug(f"Train accuracy: {correct:>7f}, loss: {average_loss:>7f}")
        return average_loss, correct

    def validation_step(self, dataloader, scheduler=None):
        size = len(dataloader.dataset)
        # Initalise loss function
        BCE = nn.BCELoss()

        test_loss = []
        correct = 0

        with torch.no_grad():
            for X, y, z, _ in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.forward(X.float())
                test_loss.append(BCE(y_hat, y.unsqueeze(-1).float()).item())

                # calculate test accuracy
                pred_class = torch.round(
                    y_hat.squeeze()
                )  # torch.round(y_hat.squeeze())
                correct += torch.sum(
                    pred_class == y.float()
                ).item()  # torch.sum(pred_class == y.float()).item()

        correct /= size
        average_loss = np.mean(test_loss)
        logging.debug(f"Test accuracy: {correct:>7f}, loss: {average_loss:>7f}")

        if scheduler is not None:
            scheduler.step(average_loss)

        return average_loss, correct

    def predict(self, dataloader):

        if not isinstance(dataloader, DataLoader):
            # dataloader = DataLoader(dataloader, batch_size=100,shuffle=False) # --- batch size doesnt matter here because it's just prediction
            dataloader = pt.pandas_to_numpy_data(dataloader, batch_size=100)

        size = len(dataloader.dataset)
        pred = []
        losses = []
        correct = 0

        BCE = nn.BCELoss()

        for batch, (x, y, z, idx) in enumerate(tqdm(dataloader)):
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(x.float())

            pred.append(y_hat.squeeze().detach().cpu().numpy())
            loss = BCE(y_hat, y.unsqueeze(-1).float()).item()
            losses.append(loss)

            # calculate test accuracy
            pred_class = torch.round(y_hat.squeeze())  # torch.round(y_hat.squeeze())
            correct += torch.sum(
                pred_class == y.float()
            ).item()  # torch.sum(pred_class == y.float()).item()

        average_loss = np.sum(losses) / size

        correct /= size

        pred = np.asarray(pred, dtype=object).flatten()

        return pred, [average_loss, correct]


class Regressor(nn.Module):
    def __init__(self, device, scaler, flux, dropout=0.1):
        super().__init__()
        self.type = "regressor"
        self.device = device
        self.scaler = scaler
        self.loss = nn.MSELoss(
            reduction="sum"
        )  # LZ: ToDo this might be an input in the case the output is multitask
        self.flux = flux
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(15, 512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

    def forward(self, x):
        y_hat = self.model(x.float())
        return y_hat

    def unscale(self, y):
        # get the index of the scaler that corresponds to the target
        scaler_features = self.scaler.feature_names_in_
        scaler_index = np.where(scaler_features == self.flux)[0][0]

        return y * self.scaler.scale_[scaler_index] + self.scaler.mean_[scaler_index]

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def reset_weights(self):
        self.model.apply(weight_reset)

    def shrink_perturb(self, lam, loc, scale):
        if lam != 1:
            with torch.no_grad():
                for param in self.model.parameters():
                    loc_tensor = loc * torch.ones_like(param)
                    scale_tensor = scale * torch.ones_like(param)
                    noise_dist = torch.distributions.Normal(loc_tensor, scale)
                    noise = noise_dist.sample()

                    param_update = (param * lam) + noise
                    param.copy_(param_update)

    def loss_function(
        self, y, y_hat, unscale=False
    ):  # LZ : ToDo if given mode is predicted not to develop, set the outputs related to that mode to zero, and should not contribute to the loss

        loss = self.loss(y_hat, y.float())
        if unscale:
            y_hat = torch.Tensor(self.unscale(y_hat.detach().cpu().numpy()))
            y = torch.Tensor(self.unscale(y.detach().cpu().numpy()))
            loss_unscaled = self.loss(y_hat, y.float())
            return loss, loss_unscaled
        return loss

    def train_step(self, dataloader, optimizer, epoch=None, disable_tqdm=False):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        for batch, (X, y, z, idx) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}", disable=disable_tqdm), 0
        ):

            batch_size = len(X)
            # logging.debug(f"batch size recieved:{batch_size}")
            X = X.to(self.device)
            z = z.to(self.device)
            z_hat = self.forward(X.float())
            loss = self.loss_function(z.unsqueeze(-1).float(), z_hat)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            # logging.debug(f"Loss: {loss}")
            losses.append(loss)

        average_loss = np.sum(losses) / size
        logging.debug(f"Loss: {average_loss:>7f}")
        return average_loss

    def validation_step(self, dataloader, scheduler=None):
        size = len(dataloader.dataset)

        test_loss = []
        with torch.no_grad():
            for X, y, z, _ in dataloader:
                X = X.to(self.device)
                z = z.to(self.device)
                z_hat = self.forward(X.float())
                test_loss.append(
                    self.loss_function(z.unsqueeze(-1).float(), z_hat).item()
                )

        average_loss = np.sum(test_loss) / size

        if scheduler is not None:
            scheduler.step(average_loss)
        logging.debug(f"Test MSE: {average_loss:>7f}")
        return average_loss

    def predict(self, dataloader, unscale=False):

        if not isinstance(dataloader, DataLoader):
            batch_size = min(len(dataloader), 512)
            dataloader = pt.pandas_to_numpy_data(
                dataloader,
                regressor_var=self.flux,
                batch_size=batch_size,
                shuffle=False,
            )  # --- batch size doesnt matter here because it's just prediction

        size = len(dataloader.dataset)
        pred = []
        losses = []
        losses_unscaled = []
        for batch, (x, y, z, idx) in enumerate(tqdm(dataloader)):
            x = x.to(self.device)
            z = z.to(self.device)
            z_hat = self.forward(x.float())
            pred.append(z_hat.squeeze().detach().cpu().numpy())
            loss = self.loss_function(z.unsqueeze(-1).float(), z_hat, unscale=unscale)
            if unscale:
                losses_unscaled.append(loss[1].item())
                loss = loss[0]
            losses.append(loss.item())
        average_loss = np.sum(losses) / size

        pred = np.asarray(pred, dtype=object).flatten()

        if unscale:
            unscaled_avg_loss = np.sum(losses_unscaled) / size
            return pred, average_loss, unscaled_avg_loss
        return pred, average_loss


class ITGDataset(Dataset):
    def __init__(self, X, y, z=None, indices=None):
        self.X = X
        self.y = y
        self.z = z
        self.indices = indices

        if self.indices is None:
            # add indices to all the data points
            self.indices = np.arange(len(self.X))

    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        if self.z is not None:
            return self.X[idx], self.y[idx], self.z[idx], self.indices[idx]
        else:
            return self.X[idx], self.y[idx], self.indices[idx]

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
        target_var: str = "stable_label",
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

        cols = [c for c in self.data if c != self.label and c != "index"]
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
    train_dataset,
    val_dataset,
    epochs,
    learning_rate=0.001,
    weight_decay=True,
    # pipeline = True,
    patience=None,
    checkpoint=None,
    checkpoint_path=None,
    save_path=None,
    train_batch_size=None,
    val_batch_size=None,
):

    if train_batch_size is None:
        train_batch_size = int(len(train_dataset) / 10)
    if val_batch_size is None:
        val_batch_size = int(len(val_dataset) / 10)

    # if pipeline:
    #     train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    if model.type == "Regressor":
        regressor_var = model.flux
    else:
        regressor_var = None

    train_loader = pt.pandas_to_numpy_data(
        train_dataset,
        regressor_var=model.flux,
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = pt.pandas_to_numpy_data(
        val_dataset, batch_size=val_batch_size, shuffle=False
    )
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
            loss, train_acc = model.train_step(train_loader, opt, epoch=epoch)
            losses.append(loss)
            train_accuracy.append(train_acc)

            val_loss, val_acc = model.validation_step(val_loader)
            validation_losses.append(val_loss)
            val_accuracy.append(val_acc)

            stopping_metric = -np.asarray(val_accuracy)

        elif model.type == "regressor":
            logging.debug(f"Epoch: {epoch}")
            loss = model.train_step(train_loader, opt, epoch=epoch)
            losses.append(loss)

            val_loss = model.validation_step(val_loader)
            validation_losses.append(val_loss)

            stopping_metric = np.asarray(validation_losses)
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

    if save_path:
        torch.save(model.state_dict(), save_path)

    if model.type == "classifier":
        return model, [losses, train_accuracy, validation_losses, val_accuracy]

    elif model.type == "regressor":
        return model, [losses, validation_losses]


def load_model(model, save_path, device, scaler, flux, dropout):
    logging.info(f"Model Loaded: {model}")
    if model == "Classifier":
        classifier = Classifier(device=device)
        classifier.load_state_dict(torch.load(save_path))
        return classifier

    elif model == "Regressor":
        regressor = Regressor(device=device, scaler=scaler, flux=flux, dropout=dropout)
        regressor.load_state_dict(torch.load(save_path))
        return regressor
