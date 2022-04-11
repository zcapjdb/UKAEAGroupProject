import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os

from sklearn.preprocessing import StandardScaler

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from scripts.utils import ScaleData, train_keys, target_keys
from scripts.AutoEncoder import AutoEncoder, AutoEncoderDataset, Encoder, Decoder
from scripts.AutoEncoder import EncoderBig, DecoderBig, EncoderHuge, DecoderHuge


def load_data(path):
    train_data = pd.read_pickle(path)

    # Keep only the data that gives an output
    train_data = train_data[train_data["target"] == 1]

    keys = train_keys + ["efiitg_gb"]
    train_data = train_data[keys]
    train_data = train_data.dropna()

    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data = pd.DataFrame(
        scaler.transform(train_data), columns=keys, index=train_data.index
    )

    target = train_data["efiitg_gb"]
    train_data = train_data.drop(columns=["efiitg_gb"])

    index_array = train_data.index.values

    return train_data, target, index_array, scaler


def encode(model_path, data):
    # Load in autoencoder model
    model = AutoEncoder.load_from_checkpoint(
        model_path,
        encoder=EncoderHuge,
        decoder=DecoderHuge,
        n_input=15,
        batch_size=2048,
        epochs=150,
        learning_rate=0.001,
    )
    encoder = model.encoder

    data_array = data.values
    train_tensor = torch.tensor(data_array, dtype=torch.float32)

    encoded_points = encoder(train_tensor)
    encoded_array = encoded_points.detach().numpy()

    return encoded_array


def create_bin_lists(encoded_array, bins=50):
    H, edges = np.histogramdd(encoded_array, bins=(bins, bins, bins))

    #!!! np digitize has an extra bin at the end compared to histogramdd!!
    # get which bin each point is in
    bin_x = np.fmin(np.digitize(encoded_array[:, 0], edges[0]), len(edges[0]) - 1)
    bin_y = np.fmin(np.digitize(encoded_array[:, 1], edges[1]), len(edges[1]) - 1)
    bin_z = np.fmin(np.digitize(encoded_array[:, 2], edges[2]), len(edges[2]) - 1)

    bin_lists = [[] for _ in range(bins**3)]

    for i in range(len(encoded_array)):
        # digitize returns 1 indexed arrays instead of 0 indexed for some reason!!
        idx = np.ravel_multi_index(
            (bin_x[i] - 1, bin_y[i] - 1, bin_z[i] - 1), (bins, bins, bins)
        )
        bin_lists[idx].append(
            i
        )  # appending the index of the point in the list of points in the bin

    return bin_lists


def get_points_in_bin(bin_list, n_points=10):
    index = []
    counts = []

    for i in range(n_points):
        sampled = False
        count = 0
        while not sampled:
            count += 1

            # randomly select integer from 0 to len(bin_list) and sample a point from that bin
            bin_number = np.random.randint(0, len(bin_list))

            bin_length = len(bin_list[bin_number])

            # sample from bin only if bin is not empty
            if bin_length > 0:
                sampled_index = bin_list[bin_number][np.random.randint(0, bin_length)]

                # check if point is already sampled
                if sampled not in index:
                    sampled = True

        index.append(sampled_index)
        counts.append(count)

    return index, counts


def sampled_data(train_data, target, index_array, indices, scaler):

    data = train_data.copy()
    data["efiitg_gb"] = target

    sampled_data = data.loc[index_array[indices]]

    sampled_data = pd.DataFrame(
        scaler.inverse_transform(sampled_data),
        columns=scaler.feature_names_in_,
        index=sampled_data.index,
    )

    return sampled_data


def plot_data(train_data, sampled_data):
    original = train_data.copy()
    sampled = sampled_data.copy()

    original["hue"] = "original"
    sampled["hue"] = "closest"

    concat_df = pd.concat([original.sample(len(sampled.index)), sampled]).reset_index(
        drop=True
    )

    if not os.path.exists("./sample_plots"):
        os.makedirs("./sample_plots")

    for i in train_keys:
        plt.figure()
        sns.histplot(data=concat_df, x=i, bins=50, hue="hue")
        plt.savefig(f"./sample_plots/{i}_{str(len(sampled.index.values))}.png")
        plt.clf()


def main():
    train_path = "/unix/atlastracking/jbarr/train_data_clipped.pkl"
    model_path = "/unix/atlastracking/jbarr/UKAEAGroupProject/experiment_name=0-epoch=127-val_loss=0.03.ckpt"

    print("Loading data...")
    train_data, target, index_array, scaler = load_data(train_path)

    print("Encoding data...")
    # Call autoencoder on training data
    encoded_array = encode(model_path, train_data)

    print("Creating bin lists...")
    # put encoded points into bins in latent space
    bin_lists = create_bin_lists(encoded_array)

    training_sample_sizes = [1_000, 5_000, 10_000, 15_000, 20_000, 25_000]
    for size in training_sample_sizes:
        print(f"Sampling data of size: {size}")
        # sample uniformly from bins returning the indices of the sampled points along with the number of steps to sample
        indices, counts = get_points_in_bin(bin_lists, n_points=size)

        print("Unscaling data...")
        # get the sampled data and unscale it
        sample_data = sampled_data(train_data, target, index_array, indices, scaler)

        # pickle sampled data
        with open(f"../../../sampled_data_{str(size)}.pkl", "wb") as f:
            pickle.dump(sample_data, f)

        print("Plotting data...")
        plot_data(train_data, sample_data)


if __name__ == "__main__":
    main()
