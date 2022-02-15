import numpy as np
import pandas as pd
import h5py as h5
import seaborn as sns
import tensorflow as tf
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle

# The inputs to the Neural Network
with h5.File("../qlk_jetexp_nn_training_database_minimal.h5", "r") as f:
    inputs = f["input"]["block0_values"][()]
    input_names = f["input"]["block0_items"][()]
    index_inp = f["input"]["axis1"][()]  # row number from 0 to len(inputs)

    # The target outputs for the NN
    outputs = f["output"]["block0_values"][()]
    output_names = f["output"]["block0_items"][()]
    index_out = f["output"]["axis1"][
        ()
    ]  # row number from 0 to len(inputs) with some missing rows


# Load the data into the dataframe
df_in = pd.DataFrame(inputs, index_inp, input_names)
df_out = pd.DataFrame(outputs, index_out, output_names)

train_data = pd.read_pickle(
    "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data.pkl"
)

X_train, Y_train = train_data.iloc[:, :-1].to_numpy(), train_data.iloc[:, -1].to_numpy()

validation_data = pd.read_pickle(
    "/share/rcifdata/jbarr/UKAEAGroupProject/data/validation_data.pkl"
)

X_val, Y_val = (
    validation_data.iloc[:, :-1].to_numpy(),
    validation_data.iloc[:, -1].to_numpy(),
)

# standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_val = scaler.transform(X_val)


parameters = {"nodes": [5, 10, 20, 30], "layers": [2, 3, 4]}


def grid_search(build_fn, parameters, train_data, val_data):
    """
    Inputs:
        build_fn: a function that will be used to build the neural network
        parameters: a dictionary of model parameters
        train_data:
        val_data
    """

    # unpack data

    x_train, y_train = train_data

    x_val, y_val = val_data

    results_dict = {}

    counter = 0

    best_val_loss = sys.float_info.max

    for i in parameters["layers"]:

        # List of possible node combinations
        n = i
        nodes = tuple([parameters["nodes"] for j in range(i)])

        combs = np.array(np.meshgrid(*nodes)).T.reshape(-1, n)

        for node in combs:

            # build model
            model = build_fn(i, node)

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")

            history = model.fit(x_train, y_train, batch_size=4096, epochs=25, verbose=2)

            evaluate = model.evaluate(x_val, y_val, batch_size=4096)

            trial_dict = {
                "layers": i,
                "nodes": node,
                "history": history.history,
                "perfomance": evaluate,
            }

            if evaluate[1] < best_val_loss:
                results_dict["best_model"] = trial_dict

            results_dict["trial_" + str(counter)] = trial_dict

            file_name = f"./grid_search/trial_{str(counter)}.pkl"
            with open(file_name, "wb") as file:
                pickle.dump(trial_dict, file)

            counter += 1
    return results_dict


def build_classifier(n_layers, nodes):
    model = tf.keras.Sequential()

    # Flexible number of hidden layers
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(nodes[i], activation="relu"))

    # Final classifer layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


results_dict = grid_search(
    build_classifier, parameters, (x_train, Y_train), (x_val, Y_val)
)

with open("/home/tmadula/UKAEAGroupProject/grid_search_results.pickle", "wb") as file:
    pickle.dump(results_dict, file)
