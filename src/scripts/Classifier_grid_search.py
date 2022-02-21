import numpy as np
import pandas as pd
import h5py as h5
import seaborn as sns
import tensorflow as tf
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


train_data = pd.read_pickle(
    "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
)
train_keys = [
    "ane",
    "ate",
    "autor",
    "machtor",
    "x",
    "zeff",
    "gammae",
    "q",
    "smag",
    "alpha",
    "ani1",
    "ati0",
    "normni1",
    "ti_te0",
    "lognustar",
]

X_train, Y_train = train_data[train_keys].to_numpy(), train_data["target"].to_numpy()

validation_data = pd.read_pickle(
    "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
)

X_val, Y_val = (
    validation_data[train_keys].to_numpy(),
    validation_data["target"].to_numpy(),
)


# standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_val = scaler.transform(X_val)


parameters = {"nodes": [5, 10, 20, 30], "layers": [2, 3, 4]}


parameters = {"nodes": [128, 256, 512], "layers": [4, 5, 6]}


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

    inshape = x_train.shape[1]

    results_dict = {}

    counter = 0

    best_val_loss = sys.float_info.max

    for i in parameters["layers"]:

        # List of possible node combinations
        n = i
        # nodes = tuple([parameters['nodes'] for j in range(i)])

        # combs = np.array(np.meshgrid(*nodes)).T.reshape(-1,n)

        combs = [[nodesize] * i for nodesize in parameters["nodes"]]

        for node in combs:

            # build model
            model = build_fn(i, node, inshape)

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")

            stop_early = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10
            )

            history = model.fit(
                x_train,
                y_train,
                batch_size=4096,
                epochs=80,
                verbose=2,
                callbacks=[stop_early],
                validation_split=0.2,
            )

            evaluate = model.evaluate(x_val, y_val, batch_size=4096)

            trial_dict = {
                "layers": i,
                "nodes": node,
                "history": history.history,
                "perfomance": evaluate,
            }

            if evaluate[1] < best_val_loss:
                results_dict["best_model"] = trial_dict

            results_dict["trial_0" + str(counter)] = trial_dict

            file_name = f"/home/tmadula/grid_search/trial_0{str(counter)}.pkl"
            with open(file_name, "wb") as file:
                pickle.dump(trial_dict, file)

            counter += 1
    return results_dict


def build_classifier(n_layers, nodes, inshape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(inshape,)))
    # Flexible number of hidden layers
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(nodes[i], activation="relu"))
        model.add(tf.keras.layers.Dropout(0.1))

    # Final classifer layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.summary()
    return model

    # Final classifer layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model


with open("/home/tmadula/grid_search/grid_search_results.pickle", "wb") as file:
    pickle.dump(results_dict, file)


results_dict = grid_search(
    build_classifier, parameters, (x_train, Y_train), (x_val, Y_val)
)

with open("/home/tmadula/UKAEAGroupProject/grid_search_results.pickle", "wb") as file:
    pickle.dump(results_dict, file)
