import comet_ml
import logging
import os 

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from scripts.utils import train_keys
# import pickle5 as pickle

# os.environ['NUMEXPR_MAX_THREADS'] = '48'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def build_model_graph(experiment, inshape):
    model = Sequential()
    model.add(
        Dense(
            experiment.get_parameter("D1_units"),
            activation= experiment.get_parameter('activation'),
            input_shape=(inshape,),
        )
    )
    model.add(Dropout(0.1))
    model.add(
        Dense(
            experiment.get_parameter("D2_units"),
            activation= experiment.get_parameter('activation')
        )
    )
    model.add(Dropout(0.1))
    model.add(
        Dense(
            experiment.get_parameter("D3_units"),
            activation= experiment.get_parameter('activation')
        )
    )
    model.add(Dropout(0.1))
    model.add(
        Dense(
            experiment.get_parameter("D4_units"),
            activation= experiment.get_parameter('activation')
        )
    )
    model.add(Dropout(0.1))
    model.add(
        Dense(
            experiment.get_parameter("D5_units"),
            activation= experiment.get_parameter('activation')
        )
    )

    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"],
    )
    return model

def train(experiment, model, x_train, y_train, x_test, y_test, n):
    if n < 4096: 
        batch_size = 512
    else: 
        batch_size = 1024
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=40,
        validation_data=(x_test, y_test),
    )

def evaluate(experiment, model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    LOGGER.info("Score %s", score)

def get_dataset(n=None):
    train_path = "/home/tmadula/data/UKAEA/train_data_clipped.pkl"
    valid_path = "/home/tmadula/data/UKAEA/valid_data_clipped.pkl"
    
    # with open(train_path, "rb") as fh:
    #     train_data = pickle.load(fh)
    
    # with open(valid_path, "rb") as fh:
    #     validation_data = pickle.load(fh)

    train_data = pd.read_pickle(train_path)

    X_train, y_train = train_data.iloc[:,:15].to_numpy(), train_data.iloc[:,-1].to_numpy()

    validation_data = pd.read_pickle(valid_path)

    X_val, y_val = validation_data.iloc[:,:15].to_numpy(), validation_data.iloc[:,-1].to_numpy()

    scaler = StandardScaler()

    x_train = scaler.fit_transform(X_train)

    x_val = scaler.transform(X_val)
    
    if n:

        permuted_idx = np.random.permutation(x_train.shape[0])

        x_train, y_train = x_train[permuted_idx], y_train[permuted_idx]
        
        permuted_idx = np.random.permutation(x_val.shape[0]) 

        x_val, y_val = x_val[permuted_idx], y_val[permuted_idx]

        x_train, y_train = x_train[:n], y_train[:n]
        x_val, y_val = x_val[:n], y_val[:n] 

    return x_train, y_train, x_val, y_val


# loop to optimize the network per set of data points
list_n = [1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 30_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
# Get the dataset:

for n in list_n: 
    x_train, y_train, x_test, y_test = get_dataset(n)


    # The optimization config:
    config = {
        "algorithm": "bayes",
        "name": "In Out Classifier optimisation",
        "spec": {"maxCombo": 50, "objective": "maximize", "metric": "val_accuracy"},
        "parameters": {
            "D1_units": {
            "type": "integer",
            "scalingType": "uniform",
            "min": 128, 
            "max": 512, 
        },
        "D2_units": {
            "type": "integer",
            "scalingType": "uniform",
            "min": 128, 
            "max": 512, 
        },
        "D3_units": {
           "type": "integer",
            "scalingType": "uniform",
            "min": 128, 
            "max": 512, 
        },
        "D4_units": {
            "type": "integer",
            "scalingType": "uniform",
            "min": 128, 
            "max": 512, 
        },
        "D5_units": {
            "type": "integer",
            "scalingType": "uniform",
            "min": 128, 
            "max": 512, 
        },
            "activation": {"type": "categorical", "values": ["tanh", "relu"]}, 
        },
        "trials": 1,
    }

    opt = comet_ml.Optimizer(config)

    for experiment in opt.get_experiments(project_name="IO_optimiser"):
        # Log parameters, or others:
        experiment.log_parameter("epochs",60)

        experiment.add_tag(f'n_{n}')

        # Build the model:
        model = build_model_graph(experiment, 15)

        # Train it:
        train(experiment, model, x_train, y_train, x_test, y_test, n)

        # How well did it do?
        evaluate(experiment, model, x_test, y_test)

        # Optionally, end the experiment:
        experiment.end()