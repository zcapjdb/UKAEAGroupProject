import comet_ml
import logging
import os 

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scripts.utils import train_keys
import pickle5 as pickle 

os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def build_model_graph(experiment, inshape):
    model = Sequential()
    model.add(
        Dense(
            experiment.get_parameter("D1_units"),
            activation= experiment.get_parameter('activation'),
            input_shape=(inshape,),
        )
    )
    model.add(
        Dense(
            experiment.get_parameter("D2_units"),
            activation= experiment.get_parameter('activation')
        )
    )
    model.add(
        Dense(
            experiment.get_parameter("D3_units"),
            activation= experiment.get_parameter('activation')
        )
    )
    model.add(
        Dense(
            experiment.get_parameter("D4_units"),
            activation= experiment.get_parameter('activation')
        )
    )

    model.add(
        Dense(
            experiment.get_parameter("D5_units"),
            activation= experiment.get_parameter('activation')
        )
    )

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate = experiment.get_parameter('lr')),
        metrics=["accuracy"],
    )
    return model

def train(experiment, model, x_train, y_train, x_test, y_test):

    model.fit(
        x_train,
        y_train,
        batch_size=experiment.get_parameter("batch_size"),
        epochs=experiment.get_parameter("epochs"),
        validation_data=(x_test, y_test),
    )

def evaluate(experiment, model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    LOGGER.info("Score %s", score)

def get_dataset():
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

    return x_train, y_train, x_val, y_val


# Get the dataset:
x_train, y_train, x_test, y_test = get_dataset()

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize MNIST Network",
    "spec": {"maxCombo": 10, "objective": "minimize", "metric": "loss"},
    "parameters": {
        "D1_units": {
            "type": "integer",
            "mu": 256,
            "sigma": 50,
            "scalingType": "normal",
        },
        "D2_units": {
            "type": "integer",
            "mu": 256,
            "sigma": 50,
            "scalingType": "normal",
        },
        "D3_units": {
            "type": "integer",
            "mu": 256,
            "sigma": 50,
            "scalingType": "normal",
        },
        "D4_units": {
            "type": "integer",
            "mu": 256,
            "sigma": 50,
            "scalingType": "normal",
        },
        "D5_units": {
            "type": "integer",
            "mu": 256,
            "sigma": 50,
            "scalingType": "normal",
        },
        "batch_size": {"type": "discrete", "values": [512, 1024, 2048, 4096]},
        "epochs": {"type": "discrete", "values": [40,60,80]},
        "activation": {"type": "categorical", "values": ["tanh", "relu"]}, 
        "lr": {"type": "discrete", "values": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]}
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)

for experiment in opt.get_experiments(project_name="my_project"):
    # Log parameters, or others:
    experiment.log_parameter("epochs", 10)

    # Build the model:
    model = build_model_graph(experiment)

    # Train it:
    train(experiment, model, x_train, y_train, x_test, y_test)

    # How well did it do?
    evaluate(experiment, model, x_test, y_test)

    # Optionally, end the experiment:
    experiment.end()