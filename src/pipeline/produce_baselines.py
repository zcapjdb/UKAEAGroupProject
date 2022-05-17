import os
import torch 

from pipeline.pipeline_tools import prepare_data
from pipeline.Models import NRegressor, Regressor, Classifier, NRegressor, train_model

import argparse
import yaml 
import pickle
import logging, verboselogs, coloredlogs
 

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file", required=True)
args = parser.parse_args()

verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

# Get the device we are running on
gpu = torch.cuda.is_available()

if gpu:
    device = torch.cuda.current_device()
    print(f'GPU device: {device}')
else:
    device = 'cpu'
    print('No GPU')

# --------------------------------------------- Digest Config ----------------------------------------------------------
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


PATHS = cfg["data"]
FLUXES = cfg["flux"]
PARAMS = cfg["hyperparameters"]
OUTPUT = cfg["save_paths"]
MODE = cfg["data_mode"]
TYPE = cfg["model_type"]

if TYPE == "regressors" or TYPE =="classifier": 
    logging.info(f"Training a {TYPE} using {MODE} dataset for {FLUXES[0]}") 
if TYPE =="nregressor":
    logging.info(f"Training an {TYPE} using {MODE} dataset for {FLUXES}") 
# --------------------------------------------- Load data ----------------------------------------------------------

train_dataset, eval_dataset, test_dataset, scaler = prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], fluxes=FLUXES, scale=True, samplesize_debug=0.1
)

if MODE =="random":
    n_data = len(train_dataset) 

    if n_data < PARAMS["rand_sample_size"]:
        logging.info(f"Training on {n_data} data points")
        PARAMS["rand_sample_size"] = n_data
    
    train_dataset = train_dataset.sample(PARAMS["rand_sample_size"])
    train_idx = list(train_dataset.data.index)
# ------------------------------------------- Train models ------------------------------------------


# Get model
if TYPE == "nregressor":
    model = NRegressor(len(FLUXES))
elif TYPE =="regressor": 
    model = Regressor(device, scaler, FLUXES[0])
elif TYPE =="classifier": 
    model = Classifier(device, scaler, FLUXES[0])
else: 
    raise Exception("Model type not supported")

logging.debug("Training Model")

# Train model
trained_model, losses = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=eval_dataset,
    regressor_var = FLUXES,
    epochs = PARAMS["epochs"], 
    patience=PARAMS["patience"],
    train_batch_size=PARAMS["train_batch_size"], 
    val_batch_size=PARAMS["valid_batch_size"],
)


# Evaluate Model performance
logging.debug("Evaluating Model Performance")
predictions, test_lossses, indices = model.predict(test_dataset,unscale=False)

output_dict = {
    "metrics": losses,
    "test_losses": test_lossses,
    "indices": indices
}

if MODE =='full': 
    loss_name = f"{MODE}_{FLUXES[0]}_{TYPE}_losses.pkl"
    output_path = os.path.join(OUTPUT["losses"], loss_name)

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)

    model_name = f"{MODE}_{FLUXES[0]}_{TYPE}.pt"
    model_out = os.path.join(OUTPUT["models"], model_name)
    torch.save(model.state_dict(), model_out)

elif MODE == 'random':
    # same the indecies of the training set
    output_dict['train_indecies'] = train_idx

    loss_name = f"{MODE}_{FLUXES[0]}_{TYPE}_losses_{PARAMS['rand_sample_size']//1_000}K.pkl"
    output_path = os.path.join(OUTPUT["losses"], loss_name)

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)

    model_name = f"{MODE}_{FLUXES[0]}_{PARAMS['rand_sample_size']}_{TYPE}.pt"
    model_out = os.path.join(OUTPUT["models"], model_name)
    torch.save(model.state_dict(), model_out)

    