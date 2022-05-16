import os
import torch 

from pipeline.pipeline_tools import prepare_data
from pipeline.Models import Regressor, Classifier, train_model

import argparse
import yaml 
import pickle
import logging, verboselogs, coloredlogs
 

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file", required=True)
args = parser.parse_args()

verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")#cfg["logging_level"])

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
FLUX = cfg["flux"]
PARAMS = cfg["hyperparameters"]
OUTPUT = cfg["save_paths"]
MODE = cfg["data_mode"]
TYPE = cfg["model_type"]
Nboot = cfg["Nboot"]

logging.info(f"Training a {TYPE} using {MODE} dataset for {FLUX}")
# --------------------------------------------- Load data ----------------------------------------------------------

train_dataset, eval_dataset, test_dataset, _ = prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], target_column=FLUX, samplesize_debug=0.1
)

if MODE =="random": 
    train_dataset = train_dataset.sample(PARAMS["rand_sample_size"])
# ------------------------------------------- Train models ------------------------------------------


# Get model
if TYPE =="regressor": 
    model = Regressor(device)
elif TYPE =="classifier": 
    model = Classifier(device)
else: 
    raise Exception("Model type not supported")

logging.debug("Training Model")

# Train model
trained_model, losses = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=eval_dataset, 
    epochs = PARAMS["epochs"], 
    patience=PARAMS["patience"],
    train_batch_size=PARAMS["train_batch_size"], 
    val_batch_size=PARAMS["valid_batch_size"], 
    pipeline=False
)


# Evaluate Model performance
logging.debug("Evaluating Model Performance")
predictions, test_lossses, test_losses_unscaled = model.predict(test_dataset,unscale=True)

output_dict = {
    "metrics": losses,
    "test_losses": test_lossses,
    "test_losses_unscaled": test_losses_unscaled
}

if MODE =='full': 
    loss_name = f"{MODE}_{FLUX}_{TYPE}_losses.pkl"
    output_path = os.path.join(OUTPUT["losses"], loss_name)

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)

    model_name = f"{MODE}_{FLUX}_{TYPE}.pt"
    model_out = os.path.join(OUTPUT["models"], model_name)
    torch.save(model.state_dict(), model_out)

elif MODE == 'random':
    loss_name = f"{MODE}_{FLUX}_{TYPE}_losses_{PARAMS['rand_sample_size']}_{Nboot}.pkl"
    output_path = os.path.join(OUTPUT["losses"], loss_name)

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
    
    #output training dataset used
    data_name = f"{FLUX}_{TYPE}_train_data_{PARAMS['rand_sample_size']}_{Nboot}.pkl"
    output_path = os.path.join(OUTPUT["losses"], data_name)
    train_dataset.data.to_pickle(output_path)

    model_name = f"{MODE}_{FLUX}_{PARAMS['rand_sample_size']}_{TYPE}_{Nboot}.pt"
    model_out = os.path.join(OUTPUT["models"], model_name)
    torch.save(model.state_dict(), model_out)

    