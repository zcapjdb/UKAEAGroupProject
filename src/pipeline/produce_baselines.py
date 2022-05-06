import torch 

from pipeline.pipeline_tools import prepare_data
from pipeline.Models import Regressor, train_model

import argparse
import yaml 
import pickle
import logging, verboselogs, coloredlogs
import tracemalloc
 


verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")#cfg["logging_level"])

# --------------------------------------------- Load data ----------------------------------------------------------
PATHS = {
    "train": "/share/rcifdata/tmadula/data/UKAEA/train_data_clipped.pkl",
    "validation": "/share/rcifdata/tmadula/data/UKAEA/valid_data_clipped.pkl",
    "test": "/share/rcifdata/tmadula/data/UKAEA/test_data_clipped.pkl"
}

FLUX = "efiitg_gb"

train_dataset, eval_dataset, test_dataset = prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], target_column=FLUX, samplesize_debug=0.1
)
# ------------------------------------------- Train models ------------------------------------------

# Get the device we are running on
gpu = torch.cuda.is_available()

if gpu:
    device = torch.cuda.current_device()
    print(f'GPU device: {device}')
else:
    device = 'cpu'
    print('No GPU')


# Get model
regressor = Regressor(device)
logging.debug("Training Model")
# Train model
_, train_losses, eval_losses = train_model(
    model=regressor,
    train_dataset=train_dataset,
    val_dataset=eval_dataset, 
    epochs = 200, 
    patience=50,
    train_batch_size=4096, 
    val_batch_size=4096, 
    pipeline=False
)

# Evaluate Model performance
predictions, test_lossses = regressor.predict(test_dataset)

output_dict = {
    "train_losses": train_losses, 
    "valid_losses": eval_losses, 
    "test_losses": test_lossses
}

# Save losses
output_path = "full_dataset_losses.pkl"
with open(output_path, "wb") as f:
    pickle.dump(output_dict, f)

# Save trained model
torch.save(regressor.state_dict(), "./")



