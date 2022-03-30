# Load the required data

from scripts.pipeline_tools import prepare_data
from scripts.Models import load_model

# TODO: Put some of these variables in a yaml config file

pretrained = {
    "ITG_class": {
        "trained": True,
        "save_path": "/home/tmadula/UKAEAGroupProject/src/notebooks/classifier_model.pt",
    },
    "ITG_reg": {
        "trained": True,
        "save_path": "/home/tmadula/UKAEAGroupProject/src/notebooks/regression_model.pt",
    },
}

# Data loading

TRAIN_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"

VALIDATION_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"



train_data, val_data = prepare_data(TRAIN_PATH, VALIDATION_PATH)

# Load pretrained models
models = {}
for model in pretrained:
    if pretrained[model]["trained"] == True:
        trained_model = load_model(model, pretrained[model]["save_path"])
        models[model] = trained_model

for model_name in models:
    print(models[model_name])

# Train untrained models (may not be needed)

# Sample subset of data to use in active learning (10K for now)

# Pass points through the ITG Classifier and return points that pass (what threshold?)


# Run MC dropout on points that pass the ITG classifier 

# Return X % of data points with highest uncertainty

# Retrain Regressor (Further research required)

# Pipeline diagnosis (Has the uncertainty decreased for new points)

# Pipeline diagnosis (How has the uncertainty changed for original training points)
