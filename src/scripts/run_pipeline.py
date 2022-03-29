# Load the required data

from pipeline_tools import prepare_data


TRAIN_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/"

VALIDATION_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/"

train_data, val_data = prepare_data(TRAIN_PATH, VALIDATION_PATH)

# Load pretrained classifiers

# Train ITG_regressor 

# Evalute Perfromance of ITG_regressor

