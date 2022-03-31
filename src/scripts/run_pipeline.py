# Load the required data

from scripts.pipeline_tools import classifier_accuracy, prepare_data, select_unstable_data
from scripts.Models import ITGDatasetDF, load_model
from sklearn.preprocessing import StandardScaler

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



train_data, val_data = prepare_data(TRAIN_PATH, VALIDATION_PATH, target_column='efiitg_gb', target_var='itg')

scaler = StandardScaler()
scaler.fit_transform(train_data.drop(['itg'], axis = 1))

train_dataset = ITGDatasetDF(train_data, target_column='efiitg_gb', target_var='itg')
valid_dataset = ITGDatasetDF(train_data, target_column='efiitg_gb', target_var='itg')

# # TODO: further testing of the scale function
train_dataset.scale(scaler)
valid_dataset.scale(scaler)


# Load pretrained models
models = {}
for model in pretrained:
    if pretrained[model]["trained"] == True:
        trained_model = load_model(model, pretrained[model]["save_path"])
        models[model] = trained_model

for model_name in models:
    print(f'Model: {model_name}')
    print(models[model_name])

# Train untrained models (may not be needed)

# Sample subset of data to use in active learning (10K for now)

valid_sample = valid_dataset.sample(10_000)

# print(valid_sample.data.columns)

# Pass points through the ITG Classifier and return points that pass (what threshold?)
select_unstable_data(valid_sample, 10, models['ITG_class'], target_col='efiitg_gb',target_var='itg') 
classifier_accuracy(valid_sample, target_var='itg')

# Run MC dropout on points that pass the ITG classifier 

# Return X % of data points with highest uncertainty

# Retrain Regressor (Further research required)

# Pipeline diagnosis (Has the uncertainty decreased for new points)

# Pipeline diagnosis (How has the uncertainty changed for original training points)
