# Load the required data

from scripts.pipeline_tools import (
    prepare_data,
    regressor_uncertainty,
    select_unstable_data,
    retrain_regressor,
    retrain_classifier,
    uncertainty_change,
)
from scripts.Models import ITGDatasetDF, load_model, ITGDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from scripts.utils import train_keys

# TODO: Put some of these variables in a yaml config file

pretrained = {
    "ITG_class": {
        "trained": True,
        # "save_path": "/home/tmadula/UKAEAGroupProject/src/notebooks/classifier_model.pt",
        "save_path": "/unix/atlastracking/jbarr/UKAEAGroupProject/src/notebooks/classifier_model.pt",
        # "save_path": "/Users/thandikiremadula/Desktop/UKAEA_data/classifier_model.pt"
    },
    "ITG_reg": {
        "trained": True,
        # "save_path": "/home/tmadula/UKAEAGroupProject/src/notebooks/regression_model.pt",
        "save_path": "/unix/atlastracking/jbarr/UKAEAGroupProject/src/notebooks/regression_model.pt",
        # "save_path": "/Users/thandikiremadula/Desktop/UKAEA_data/regression_model.pt"
    },
}

# Data loading

# TRAIN_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
TRAIN_PATH = "/unix/atlastracking/jbarr/train_data_clipped.pkl"
# TRAIN_PATH = "/Users/thandikiremadula/Desktop/UKAEA_data/train_data_clipped.pkl"

# VALIDATION_PATH = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
VALIDATION_PATH = "/unix/atlastracking/jbarr/valid_data_clipped.pkl"
# VALIDATION_PATH = "/Users/thandikiremadula/Desktop/UKAEA_data/valid_data_clipped.pkl"

RETRAIN_CLASSIFIER = False

train_data, val_data = prepare_data(
    TRAIN_PATH,
    VALIDATION_PATH,
    target_column="efiitg_gb",
    target_var="itg",
    valid_size=1_000_000,
)

scaler = StandardScaler()
scaler.fit_transform(train_data.drop(["itg"], axis=1))

train_dataset = ITGDatasetDF(train_data, target_column="efiitg_gb", target_var="itg")
valid_dataset = ITGDatasetDF(val_data, target_column="efiitg_gb", target_var="itg")

# TODO: further testing of the scale function
train_dataset.scale(scaler)
valid_dataset.scale(scaler)

# Load pretrained models
models = {}
for model in pretrained:
    if pretrained[model]["trained"] == True:
        trained_model = load_model(model, pretrained[model]["save_path"])
        models[model] = trained_model

for model_name in models:
    print(f"Model: {model_name}")
    print(models[model_name])

# Train untrained models (may not be needed)

# Sample subset of data to use in active learning (10K for now)
# TODO: Needs to be the true training samples used!!!
train_sample = train_dataset.sample(10_000)
valid_sample = valid_dataset.sample(10_000)


# remove the sampled data points from the dataset
valid_dataset.remove(valid_sample.data.index)

# Pass points through the ITG Classifier and return points that pass (what threshold?)
valid_sample, misclassified_sample = select_unstable_data(valid_sample, batch_size=100, classifier=models["ITG_class"])

if RETRAIN_CLASSIFIER == True:
    # retrain the classifier on the misclassified points
    models["ITG_class"] = retrain_classifier(
        misclassified_sample,
        valid_dataset,
        models["ITG_class"],
        batch_size=100,
        epochs=5,
        verbose=True,
    )
#TODO: diagnose how well the classifier retraining does
#TODO: verbose flag a good way to control the amount of output from different functins - not yet implemented
# From first run through it does seem like training on the misclassified points hurts the validation dataset accuracy quite a bit


# Run MC dropout on points that pass the ITG classifier and return
uncertain_dataset, uncert_before, data_idx = regressor_uncertainty(
    valid_sample,
    models["ITG_reg"],
    n_runs=25,
    keep=0.25,
    plot=False,
    valid_dataset=valid_dataset,
)

train_sample.add(uncertain_dataset)

uncertain_loader = DataLoader(train_sample, batch_size=len(train_sample), shuffle=True)

# Switching validation dataset to numpy arrays as it is much faster
x_array = valid_dataset.data[train_keys].values
y_array = valid_dataset.data["itg"].values
z_array = valid_dataset.data["efiitg_gb"].values
dataset_numpy = ITGDataset(x_array, y_array, z_array)
valid_loader = DataLoader(
    dataset_numpy, batch_size=int(0.1 * len(y_array)), shuffle=True
)

prediction_before = models["ITG_reg"].predict(uncertain_loader)

# Retrain Regressor (Further research required)
retrain_regressor(
    uncertain_loader,
    valid_loader,
    models["ITG_reg"],
    learning_rate=5e-4,
    epochs=5,
    validation_step=True,
)

# TODO: Fix to use reordered indices for
prediction_after = models["ITG_reg"].predict(uncertain_loader)

_, uncert_after, _ = regressor_uncertainty(
    valid_sample,
    models["ITG_reg"],
    n_runs=10,
    keep=0.25,
    order_idx=data_idx,
    plot=False,
    valid_dataset=valid_dataset,
)

# Pipeline diagnosis (Has the uncertainty decreased for new points)
uncertainty_change(uncert_before, uncert_after, plot=False)

# Pipeline diagnosis (How has the uncertainty changed for original training points)
