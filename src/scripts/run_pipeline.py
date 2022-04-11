import coloredlogs, verboselogs, logging
import os
from scripts.pipeline_tools import (
    prepare_data,
    regressor_uncertainty,
    select_unstable_data,
    retrain_regressor,
    retrain_classifier,
    pandas_to_numpy_data,
    uncertainty_change,
    mse_change,
)
from scripts.Models import ITGDatasetDF, load_model, ITGDataset
from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import argparse


# add argument to pass config file
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


# Create logger object for use in pipeline
verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level=cfg["logging_level"])

# Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50


PRETRAINED = cfg["pretrained"]
PATHS = cfg["data"]
SAVE_PATHS = cfg["save_paths"]

train_dataset, valid_dataset = prepare_data(
    PATHS["train"], PATHS["validation"], target_column="efiitg_gb", target_var="itg"
)

# Load pretrained models
logging.info("Loaded the following models:\n")
models = {}
for model in PRETRAINED:
    if PRETRAINED[model]["trained"] == True:
        trained_model = load_model(model, PRETRAINED[model]["save_path"])
        models[model] = trained_model

# Train untrained models (may not be needed)

# Sample subset of data to use in active learning (10K for now)
# TODO: Needs to be the true training samples used!!!
train_sample = train_dataset.sample(10_000)

lam = cfg["lambda"]
logging.info(f"Training for lambda: {lam}")

train_losses = []
test_losses = []
n_train_points = []
mse_before = []
mse_after = []
d_mse = []
d_train_uncert = []

for i in range(cfg["iterations"]):
    logging.info(f"Iteration: {i+1}\n")
    valid_sample = valid_dataset.sample(10_000)

    # remove the sampled data points from the dataset
    valid_dataset.remove(valid_sample.data.index)

    valid_sample, misclassified_sample = select_unstable_data(
        valid_sample, batch_size=100, classifier=models["ITG_class"]
    )

    epochs = cfg["initial_epoch"] * (i + 1)

    if cfg["retrain_classifier"]:
        # retrain the classifier on the misclassified points
        train_loss, train_acc, val_loss, val_acc = retrain_classifier(
            misclassified_sample,
            train_sample,
            valid_dataset,
            models["ITG_class"],
            batch_size=100,
            epochs=epochs,
            lam=lam,
            patience=cfg["patience"]
        )
    # TODO: diagnose how well the classifier retraining does
    # From first run through it does seem like training on the misclassified points hurts the validation dataset accuracy quite a bit

    uncertain_datset, uncert_before, data_idx = regressor_uncertainty(
        valid_sample,
        models["ITG_reg"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        valid_dataset=valid_dataset,
    )

    train_sample_origin, train_uncert_before, train_uncert_idx = regressor_uncertainty(
        train_sample,
        models["ITG_reg"],
        n_runs=cfg["MC_dropout_runs"],
        train_data=True,
    )

    train_sample.add(uncertain_datset)

    uncertain_loader = DataLoader(
        train_sample, batch_size=len(train_sample), shuffle=True
    )

    prediction_before, prediction_idx_order = models["ITG_reg"].predict(
        uncertain_loader
    )

    # regressor_unceratinty adds points back into valid_dataset so new dataloader is needed
    valid_loader_modified = pandas_to_numpy_data(valid_dataset)


    # Retrain Regressor (Further research required)
    train_loss, test_loss = retrain_regressor(
        uncertain_loader,
        valid_loader_modified,
        models["ITG_reg"],
        learning_rate=1e-3,
        epochs=epochs,
        validation_step=True,
        lam=lam,
        patience=cfg["patience"]
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    prediction_after, _ = models["ITG_reg"].predict(
        uncertain_loader, prediction_idx_order
    )

    _, uncert_after, _ = regressor_uncertainty(
        valid_sample,
        models["ITG_reg"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        order_idx=data_idx,
    )
    _, train_uncert_after, _ = regressor_uncertainty(
        train_sample_origin,
        models["ITG_reg"],
        n_runs=cfg["MC_dropout_runs"],
        order_idx=train_uncert_idx,
        train_data=True,
    )

    _ = uncertainty_change(x=uncert_before, y=uncert_after)

    d_train_uncert.append(
        uncertainty_change(x=train_uncert_before, y=train_uncert_after)
    )

    _ = mse_change(
        prediction_before,
        prediction_after,
        prediction_idx_order,
        data_idx,
        uncertain_loader,
        [uncert_before, uncert_after],
    )

    train_mse_before, train_mse_after, delta_mse = mse_change(
        prediction_before,
        prediction_after,
        prediction_idx_order,
        train_uncert_idx,
        uncertain_loader,
        uncertainties=[train_uncert_before, train_uncert_after],
        data="train",
    )
    mse_before.append(train_mse_before)
    mse_after.append(train_mse_after)
    d_mse.append(delta_mse)
    n_train = len(train_sample_origin)
    n_train_points.append(n_train)

output_dict = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "n_train_points": n_train_points,
    "mse_before": mse_before,
    "mse_after": mse_after,
    "d_mse": d_mse,
    "d_uncert": d_train_uncert,
}

if not os.path.exists(SAVE_PATHS["outputs"]):
    os.makedirs(SAVE_PATHS["outputs"])

output_path = os.path.join(SAVE_PATHS["outputs"], f"pipeline_outputs_lam_{lam}_2.pkl")
with open(output_path, "wb") as f:
    pickle.dump(output_dict, f)
