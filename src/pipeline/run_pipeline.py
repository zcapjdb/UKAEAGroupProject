import coloredlogs, verboselogs, logging
import os
import copy

import pipeline.pipeline_tools as pt
import pipeline.Models as md

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

FLUX = cfg["flux"]
PRETRAINED = cfg["pretrained"]
PATHS = cfg["data"]
SAVE_PATHS = cfg["save_paths"]

train_dataset, valid_dataset = pt.prepare_data(
    PATHS["train"], PATHS["validation"], target_column=FLUX
)
# Sample subset of data to use in active learning (10K for now)
# TODO: Needs to be the true training samples used!!!
train_sample = copy.deepcopy(train_dataset)

plot_sample = valid_dataset.sample(10_000)

valid_dataset.remove(plot_sample.data.index)

valid_plot_loader = pt.pandas_to_numpy_data(plot_sample)

# Load pretrained models
logging.info("Loaded the following models:\n")
models = {}
for model in PRETRAINED:
    if PRETRAINED[model][FLUX]["trained"] == False:
        train_sample = train_dataset.sample(20_000)
    else:
        train_sample = train_dataset

    if PRETRAINED[model][FLUX]["trained"] == True:
        trained_model = md.load_model(model, PRETRAINED[model][FLUX]["save_path"])
        models[model] = trained_model

    else:
        logging.info(f"{model} not trained - training now")
        models[model] = (
            model.Classifier() if model == "Classifier" else model.Regressor()
        )
        models[model], _ = md.train_model(
            models[model],
            train_sample,
            valid_dataset,
            save_path=PRETRAINED[model][FLUX]["save_path"],
            epochs=cfg["train_epochs"],
            patience=cfg["train_patience"],
        )

if len(train_sample) > 100_000:
    logger.warning("Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data")


lam = cfg["lambda"]
logging.info(f"Training for lambda: {lam}")

# Dictionary to store results of the classifier and regressor for later use
output_dict = pt.output_dict

for i in range(cfg["iterations"]):
    logging.info(f"Iteration: {i+1}\n")
    valid_sample = valid_dataset.sample(10_000)

    # remove the sampled data points from the dataset
    valid_dataset.remove(valid_sample.data.index)

    valid_sample, misclassified_sample = pt.select_unstable_data(
        valid_sample, batch_size=100, classifier=models["Classifier"]
    )

    epochs = cfg["initial_epochs"] * (i + 1)

    if cfg["retrain_classifier"]:
        # retrain the classifier on the misclassified points
        (
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            missed_loss,
            missed_acc,
        ) = pt.retrain_classifier(
            misclassified_sample,
            train_sample,
            valid_dataset,
            models["Classifier"],
            batch_size=100,
            epochs=epochs,
            lam=lam,
            patience=cfg["patience"],
        )

        save_path = os.path.join(SAVE_PATHS["plots"], f"Iteration_{i+1}")
        pt.plot_classifier_retraining(
            train_loss, train_acc, val_loss, val_acc, missed_loss, missed_acc, save_path
        )
        output_dict["class_train_loss"].append(train_loss)
        output_dict["class_val_loss"].append(val_loss)
        output_dict["class_missed_loss"].append(missed_loss)
        output_dict["class_train_acc"].append(train_acc)
        output_dict["class_val_acc"].append(val_acc)
        output_dict["class_missed_acc"].append(missed_acc)

    # TODO: diagnose how well the classifier retraining does

    uncertain_dataset, uncert_before, data_idx = pt.regressor_uncertainty(
        valid_sample,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        valid_dataset=valid_dataset,
    )

    (
        train_sample_origin,
        train_uncert_before,
        train_uncert_idx,
    ) = pt.regressor_uncertainty(
        train_sample,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        train_data=True,
    )

    train_sample.add(uncertain_dataset)

    uncertain_loader = DataLoader(
        train_sample, batch_size=len(train_sample), shuffle=True
    )

    prediction_before, prediction_idx_order = models["Regressor"].predict(
        uncertain_loader
    )

    # regressor_unceratinty adds points back into valid_dataset so new dataloader is needed
    valid_loader = pt.pandas_to_numpy_data(valid_dataset)

    valid_pred_before, valid_pred_order = models["Regressor"].predict(valid_plot_loader)

    # Retrain Regressor (Further research required)
    train_loss, test_loss = pt.retrain_regressor(
        uncertain_loader,
        valid_loader,
        models["Regressor"],
        learning_rate=cfg["learning_rate"],
        epochs=epochs,
        validation_step=True,
        lam=lam,
        patience=cfg["patience"],
    )

    output_dict["train_losses"].append(train_loss)
    output_dict["test_losses"].append(test_loss)

    prediction_after, _ = models["Regressor"].predict(
        uncertain_loader, prediction_idx_order
    )
    logging.debug("Running prediction on validation data set")
    valid_pred_after,_ = models["Regressor"].predict(valid_plot_loader, valid_pred_order)

    _, uncert_after, _ = pt.regressor_uncertainty(
        valid_sample,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        order_idx=data_idx,
    )
    _, train_uncert_after, _ = pt.regressor_uncertainty(
        train_sample_origin,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        order_idx=train_uncert_idx,
        train_data=True,
    )

    logging.info("Change in uncertainty for most uncertain data points:")
    output_dict["d_novel_uncert"].append(
        pt.uncertainty_change(x=uncert_before, y=uncert_after)
    )

    logging.info("Change in uncertainty for training data:")
    output_dict["d_uncert"].append(
        pt.uncertainty_change(x=train_uncert_before, y=train_uncert_after)
    )

    _ = pt.mse_change(
        prediction_before,
        prediction_after,
        prediction_idx_order,
        data_idx,
        uncertain_loader,
        [uncert_before, uncert_after],
        save_path = SAVE_PATHS["plots"], 
        iteration=i,
        lam = lam
    )

    train_mse_before, train_mse_after, delta_mse = pt.mse_change(
        prediction_before,
        prediction_after,
        prediction_idx_order,
        train_uncert_idx,
        uncertain_loader,
        uncertainties=[train_uncert_before, train_uncert_after],
        data="train",
        save_path = SAVE_PATHS["plots"], 
        iteration=i,
        lam = lam
    )
    n_train = len(train_sample_origin)
    output_dict["valid_pred_before"].append(valid_pred_before)
    output_dict["valid_pred_after"].append(valid_pred_after)
    output_dict["mse_before"].append(train_mse_before)
    output_dict["mse_after"].append(train_mse_after)
    output_dict["d_mse"].append(delta_mse)
    output_dict["n_train_points"].append(n_train)

if not os.path.exists(SAVE_PATHS["outputs"]):
    os.makedirs(SAVE_PATHS["outputs"])

save_dest = os.path.join(SAVE_PATHS["outputs"], FLUX)
if not os.path.exists(save_dest): os.mkdir(save_dest)

output_path = os.path.join(save_dest, f"pipeline_outputs_lam_{lam}.pkl")
with open(output_path, "wb") as f:
    pickle.dump(output_dict, f)
