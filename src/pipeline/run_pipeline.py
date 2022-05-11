import coloredlogs, verboselogs, logging
import os
import copy

# import comet_ml import Experiment


import pipeline.pipeline_tools as pt
import pipeline.Models as md

from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import torch
from Models import Classifier, Regressor
import argparse
import pandas as pd

# add argument to pass config file
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


# Create logger object for use in pipeline
verboselogs.install()
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")  # cfg["logging_level"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# comet_project_name = "AL-pipeline"
# experiment = Experiment(project_name = comet_project_name)


# Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50

FLUXES = cfg["flux"]  # is now a list of the relevant fluxes
PRETRAINED = cfg["pretrained"]
PATHS = cfg["data"]
SAVE_PATHS = cfg["save_paths"]
sample_size = cfg[
    "sample_size_debug"
]  # percentage of data to use - lower for debugging
lam = cfg["hyperparams"]["lambda"]
train_size = cfg["hyperparams"]["train_size"]
valid_size = cfg["hyperparams"]["valid_size"]
test_size = cfg["hyperparams"]["test_size"]
batch_size = cfg["hyperparams"]["batch_size"]
candidate_size = cfg["hyperparams"]["candidate_size"]
# Dictionary to store results of the classifier and regressor for later use
output_dict = pt.output_dict

# To Do:  explore candidate_size hyperparam, explore architecture, validation loss shouldn't be used as test loss


# --------------------------------------------- Load data ----------------------------------------------------------
train_dataset, eval_dataset, test_dataset, scaler = pt.prepare_data(
    PATHS["train"],
    PATHS["validation"],
    PATHS["test"],
    fluxes=FLUXES,
    samplesize_debug=sample_size,
)
# --- holdout set is from the test set
plot_sample = test_dataset.sample(test_size)  # Holdout dataset
holdout_set = plot_sample
# holdout_set = pt.pandas_to_numpy_data(plot_sample) # Holdout set, remaining validation is unlabeled pool
holdout_loader = DataLoader(
    holdout_set, batch_size=batch_size, shuffle=False
)  # ToDo =====>> use helper function
# --- validation set is fixed and from the evaluation
valid_dataset = eval_dataset.sample(valid_size)  # validation set
# --- unlabelled pool is from the evaluation set minus the validation set (note, I'm not using "validation" and "evaluation" as synonyms)
eval_dataset.remove(valid_dataset.data.index)  #
unlabelled_pool = eval_dataset

# --- Set up saving
save_dest = os.path.join(SAVE_PATHS["outputs"], FLUXES[0])
if not os.path.exists(save_dest):
    os.makedirs(save_dest)

# Load pretrained models
logging.info("Loaded the following models:\n")

# save copy of yaml file used for reproducibility
with open(os.path.join(save_dest, "config.yaml"), "w") as f:
    out = yaml.dump(cfg, f, default_flow_style=False)


# ------------------------------------------- Load or train first models ------------------------------------------
models = {}
for model in PRETRAINED:
    train_sample = train_dataset.sample(train_size)
    for FLUX in FLUXES:
        if PRETRAINED[model][FLUX]["trained"] == True:
            trained_model = md.load_model(
                model, PRETRAINED[model][FLUX]["save_path"], device, scaler, FLUX
            )
            models[FLUX] = {model: trained_model.to(device)}

        else:
            logging.info(f"{FLUX} {model} not trained - training now")
            models[FLUX][model] = (
                Classifier(device)
                if model == "Classifier"
                else Regressor(device, scaler, FLUX)
            )

            models[FLUX][model], losses = md.train_model(
                models[FLUX][model],
                train_sample,
                valid_dataset,
                save_path=PRETRAINED[model][FLUX]["save_path"],
                epochs=cfg["train_epochs"],
                patience=cfg["train_patience"],
            )
            if model == "Regressor":  # To Do ==== >> do the same for classifier
                train_loss, valid_loss = losses
                output_dict["train_loss_init"].append(train_loss)
            # if model == "Classifier":  --- not used currently
            #    losses, train_accuracy, validation_losses, val_accuracy = losses

# ---- Losses before the pipeline starts #TODO: Fix output dict to be able to handle multiple variables
for FLUX in FLUXES:
    _, holdout_loss = models[FLUX]["Regressor"].predict(holdout_loader)
    output_dict["test_loss_init"].append(holdout_loss)


if len(train_sample) > 100_000:
    logger.warning(
        "Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data"
    )


logging.info(f"Training for lambda: {lam}")

# ------------------------------------------------------- AL pipeline ---------------------------------------------
classifier_buffer = []
buffer_size = 0

for i in range(cfg["iterations"]):
    logging.info(f"Iteration: {i+1}\n")

    if i != 0:
        # reset the output dictionary for each iteration
        for value in output_dict.values():
            del value[:]

    # --- at each iteration the labelled pool is updated - 10_000 samples are taken out, the most uncertain are put back in
    candidates = unlabelled_pool.sample(candidate_size)
    # --- remove the sampled data points from the dataset
    unlabelled_pool.remove(candidates.data.index)

    # --- See Issue #37 --- candidates are only those that the classifier selects as unstable.
    candidates = pt.select_unstable_data(
        candidates,
        batch_size=batch_size,
        classifier=models[FLUXES[0]]["Classifier"],
        device=device,
    )  # It's not the classifier's job to say whether a point is stable or not. This happens at the end of the pipeline when we get the true labels by running Qualikiz.

    epochs = cfg["initial_epochs"] * (i + 1)

    # ---  get most uncertain candidate inputs as decided by regressor   --- NEW AL FRAMEWORK GOES HERE
    # get the uncertainties from regressors
    candidates_uncerts, data_idxs = [], []

    for FLUX in FLUXES:
        temp_uncert, temp_idx = pt.get_uncertainty(
            candidates,
            models[FLUX]["Regressor"],
            n_runs=cfg["MC_dropout_runs"],
            keep=cfg["keep_prob"],
            device=device,
        )
        candidates_uncerts.append(temp_uncert)
        data_idxs.append(temp_idx)

    (
        candidates,
        candidates_uncert_before,
        data_idx,
        unlabelled_pool,
    ) = pt.get_most_uncertain(
        candidates,
        unlabelled_pool=unlabelled_pool,
        out_stds=candidates_uncerts,
        idx_arrays=data_idxs,
    )
    prediction_candidates_before = []
    for FLUX in FLUXES:
        prediction_candidates_before.append(
            models[FLUX]["Regressor"].predict(candidates)
        )

    # =================== >>>>>>>>>> Here goes the Qualikiz acquisition <<<<<<<<<<<<< ==================

    # I am assuming here that the acquisition will modify the candidates dataset.
    # ToDo - May need to modify dataset to account for real data points that won't have real labels. Use dummy labels until acquisition is done?

    # ToDo ===========>>>> Now we do have the labels because Qualikiz gave them to us!  Need to discard misclassified data from enriched_train_loader, and retrain the classifier if buffer_size is big enough
    # check for misclassified data in candidates and add them to the buffer
    misclassified_data, num_misclassified = pt.check_for_misclassified_data(candidates)
    buffer_size += num_misclassified
    classifier_buffer.append(misclassified_data)
    logging.info(f"Misclassified data: {num_misclassified}")
    logging.info(f"Total Buffer size: {buffer_size}")

    # --- Classifier retraining:
    if cfg["retrain_classifier"]:
        if buffer_size >= cfg["hyperparams"]["buffer_size"]:
            # concatenate datasets from the buffer
            misclassified = pd.concat(classifier_buffer)
            misclassified_dataset = md.ITGDatasetDF(
                misclassified, FLUXES[0], keep_index=True
            )

            logging.info("Buffer full, retraining classifier")
            # retrain the classifier on the misclassified points
            losses, accs = pt.retrain_classifier(
                misclassified_dataset,
                train_sample,
                valid_dataset,
                models["Classifier"],
                batch_size=batch_size,
                epochs=epochs,
                lam=lam,
                patience=cfg["patience"],
            )

            output_dict["class_train_loss"].append(losses[0])
            output_dict["class_val_loss"].append(losses[1])
            output_dict["class_missed_loss"].append(losses[2])
            output_dict["class_train_acc"].append(accs[0])
            output_dict["class_val_acc"].append(accs[1])
            output_dict["class_missed_acc"].append(accs[2])

            # reset buffer
            classifier_buffer = []
            buffer_size = 0

    # --- train data enriched by new unstable candidate points
    train_sample.add(candidates)
    n_train = len(train_sample)

    # --- validation on holdout set before regressor is retrained (this is what's needed for AL)
    holdout_pred_before = []
    for FLUX in FLUXES:
        preds, _ = models[FLUX]["Regressor"].predict(holdout_loader)
        holdout_pred_before.append(preds)

    # ---------------------------------------------- Retrain Regressor with added data (ToDo: Further research required)---------------------------------
    train_loss, test_loss = pt.retrain_regressor(
        train_sample,
        valid_dataset,
        models["Regressor"],
        learning_rate=cfg["learning_rate"],
        epochs=epochs,
        validation_step=True,
        lam=lam,
        patience=cfg["patience"],
        batch_size=batch_size,
    )

    # TODO: Retrain second regressor


    # --- validation on holdout set after regressor is retrained
    logging.info("Running prediction on validation data set")
    holdout_pred_after, holdout_loss, holdout_loss_unscaled = models[
        "Regressor"
    ].predict(holdout_loader, unscale=True)

    # TODO: Same for second regressor

    _, candidates_uncert_after, _, _ = pt.regressor_uncertainty(
        candidates,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        order_idx=data_idx,
    )  # --- uncertainty of newly added points TODO: Breakdown this calculation


    logging.info("Change in uncertainty for most uncertain data points:")

    output_dict["d_novel_uncert"].append(
        pt.uncertainty_change(
            x=candidates_uncert_before,
            y=candidates_uncert_after,
            plot_title="Novel data",
            iteration=i,
            save_path=save_dest,
        )
    )  # TODO: need for second regressor

    output_dict["novel_uncert_before"].append(candidates_uncert_before)
    output_dict["novel_uncert_after"].append(candidates_uncert_after)

    output_dict["holdout_pred_before"].append(
        holdout_pred_before
    )  # these two are probably the only important ones
    output_dict["holdout_pred_after"].append(holdout_pred_after)
    output_dict["holdout_ground_truth"].append(holdout_set.target)
    output_dict["retrain_losses"].append(train_loss)
    output_dict["retrain_test_losses"].append(test_loss)
    output_dict["post_test_loss"].append(holdout_loss)
    output_dict["post_test_loss_unscaled"].append(holdout_loss_unscaled)

    try:
        output_dict["mse_before"].append(
            train_mse_before
        )  # these three relate to the training MSE, probably not so useful to inspect
        output_dict["mse_after"].append(train_mse_after)
        output_dict["d_mse"].append(delta_mse)
    except:
        pass
    output_dict["n_train_points"].append(n_train)

    # --- Save at end of iteration
    output_path = os.path.join(
        save_dest, f"pipeline_outputs_lam_{lam}_iteration_{i}.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
    regressor_path = os.path.join(save_dest, f"regressor_lam_{lam}_iteration_{i}.pkl")
    torch.save(models["Regressor"].state_dict(), regressor_path)
    classifier_path = os.path.join(save_dest, f"classifier_lam_{lam}_iteration_{i}.pkl")
    torch.save(models["Classifier"].state_dict(), classifier_path)
