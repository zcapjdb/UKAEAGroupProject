import coloredlogs, verboselogs, logging
import os
import copy

import pipeline.pipeline_tools as pt
import pipeline.Models as md

from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import torch
from Models import Classifier, Regressor
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
coloredlogs.install(level="DEBUG")#cfg["logging_level"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50

FLUX = cfg["flux"]
PRETRAINED = cfg["pretrained"]
PATHS = cfg["data"]
SAVE_PATHS = cfg["save_paths"]
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
train_dataset, eval_dataset, test_dataset = pt.prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], target_column=FLUX, samplesize_debug=0.1
)
# --- holdout set is from the test set
plot_sample = test_dataset.sample(test_size)  # Holdout dataset
holdout_set = plot_sample 
#holdout_set = pt.pandas_to_numpy_data(plot_sample) # Holdout set, remaining validation is unlabeled pool
holdout_loader = DataLoader(holdout_set, batch_size=batch_size,shuffle=False)  # ToDo =====>> use helper function
# --- validation set is fixed and from the evaluation
valid_dataset = eval_dataset.sample(valid_size) # validation set
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)  # ToDo =====>> use helper function
# --- unlabelled pool is from the evaluation set minus the validation set (note, I'm not using "validation" and "evaluation" as synonyms)
eval_dataset.remove(valid_dataset.data.index) # 
unlabelled_pool = eval_dataset


# Load pretrained models
logging.info("Loaded the following models:\n")


# ------------------------------------------- Load or train first models ------------------------------------------
models = {}
for model in PRETRAINED:
#    if PRETRAINED[model][FLUX]["trained"] == False:
#        train_sample = train_dataset.sample(20_000)
#    else:
#        train_sample = train_dataset     # THIS GIVES MEM ERROR!! CAN'T APPEND LISTS FOR THE SIZE OF THE ENTIRE TRAINING DATA!! see line 334 in pipeline_tools

    train_sample = train_dataset.sample(train_size) 
    if PRETRAINED[model][FLUX]["trained"] == True:
        trained_model = md.load_model(model, PRETRAINED[model][FLUX]["save_path"])
        models[model] = trained_model.to(device)

    else:
        logging.info(f"{model} not trained - training now")
        models[model] = (
            Classifier(device) if model == "Classifier" else Regressor(device)
        )
        
        models[model], losses  = md.train_model(
            models[model],
            train_sample,
            valid_dataset, 
            save_path=PRETRAINED[model][FLUX]["save_path"],
            epochs=cfg["train_epochs"],
            patience=cfg["train_patience"],
        )
        if model == 'Regressor': #To Do ==== >> do the same for classifier
            train_loss, valid_loss = losses
        #if model == "Classifier":  --- not used currently
        #    losses, train_accuracy, validation_losses, val_accuracy = losses

    # ---- Losses before the pipeline starts
_, holdout_loss = models["Regressor"].predict(holdout_loader)
output_dict["test_losses"].append(holdout_loss)
output_dict["train_losses"].append(train_loss)

if len(train_sample) > 100_000:
    logger.warning("Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data")


logging.info(f"Training for lambda: {lam}")

# ------------------------------------------------------- AL pipeline ---------------------------------------------

for i in range(cfg["iterations"]):
    logging.info(f"Iteration: {i+1}\n")

    # --- at each iteration the labelled pool is updated - 10_000 samples are taken out, the most uncertain are put back in
    candidates = unlabelled_pool.sample(candidate_size)  
    # --- remove the sampled data points from the dataset
    unlabelled_pool.remove(candidates.data.index)

    # --- See Issue #37 --- candidates are only those that the classifier selects as unstable.
    candidates = pt.select_unstable_data(
        candidates, batch_size=batch_size, classifier=models["Classifier"], device=device
    )   # It's not the classifier's job to say whether a point is stable or not. This happens at the end of the pipeline when we get the true labels by running Qualikiz.
       

    epochs = cfg["initial_epochs"] * (i + 1)

    # --- Classifier retraining: ============>>>> ToDo: Need to put this at the end of the pipeline instead
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
            holdout_set,
            models["Classifier"],
            batch_size=batch_size,
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


    # ---  get most uncertain candidate inputs as decided by regressor   --- NEW AL FRAMEWORK GOES HERE
    candidates, candidates_uncert_before, data_idx, unlabelled_pool = pt.regressor_uncertainty(
        candidates, # ---only unstable candidates are returned
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        unlabelled_pool=unlabelled_pool,  #---non uncertain points are added back to the unlabeled pool
        device = device,
    )
    prediction_candidates_before = models["Regressor"].predict(candidates)

    # --- get prediction and uncertainty of train sample (is that really needed?)
    (
        train_sample_origin,   # --- why is it called different tha train_sample? Are they not exactly the same?
        train_uncert_before,
        train_uncert_idx,
    ) = pt.regressor_uncertainty(
        train_sample,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        train_data=True,
    )

    prediction_train_origin, loss_train_origin = models["Regressor"].predict(train_sample_origin)


    # --- train data enriched by new unstable candidate points
    train_sample.add(candidates) 
    enriched_train_loader = DataLoader(
        train_sample, batch_size=batch_size, shuffle=True    
    ) 

    # ---  get predictions for enriched train sample before retraining (perhaps not useful?)
    prediction_before, _ = models["Regressor"].predict(enriched_train_loader)
    
    # --- validation on holdout set before regressor is retrained (this is what's needed for AL)
    holdout_pred_before, _ = models["Regressor"].predict(holdout_set) 

    # =================== >>>>>>>>>> Here goes the Qualikiz acquisition <<<<<<<<<<<<< ==================

    # ---------------------------------------------- Retrain Regressor with added data (ToDo: Further research required)---------------------------------
    # ToDo ===========>>>> Now we do have the labels because Qualikiz gave them to us!  Need to discard misclassified data from enriched_train_loader, and retrain the classifier if buffer_size is big enough
    train_loss, test_loss = pt.retrain_regressor(
        enriched_train_loader,
        valid_loader,  # ToDo ====>>> modify for consistency with md.train_model(): either datasets or dataloaders!
        models["Regressor"],
        learning_rate=cfg["learning_rate"],
        epochs=epochs,
        validation_step=True,
        lam=lam,
        patience=cfg["patience"],
    )

    # ToDo =================>>>>>>> Classifier retraining goes here, if buffer_size greater than <user defined> 

     # --- predictions for the enriched train sample after (is that really needed?)
    enriched_train_prediction_after, _ = models["Regressor"].predict(enriched_train_loader
    )
     # --- validation on holdout set after regressor is retrained
    logging.info("Running prediction on validation data set")
    holdout_pred_after,holdout_loss = models["Regressor"].predict(holdout_loader)  # ToDo ============>>> predict should return the MSE loss as well

    _, candidates_uncert_after, _, _ = pt.regressor_uncertainty(
        candidates,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        keep=cfg["keep_prob"],
        order_idx=data_idx,
    ) # --- uncertainty of newly added points
    _, train_uncert_after, _ = pt.regressor_uncertainty(
        train_sample_origin,
        models["Regressor"],
        n_runs=cfg["MC_dropout_runs"],
        order_idx=train_uncert_idx,
        train_data=True,
    ) # --- uncertainty on first training set before points were added (is that really needed?)

    logging.info("Change in uncertainty for most uncertain data points:")
    output_dict["d_novel_uncert"].append(
        pt.uncertainty_change(x=candidates_uncert_before, y=candidates_uncert_after, plot_title='Novel data', iteration=i)
    )

    logging.info("Change in uncertainty for training data:")
    output_dict["d_uncert"].append(
        pt.uncertainty_change(x=train_uncert_before, y=train_uncert_after, plot_title='Train data', iteration=i)
    )

    # --- Prediction on train dataset not needed
  #  _ = pt.mse_change(
  #      prediction_before,
  #      enriched_train_prediction_after,
  #      prediction_idx_order,
  #      data_idx,
  #      enriched_train_loader,
  #      [candidates_uncert_before, candidates_uncert_after],
  #      save_path = SAVE_PATHS["plots"], 
  #      iteration=i,
  #      lam = lam
  #  )

    try:
        train_mse_before, train_mse_after, delta_mse = pt.mse_change(
            candidates_uncert_before,
            candidates_uncert_after,
            prediction_idx_order,
            train_uncert_idx,
            enriched_train_loader,
            uncertainties=[train_uncert_before, train_uncert_after],
            data="novel",
            save_path = SAVE_PATHS["plots"], 
            iteration=i,
            lam = lam
        )
    except:
        logging.debug("pt.mse_change failed, whatever")

    n_train = len(train_sample_origin)
    output_dict["holdout_pred_before"].append(holdout_pred_before) # these two are probably the only important ones
    output_dict["holdout_pred_after"].append(holdout_pred_after)
    output_dict["test_losses"].append(holdout_loss)
    output_dict["train_losses"].append(train_loss)
    output_dict["test_losses"].append(test_loss)
    try:
        output_dict["mse_before"].append(train_mse_before) # these three relate to the training MSE, probably not so useful to inspect 
        output_dict["mse_after"].append(train_mse_after)
        output_dict["d_mse"].append(delta_mse)
    except:
        pass
    output_dict["n_train_points"].append(n_train) 

if not os.path.exists(SAVE_PATHS["outputs"]):
    os.makedirs(SAVE_PATHS["outputs"])

save_dest = os.path.join(SAVE_PATHS["outputs"], FLUX)
if not os.path.exists(save_dest): os.mkdir(save_dest)

output_path = os.path.join(save_dest, f"pipeline_outputs_lam_{lam}.pkl")
with open(output_path, "wb") as f:
    pickle.dump(output_dict, f)
