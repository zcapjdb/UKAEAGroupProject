import coloredlogs, verboselogs, logging
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import copy

# import comet_ml import Experiment

import random
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
import numpy as np 
from multiprocessing import Pool



def ALpipeline(cfg):

    if not isinstance(cfg,dict):
        seed = cfg[0]
        cfg = cfg[1]
        SAVE_PATHS["outputs"] = cfg[2]
        SAVE_PATHS["plots"] = cfg[3]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

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

    sample_size = cfg[
        "sample_size_debug"
    ]  # percentage of data to use - lower for debugging
    lam = cfg["hyperparams"]["lambda"]
    train_size = cfg["hyperparams"]["train_size"]
    valid_size = cfg["hyperparams"]["valid_size"]
    test_size = cfg["hyperparams"]["test_size"]
    batch_size = cfg["hyperparams"]["batch_size"]
    candidate_size = cfg["hyperparams"]["candidate_size"]
    dropout = cfg["hyperparams"]["dropout"]
    # Dictionary to store results of the classifier and regressor for later use
    output_dict = pt.output_dict

    # To Do:  explore candidate_size hyperparam, explore architecture, validation loss shouldn't be used as test 

    # --- Set up saving
    save_dest = os.path.join(SAVE_PATHS["outputs"], FLUXES[0])
    if not os.path.exists(save_dest):
        os.makedirs(save_dest)

    logging.info("Saving to {}".format(save_dest))

    # save copy of yaml file used for reproducibility
    with open(os.path.join(save_dest, "config.yaml"), "w") as f:
        out = yaml.dump(cfg, f, default_flow_style=False)

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

    valid_dataset = eval_dataset.sample(valid_size)  # validation set
    # --- unlabelled pool is from the evaluation set minus the validation set (note, I'm not using "validation" and "evaluation" as synonyms)
    eval_dataset.remove(valid_dataset.data.index)  #
    unlabelled_pool = eval_dataset
    buffer_size = 0
    classifier_buffer = []
    # Load pretrained models
    logging.info("Loaded the following models:\n")

    # ------------------------------------------- Load or train first models ------------------------------------------
    models = {f:{} for f in FLUXES}
    for model in PRETRAINED:
        logging.debug(f"Size of training dataset {len(train_dataset)}")
        train_sample = train_dataset.sample(train_size)
        if model == "Regressor":
            for FLUX in FLUXES:
                if PRETRAINED[model][FLUX]["trained"] == True:
                    trained_model = md.load_model(
                        model, PRETRAINED[model][FLUX]["save_path"], device, scaler, FLUX, dropout
                    )
                    if FLUX not in models.keys():
                        models[FLUX] = {model: trained_model.to(device)}
                    else:
                        models[FLUX][model] = trained_model.to(device)

                else:
                    logging.info(f"{FLUX} {model} not trained - training now")
                    # models[FLUX][model] = (
                    #     Classifier(device)
                    #     if model == "Classifier"
                    #     else Regressor(device, scaler, FLUX)
                    # )

                    models[FLUX][model] = Regressor(device, scaler, FLUX, dropout)

                    models[FLUX][model], losses = md.train_model(
                        models[FLUX][model],
                        train_sample,
                        valid_dataset,
                        save_path=PRETRAINED[model][FLUX]["save_path"],
                        epochs=cfg["train_epochs"],
                        patience=cfg["train_patience"],
                    )
                    
                    train_loss, valid_loss = losses
                    output_dict["train_loss_init"].append(train_loss)

                    # if model == "Classifier":  --- not used currently
                    #    losses, train_accuracy, validation_losses, val_accuracy = losses
        else:
            if PRETRAINED[model][FLUXES[0]]["trained"] == True:
                trained_model = md.load_model(
                    model, PRETRAINED[model][FLUXES[0]]["save_path"], device, scaler, FLUXES[0], dropout
                )
                models[FLUXES[0]] = {model: trained_model.to(device)} 
            else: 
                logging.info(f"{FLUXES[0]} {model} not trained - training now")
                models[FLUXES[0]][model] = Classifier(device)

                models[FLUXES[0]][model], losses = md.train_model(
                        models[FLUXES[0]][model],
                        train_sample,
                        valid_dataset,
                        save_path=PRETRAINED[model][FLUXES[0]]["save_path"],
                        epochs=cfg["train_epochs"],
                        patience=cfg["train_patience"],
                    )

    for FLUX in FLUXES:
        logging.debug(f"Models for {FLUX} : {models[FLUX].keys()}")

    # ---- Losses before the pipeline starts #TODO: Fix output dict to be able to handle multiple variables
    for FLUX in FLUXES:
        logging.info(f"Test loss for {FLUX} before pipeline:")
        _, holdout_loss, holdout_loss_unscaled = models[FLUX]["Regressor"].predict(holdout_set, unscale = True)
        logging.info(f"Holdout Loss: {holdout_loss}")
        logging.info(f"Holdout Loss Unscaled: {holdout_loss_unscaled}")
        output_dict["test_loss_init"].append(holdout_loss)
        output_dict["test_loss_init_unscaled"].append(holdout_loss_unscaled)
        
    _, holdout_class_losses = models[FLUXES[0]]["Classifier"].predict(holdout_set) 
    output_dict['class_test_acc_init'].append(holdout_class_losses[1])
    # Create logger object for use in pipeline
    verboselogs.install()
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")#cfg["logging_level"])

    if len(train_sample) > 100_000:
        logger.warning(
            "Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data"
        )

        #comet_project_name = "AL-pipeline"
        #experiment = Experiment(project_name = comet_project_name)


        # Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50

    classifier_buffer = []
    buffer_size = 0

    for i in range(cfg["iterations"]):
        logging.info(f"Iteration: {i+1}\n")

        #if i != 0:
        #    # reset the output dictionary for each iteration
        #    for value in output_dict.values():
        #        del value[:]

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
            model=models[FLUX]["Regressor"],
            acquisition=cfg["acquisition"],
        )

        logging.debug(f"Number of most uncertain {len(data_idx)}")
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
        candidates, misclassified_data, num_misclassified, data_idx, candidates_uncert_before = pt.check_for_misclassified_data(
            candidates, 
            uncertainty=candidates_uncert_before, 
            indices=data_idx
            )
        buffer_size += num_misclassified
        classifier_buffer.append(misclassified_data)
        logging.info(f"Misclassified data: {num_misclassified}")
        logging.info(f"Total Buffer size: {buffer_size}")

        # --- set up retraining by rescaling all points according to new training data --------------------

       # --- unscale all datasets, 
        candidates.scale(scaler, unscale=True)
        train_sample.scale(scaler, unscale=True)
        unlabelled_pool.scale(scaler, unscale=True)
        valid_dataset.scale(scaler, unscale=True)
        holdout_set.scale(scaler, unscale=True)
    # --- train data is enriched by new unstable candidate points
        train_sample.add(candidates)
        # --- get new scaler from enriched training set, rescale them with new scaler
        scaler = StandardScaler()
        scaler.fit_transform(train_sample.data.drop(["stable_label","index"], axis=1))
        train_sample.scale(scaler)
        unlabelled_pool.scale(scaler)
        valid_dataset.scale(scaler)
        holdout_set.scale(scaler)


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
                    models[FLUXES[0]]["Classifier"],
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



        # --- validation on holdout set before regressor is retrained (this is what's needed for AL)
        holdout_pred_before = []
        train_losses, val_losses = [], []
        train_losses_unscaled, val_losses_unscaled = [], []
        holdout_pred_after, holdout_loss, holdout_loss_unscaled = [], [], []
        for FLUX in FLUXES:
            preds, _ = models[FLUX]["Regressor"].predict(holdout_set)
            holdout_pred_before.append(preds)

            retrain_losses = pt.retrain_regressor(
                train_sample,
                valid_dataset,
                models[FLUX]["Regressor"],
                learning_rate=cfg["learning_rate"],
                epochs=epochs,
                validation_step=True,
                lam=lam,
                patience=cfg["patience"],
                batch_size=batch_size,
            )

            train_loss, val_loss, train_loss_unscaled, val_loss_unscaled = retrain_losses

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_losses_unscaled.append(train_loss_unscaled)
            val_losses_unscaled.append(val_loss_unscaled)
            
            logging.info(f"Running prediction on validation data set")
            # --- validation on holdout set after regressor is retrained
            hold_pred_after, hold_loss, hold_loss_unscaled = models[FLUX]["Regressor"].predict(holdout_set, unscale=True)
            holdout_pred_after.append(hold_pred_after)
            holdout_loss.append(hold_loss)
            holdout_loss_unscaled.append(hold_loss_unscaled)
            logging.info(f"{FLUX} test loss: {hold_loss}")
            logging.info(f"{FLUX} test loss unscaled: {hold_loss_unscaled}")
  


        candidates_uncerts_after, data_idxs_after = [], []

        for FLUX in FLUXES:
            temp_uncert, temp_idx = pt.get_uncertainty(
                candidates,
                models[FLUX]["Regressor"],
                n_runs=cfg["MC_dropout_runs"],
                device=device,
            )

            candidates_uncerts_after.append(temp_uncert)
            data_idxs_after.append(temp_idx)
        
        candidates_uncert_after = pt.reorder_arrays(
            candidates_uncerts_after,
            data_idxs_after,
            data_idx
            )
        
        candidates_uncert_after = np.array(candidates_uncert_after)
        candidates_uncert_after = np.sum(candidates_uncert_after, axis = 0)

        logging.info("Change in uncertainty for most uncertain data points:")

        output_dict["d_novel_uncert"].append(
            pt.uncertainty_change(
                x=candidates_uncert_before,
                y=candidates_uncert_after,
                plot_title="Novel data",
                iteration=i,
                save_path=save_dest,
            )
        )


        output_dict["novel_uncert_before"].append(candidates_uncert_before)
        output_dict["novel_uncert_after"].append(candidates_uncert_after)

        output_dict["holdout_pred_before"].append(
            holdout_pred_before
        )  # these two are probably the only important ones
        output_dict["holdout_pred_after"].append(holdout_pred_after)
        output_dict["holdout_ground_truth"].append(holdout_set.target)
        output_dict["retrain_losses"].append(train_losses)
        output_dict["retrain_val_losses"].append(val_losses)
        output_dict["retrain_losses_unscaled"].append(train_losses_unscaled)
        output_dict["retrain_val_losses_unscaled"].append(val_losses_unscaled)
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
        
 

        n_train = len(train_sample)
        output_dict["n_train_points"].append(n_train)

        # --- Save at end of iteration
        output_path = os.path.join(
            save_dest, f"pipeline_outputs_lam_{lam}_iteration_{i}.pkl"
        )
        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)
        
        for FLUX in FLUXES:
            regressor_path = os.path.join(save_dest, f"{FLUX}_regressor_lam_{lam}_iteration_{i}.pkl")
            torch.save(models[FLUX]["Regressor"].state_dict(), regressor_path)
        
        classifier_path = os.path.join(save_dest, f"{FLUXES[0]}_classifier_lam_{lam}_iteration_{i}.pkl")
        torch.save(models[FLUXES[0]]["Classifier"].state_dict(), classifier_path)
    return output_dict        


if __name__=='__main__':
    # add argument to pass config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", required=True)
    parser.add_argument("-o", "--output_dir", help="outputs directory", required=False, default=None)
    parser.add_argument("-p", "--plot_dir", help="plots directory", required=False, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if args.output_dir is not None:
        SAVE_PATHS["outputs"] = args.output_dir
    else:
        SAVE_PATHS["outputs"] = cfg["save_paths"]["outputs"]

    if args.plot_dir is not None:
        SAVE_PATHS["plots"] = args.plot_dir
    else:
        SAVE_PATHS["plots"] = cfg["save_paths"]["plots"]

    Nbootstraps = cfg['Nbootstraps']        
    lam = cfg["hyperparams"]["lambda"]
    model_size = cfg['hyperparams']['model_size']
    Ntrain = cfg["hyperparams"]["train_size"]
    Niter = cfg["iterations"]
    Ncand = cfg["hyperparams"]["candidate_size"]

    retrain = cfg["retrain_classifier"]
    if Nbootstraps>1:
        #cfg = np.repeat(cfg,Nbootstraps)
        seeds = np.arange(Nbootstraps).astype(int)
        inp = []
        for s in seeds:
            inp.append([s,cfg, SAVE_PATHS["outputs"], SAVE_PATHS["plots"]])
        with Pool(Nbootstraps) as p:            
            output = p.map(ALpipeline,inp)
        output = {'out':output}
        if args.output_dir is None:
            total = int(Ntrain+0.2*Ncand*0.25*Niter)  #--- assuming ITG (20%) and current strategy for the acquisition (upper quartile of uncertainty)
            output_dir = f"/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/{total}_{Ntrain}/"

        else:
            output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}bootstrapped_AL_lam_{lam}_{model_size}_classretrain_{retrain}_norescale.pkl","wb") as f:
            pickle.dump(output,f)                                
