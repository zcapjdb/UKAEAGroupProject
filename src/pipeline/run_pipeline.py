from email.policy import default
import coloredlogs, verboselogs, logging
from multiprocessing import Pool
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pylab as plt
import copy

import random
import pipeline.pipeline_tools as pt
import pipeline.Models as md

from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import torch
from pipeline.Models import Classifier, Regressor
import argparse
import pandas as pd
import numpy as np 
from multiprocessing import Pool

def get_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)     
    seed_byteTensor = torch.random.get_rng_state()
    return seed_byteTensor

def ALpipeline(cfg):

    # how cfg should be declared in Clpipeline or AL pipeline: cfg = {'run_mode':run_mode,'cfg':config}, config will be a list for AL bootstrap, and a dict for AL not bootstrap; it's always a dict for CL
    if cfg['run_mode'] == 'AL':  #ToDo:=====> add to config
        run_mode = 'AL'
        if not isinstance(cfg['cfg'],dict):
            SAVE_PATHS = {}
            cfg = cfg['cfg']
            seed = cfg[0] 
            SAVE_PATHS["outputs"] = cfg[2]
            SAVE_PATHS["plots"] = cfg[3]
            cfg = cfg[1]        
            seed_byteTensor = get_seeds(seed)           
            first_CL_iter = False
        else:
            cfg = cfg['cfg']
    elif cfg['run_mode'] == 'CL': 
        run_mode = 'CL'
        if not isinstance(cfg['cfg'],dict):  

            cfg = cfg['cfg']
            seed = cfg[0]
            SAVE_PATHS = {}
            SAVE_PATHS["outputs"] = cfg[2]  # todo=====> figure out whether save paths should be like this
            SAVE_PATHS["plots"] = cfg[3]            
            scaler = cfg[4]['scaler']
            train_sample = cfg[4]['train']['train_regr']
            train_classifier = cfg[4]['train']['train_class']
            valid_classifier = cfg[4]['val']['val_class']
            valid_dataset = cfg[4]['val']['val_regr']
            holdout_classifier = cfg[4]['test']['test_class']
            holdout_set = cfg[4]['test']['test_regr']
            plot_sample = holdout_set
            unlabelled_pool = cfg[4]['unlabelled']   
            models = cfg[5]
            first_CL_iter = cfg[6]
            j = cfg[7]
            cfg = cfg[1]
            seed_byteTensor = get_seeds(seed)
        else:
            raise ValueError('for CL a list is always expected')

    # Create logger object for use in pipeline
    verboselogs.install()
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")  # cfg["logging_level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # --- enforced due to cluster issues

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
    CLASSIFIER = True  #CLASSIFIER = cfg["use_classifier"]
    dropout = cfg["hyperparams"]["dropout"]
    keep_prob = cfg['keep_prob']
    from_scratch = cfg['from_scratch']



    # Dictionary to store results of the classifier and regressor for later use
    output_dict = pt.output_dict

    # --- Set up saving
    save_dest = os.path.join(SAVE_PATHS["outputs"], FLUXES[0])
    if not os.path.exists(save_dest):
        os.makedirs(save_dest, exist_ok=True)

    logging.info("Saving to {}".format(save_dest))

    # save copy of yaml file used for reproducibility
    with open(os.path.join(save_dest, "config.yaml"), "w") as f:
        out = yaml.dump(cfg, f, default_flow_style=False)


    # --------------------------------------------- Load data ----------------------------------------------------------
    if run_mode == 'AL':
        train_sample, train_classifier, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, _, scaler = pt.get_data(cfg,j=0)
    elif run_mode == 'CL':
        print('RUNNING IN CL MODE')
        pass # --- data is passed from CL pipeline, see start of the function
    

    buffer_size = 0
    classifier_buffer = None

    # ------------------------------------------- Load or train first models ------------------------------------------
    if run_mode == 'AL'  or (run_mode=='CL' and first_CL_iter):
        models = {f:{} for f in FLUXES}
        
        for model in PRETRAINED:
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

                        models[FLUX][model] = Regressor(
                            device=device, 
                            scaler=scaler, 
                            flux=FLUX, 
                            dropout=dropout,
                            model_size=cfg['hyperparams']['model_size']
                            )
        
                        models[FLUX][model], losses = md.train_model(
                            models[FLUX][model],
                            train_sample,
                            valid_dataset,
                            save_path=PRETRAINED[model][FLUX]["save_path"],
                            epochs=cfg["train_epochs"],
                            patience=cfg["train_patience"],
                        )
                        
                        train_loss, train_loss_unscaled, valid_loss, valid_loss_unscaled = losses
                        output_dict["train_loss_init"].append(train_loss)

                            # if model == "Classifier":  --- not used currently
                            #    losses, train_accuracy, validation_losses, val_accuracy = losses
      

            elif CLASSIFIER:
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
                            train_classifier,
                            valid_classifier,
                            save_path=PRETRAINED[model][FLUXES[0]]["save_path"],
                            epochs=cfg["train_epochs"],
                            patience=cfg["train_patience"],
                        )

        for FLUX in FLUXES:
            logging.debug(f"Models for {FLUX} : {models[FLUX].keys()}")

        # ---- Losses before the pipeline starts #TODO: Fix output dict to be able to handle multiple variables
        for FLUX in FLUXES:
            logging.info(f"Test loss for {FLUX} before pipeline:")
            _, holdout_loss, holdout_loss_unscaled, holdout_loss_unscaled_norm = models[FLUX]["Regressor"].predict(holdout_set, unscale=True, mean=None )
            logging.info(f"Holdout Loss: {holdout_loss}")
            logging.info(f"Holdout Loss Unscaled: {holdout_loss_unscaled}")
            output_dict["test_loss_init"].append(holdout_loss)
            output_dict["test_loss_init_unscaled"].append(holdout_loss_unscaled)
        if CLASSIFIER:   
            _, holdout_class_losses = models[FLUXES[0]]["Classifier"].predict(holdout_classifier) 
            output_dict['class_test_acc_init'].append(holdout_class_losses[1])
            output_dict['class_precision_init'].append(holdout_class_losses[2])
            output_dict['class_recall_init'].append(holdout_class_losses[3])
            output_dict['class_f1_init'].append(holdout_class_losses[4])
            output_dict['class_auc_init'].append(holdout_class_losses[5])
    else:
        pass # --- models are passed from CL pipeline            

    # Create logger object for use in pipeline
    verboselogs.install()
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")

    if len(train_sample) > 100_000:
        logger.warning(
            "Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data"
        )

        # Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50

    holdout_plot = copy.deepcopy(holdout_set)
    holdout_plot.scale(scaler, unscale=True)
    for i in range(cfg["iterations"]):
        logging.info(f"Iteration: {i+1}\n")
        logging.info('unlabelled pool: ',len(unlabelled_pool))
        epochs = cfg["initial_epochs"] * (i + 1)
        
        # --- at each iteration the labelled pool is updated - 10_000 samples are taken out, the most uncertain are put back in
        candidates = unlabelled_pool.sample(candidate_size)
        # --- remove the sampled data points from the dataset
        unlabelled_pool.remove(candidates.data.index)

        # --- See Issue #37 --- candidates are only those that the classifier selects as unstable.
        if CLASSIFIER: # if we are using the classifier use it to remove stable labels
            candidates = pt.select_unstable_data(
                candidates,
                batch_size=batch_size,
                classifier=models[FLUXES[0]]["Classifier"],
                device=device,
            ) 
            logging.info(f"{len(candidates)} candidates selected")


        # ---  get most uncertain candidate inputs as decided by regressor   --- NEW AL FRAMEWORK GOES HERE
        candidates_uncerts, data_idxs = [], []

        for FLUX in FLUXES:
            temp_uncert, temp_idx = pt.get_uncertainty(
                candidates,
                models[FLUX]["Regressor"],
                n_runs=cfg["MC_dropout_runs"],
                device=device,
                iter_num=i,
        #        plot=True
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
            keep = keep_prob,
            model=models[FLUX]["Regressor"],
            acquisition=cfg["acquisition"],
        )

       # pt.plot_TSNE(candidates, train_sample, iter_num=i)
        
        logging.debug(f"Number of most uncertain {len(data_idx)}")
        prediction_candidates_before = []
        for FLUX in FLUXES:
            prediction_candidates_before.append(
                models[FLUX]["Regressor"].predict(candidates, unscale=False)
            )

        # =================== >>>>>>>>>> Here goes the Qualikiz acquisition <<<<<<<<<<<<< ==================

        # I am assuming here that the acquisition will modify the candidates dataset.
        # ToDo - May need to modify dataset to account for real data points that won't have real labels. Use dummy labels until acquisition is done?

        # ToDo ===========>>>> Now we do have the labels because Qualikiz gave them to us!  Need to discard misclassified data from enriched_train_loader, and retrain the classifier if buffer_size is big enough
        # check for misclassified data in candidates and add them to the buffer
        if CLASSIFIER:
            candidates, misclassified_data, num_misclassified, data_idx, candidates_uncert_before = pt.check_for_misclassified_data(
                candidates, 
                uncertainty=candidates_uncert_before, 
                indices=data_idx
                )
            buffer_size += num_misclassified
            if classifier_buffer is None:
                classifier_buffer = md.ITGDatasetDF(
                    misclassified_data, FLUXES[0], keep_index=True
                )             
            else:
                misclassified_data = md.ITGDatasetDF(
                    misclassified_data, FLUXES[0], keep_index=True
                )                             
                classifier_buffer.add(misclassified_data)   
            logging.info(f"Misclassified data: {num_misclassified}")
            logging.info(f"Total Buffer size: {buffer_size}")
        
        elif len(FLUXES)>1:
            # at this stage we know which of the inputs give a stable label or not so we can drop the stable values
            #  
            pass 

        # --- set up retraining by rescaling all points according to new training data --------------------
       # --- unscale all datasets, 
        candidates.scale(scaler, unscale=True)
        train_sample.scale(scaler, unscale=True)
        train_classifier.scale(scaler,unscale=True)
        unlabelled_pool.scale(scaler, unscale=True)
        valid_dataset.scale(scaler, unscale=True)
        valid_classifier.scale(scaler,unscale=True)
        holdout_set.scale(scaler, unscale=True)
        holdout_classifier.scale(scaler, unscale=True)

        if classifier_buffer is not None:
            print('len classifier buffer at unscale:',len(classifier_buffer))
            if len(classifier_buffer)>0:
                    classifier_buffer.scale(scaler, unscale=True)

        # --- train data is enriched by new unstable candidate points
        #total = cfg['hyperparams']['train_size'] + cfg['hyperparams']['valid_size'] + cfg['hyperparams']['test_size']
        #train_add = candidates.sample(int(len(candidates)*cfg['hyperparams']['train_size']/total))
        #candidates.remove(train_add.data.index)
        #valid_add = candidates.sample(int(len(candidates)*cfg['hyperparams']['valid_size']/total))
        #candidates.remove(valid_add.data.index)
        #test_add = candidates.sample(int(len(candidates)*cfg['hyperparams']['test_size']/total))
        #candidates.remove(test_add.data.index)

      #  logging.info(f"Enriching training data with {len(train_add)} new points")

        train_sample.add(candidates)
        #train_sample.add(train_add)
        #valid_dataset.add(valid_add)
        #holdout_set.add(test_add)
        # ---  compute the mean for the loss function
        mean_train = np.mean(train_sample.data['efiitg_gb']) # ---- ToDo =====>>>>> need to upgrade to two outputs

       # bins = np.arange(-50,150)
       # plt.hist(candidates.data['efiitg_gb'], bins=bins, color='orange', histtype='step', label='candidates', density=True,lw=2)
       # plt.hist(train_sample.data['efiitg_gb'],bins=bins, color='blue', alpha=0.2, label='train', density=True)
       # plt.hist(holdout_plot.data['efiitg_gb'], bins=bins, color='red',histtype='step',label='test', density=True,lw=2)
       # plt.title(f'iteration number {i}')       

        # --- get new scaler from enriched training set, rescale them with new scaler
        scaler = StandardScaler()
        scaler.fit(train_sample.data.drop(["stable_label","index"], axis=1))
        train_sample.scale(scaler)
        candidates.scale(scaler)
        train_classifier.scale(scaler)
        unlabelled_pool.scale(scaler)
        valid_dataset.scale(scaler)
        valid_classifier.scale(scaler)
        holdout_set.scale(scaler)
        holdout_classifier.scale(scaler)
        if classifier_buffer is not None:
            if len(classifier_buffer)>0:            
                classifier_buffer.scale(scaler)
                
             
        # --- update scaler in the models
        for FLUX in FLUXES:
            for model in PRETRAINED:
                models[FLUX][model].scaler = scaler
        # --- Classifier retraining:
        if cfg["retrain_classifier"]:
            if from_scratch:
                torch.random.set_rng_state(seed_byteTensor)
                classifier = Classifier(device)
                losses, accs = pt.retrain_classifier(
                    classifier_buffer,
                    train_classifier,
                    valid_classifier,
                    models[FLUXES[0]]["Classifier"],
                    batch_size=batch_size,
                    epochs=epochs,
                    lam=lam,
                    patience=cfg["patience"],
                )
                classifier_buffer = None

                # record which iterations the classifier was retrained on
            else:                
                if buffer_size >= cfg["hyperparams"]["buffer_size"]:
                    #classifier_buffer.scale(scaler)
                    train_classifier.add(classifier_buffer)
                    logging.info(f"Buffer full, retraining classifier with {len(classifier_buffer)} points")
                    # retrain the classifier on the misclassified points
                    losses, accs = pt.retrain_classifier(
                        classifier_buffer,
                        train_classifier,
                        valid_classifier,
                        models[FLUXES[0]]["Classifier"],
                        batch_size=batch_size,
                        epochs=epochs,
                        lam=lam,
                        patience=cfg["patience"],
                    )
                    classifier_buffer = None
                    buffer_size = 0

                    #output_dict["class_train_loss"].append(losses[0])
                    #output_dict["class_val_loss"].append(losses[1])
                    #output_dict["class_missed_loss"].append(losses[2])
                    #output_dict["class_train_acc"].append(accs[0])
                    #output_dict["class_val_acc"].append(accs[1])
                    #output_dict["class_missed_acc"].append(accs[2])

        _, holdout_class_losses = models[FLUXES[0]]["Classifier"].predict(holdout_classifier) 
        output_dict['holdout_class_loss'].append(holdout_class_losses[0])
        output_dict['holdout_class_acc'].append(holdout_class_losses[1])
        output_dict['holdout_class_precision'].append(holdout_class_losses[2])
        output_dict['holdout_class_recall'].append(holdout_class_losses[3])
        output_dict['holdout_class_f1'].append(holdout_class_losses[4])
        output_dict['holdout_class_auc'].append(holdout_class_losses[5])

        # record which iterations the classifier was retrained on
        output_dict['class_retrain_iterations'].append(i)

                # reset buffer

        holdout_pred_before = []
        train_losses, val_losses = [], []
        train_losses_unscaled, val_losses_unscaled = [], []
        holdout_pred_after, holdout_loss, holdout_loss_unscaled,holdout_loss_unscaled_norm = [], [], [],[]
        for FLUX in FLUXES:
            # --- validation on holdout set before regressor is retrained (this is what's needed for AL)
#            preds, _ = models[FLUX]["Regressor"].predict(holdout_set, unscale=False)
#            holdout_pred_before.append(preds)

            if from_scratch: 
                torch.random.set_rng_state(seed_byteTensor)
                models[FLUX]['Regressor'] = Regressor(
                            device=device, 
                            scaler=scaler, 
                            flux=FLUX, 
                            dropout=dropout,
                            model_size=cfg['hyperparams']['model_size']
                            )
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
            hold_pred_after, hold_loss, hold_loss_unscaled, hold_loss_unscaled_norm = models[FLUX]["Regressor"].predict(holdout_set, unscale=True, mean=mean_train)
            holdout_pred_after.append(hold_pred_after)
            holdout_loss.append(hold_loss)
            holdout_loss_unscaled.append(hold_loss_unscaled)
            holdout_loss_unscaled_norm.append(hold_loss_unscaled_norm)
            logging.info(f"{FLUX} test loss: {hold_loss}")
            logging.info(f"{FLUX} test loss unscaled: {hold_loss_unscaled}")
            logging.info(f"{FLUX} test loss unscaled norm: {hold_loss_unscaled_norm}")
  
           # plt.hist(hold_pred_after, bins=bins, color='magenta',histtype='step',label='pred', density=True,lw=2)
           # plt.title(f'iter {i}, task {j}')
           # plt.legend()
           # plt.savefig(f"debug/hist/CL/{cfg['acquisition']}/hist{i}_{j}_{cfg['acquisition']}.png")
           # plt.close()


           # bins = np.arange(0,3,0.1)
           # plt.scatter(candidates_uncert_before)
           # plt.ylabel('yhat')
           # plt.xlabel('y')
           # plt.plot(bins,bins, lw=4, color='red')
           # plt.savefig(f"/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/debug/uncert/{cfg['acquisition']}/scatter{i}_{j}_{cfg['acquisition']}.png")
           # plt.close()

 #       candidates_uncerts_after, data_idxs_after = [], []#

#        for FLUX in FLUXES:
#            temp_uncert, temp_idx = pt.get_uncertainty(
#                candidates,
#                models[FLUX]["Regressor"],
#                n_runs=cfg["MC_dropout_runs"],
#                device=device,
#            )

#            candidates_uncerts_after.append(temp_uncert)
#            data_idxs_after.append(temp_idx)
#            
#            candidates_uncert_after = pt.reorder_arrays(
#                candidates_uncerts_after,
#                data_idxs_after,
#                data_idx
#                )
            
        #    candidates_uncert_after = np.array(candidates_uncert_after)
        #    candidates_uncert_after = np.sum(candidates_uncert_after, axis = 0)
        #    pt.plot_scatter(
        #        candidates_uncert_before,
        #        candidates_uncert_after,
        #        "Novel data",
        #        i,
        #        f"/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/debug/uncert/")#

#            test_uncert, _ =  pt.get_uncertainty(
#                holdout_set,
#                models[FLUX]["Regressor"],
#                n_runs=cfg["MC_dropout_runs"],
#                device=device,
#            )##

#            fig, ax = plt.subplots(1,1)
#            ax.hist(test_uncert, bins=np.arange(0,0.5,0.01))
#            ax.axvline(np.median(test_uncert),color='red')
#            ax.set_xlabel('test uncert')
#            ax.set_title(f"iteration {i}")
#            fig.savefig(f'./debug/uncert/test_uncert{i}.png')
#            fig.clf()


            #logging.info("Change in uncertainty for most uncertain data points:")

#        output_dict["d_novel_uncert"].append(
#            pt.uncertainty_change(
#                x=candidates_uncert_before,
#                y=candidates_uncert_after,
#                plot_title="Novel data",
#                iteration=i,
#                save_path=save_dest,
#            )
#       )
#

#        output_dict["novel_uncert_before"].append(candidates_uncert_before)
#        output_dict["novel_uncert_after"].append(candidates_uncert_after)

#        output_dict["holdout_pred_before"].append(
#            holdout_pred_before
#        )  # these two are probably the only important ones
#        output_dict["holdout_pred_after"].append(holdout_pred_after)
#        output_dict["holdout_ground_truth"].append(holdout_set.target)
#        output_dict["retrain_losses"].append(train_losses)
#        output_dict["retrain_val_losses"].append(val_losses)
#        output_dict["retrain_losses_unscaled"].append(train_losses_unscaled)
#        output_dict["retrain_val_losses_unscaled"].append(val_losses_unscaled)
#        output_dict["post_test_loss"].append(holdout_loss)
        output_dict["post_test_loss_unscaled"].append(holdout_loss_unscaled)
#        output_dict["post_test_loss_unscaled_norm"].append(holdout_loss_unscaled_norm)
#        output_dict["scale_scaler"].append(models[FLUXES[0]]["Regressor"].scaler.scale_)
#        output_dict["mean_scaler"].append(models[FLUXES[0]]["Regressor"].scaler.mean_)

#        try:
#            output_dict["mse_before"].append(
#                train_mse_before
#            )  # these three relate to the training MSE, probably not so useful to inspect
#            output_dict["mse_after"].append(train_mse_after)
#            output_dict["d_mse"].append(delta_mse)
#        except:
#            pass
        

        n_train = len(train_sample)
        output_dict["n_train_points"].append(n_train)
        logging.info(f"Number of training points at end of iteration {i + 1}: {n_train}")

    if run_mode == 'AL':   #we don't necessarily want this in CL for the moment
#        # --- Save at end of iteration
#        output_path = os.path.join(
#            save_dest, f"pipeline_outputs_lam_{lam}_iteration_{i}.pkl"
#        )
#        with open(output_path, "wb") as f:
#            pickle.dump(output_dict, f)
#        
#        for FLUX in FLUXES:
#            regressor_path = os.path.join(save_dest, f"{FLUX}_regressor_lam_{lam}_iteration_{i}.pkl")
#            torch.save(models[FLUX]["Regressor"].state_dict(), regressor_path)
#        
#        classifier_path = os.path.join(save_dest, f"{FLUXES[0]}_classifier_lam_{lam}_iteration_{i}.pkl")
#        torch.save(models[FLUXES[0]]["Classifier"].state_dict(), classifier_path)
        return output_dict

    else:
        return train_sample, train_classifier, valid_dataset, valid_classifier, holdout_set, holdout_classifier, output_dict, scaler, models # ugly


if __name__=='__main__':
    # add argument to pass config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", required=True)
    parser.add_argument("-o", "--output_dir", help="outputs directory", required=False, default=None)
    parser.add_argument("-p", "--plot_dir", help="plots directory", required=False, default=None)
    parser.add_argument("--no_classifier",default=None, action="store_false", required = False)
    parser.add_argument("--no_classifier_retrain", default=None, action="store_false", required= False)
    parser.add_argument("--train_size", default=None, required=False, type=int)
    parser.add_argument("--candidate_size", default=None, required=False, type=int)
    parser.add_argument("--acquisition", default=None, required=False)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    SAVE_PATHS = {}
    if args.output_dir is not None:
        SAVE_PATHS["outputs"] = args.output_dir
    else:
        SAVE_PATHS["outputs"] = cfg["save_paths"]["outputs"]

    if args.plot_dir is not None:
        SAVE_PATHS["plots"] = args.plot_dir
    else:
        SAVE_PATHS["plots"] = cfg["save_paths"]["plots"]
    
    if args.no_classifier is not None:
        cfg["use_classifier"] = args.no_classifier
    if args.no_classifier_retrain is not None: 
        cfg["retrain_classifier"] = args.no_classifier_retrain
    if args.train_size is not None: 
        cfg["hyperparams"]["train_size"] = args.train_size
    if args.candidate_size is not None: 
        cfg["hyperparams"]["candidate_size"] = args.candidate_size
        logging.debug(f"candidate_size: {cfg['hyperparams']['candidate_size']}")
    if args.acquisition is not None: 
        cfg["acquisition"] = args.acquisition


    Nbootstraps = cfg['Nbootstraps']        
    lam = cfg["hyperparams"]["lambda"]
    model_size = cfg['hyperparams']['model_size']
    Ntrain = cfg["hyperparams"]["train_size"]
    Niter = cfg["iterations"]
    Ncand = cfg["hyperparams"]["candidate_size"]
    keep = cfg['keep_prob']
    retrain = cfg["retrain_classifier"]
    from_scratch = cfg['from_scratch']
    acquisition =  cfg["acquisition"]

    if Nbootstraps>1:
        #cfg = np.repeat(cfg,Nbootstraps)
        seeds = [np.random.randint(0,2**32-1) for i in range(Nbootstraps)]
        inp = []
        for s in seeds:
            inp.append([s,cfg, SAVE_PATHS["outputs"], SAVE_PATHS["plots"]])
        cfg = [{'run_mode':'AL','cfg':inp[i]} for i in range(len(inp))]
        with Pool(Nbootstraps) as p:            
            output = p.map(ALpipeline,cfg)
    else:
        seed = np.random.randint(0,2**32-1)
        output = ALpipeline({'run_mode':'AL','cfg':[seed,cfg, SAVE_PATHS["outputs"], SAVE_PATHS["plots"]]})

    output = {'out':output}
    total = int(Ntrain+0.2*Ncand*0.25*Niter)  #--- assuming ITG (20%) and current strategy for the acquisition (upper quartile of uncertainty)

    if args.output_dir is None:
        total = int(Ntrain+0.2*Ncand*0.25*Niter)  #--- assuming ITG (20%) and current strategy for the acquisition (upper quartile of uncertainty)
        output_dir = f"/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/{total}_{Ntrain}/" # --- next time we should make sure we have consistent paths to avoid this

    else:
        output_dir = args.output_dir
    #output_dir = f"/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/{total}_{Ntrain}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f'Done. Saving to {output_dir}bootstrapped_AL_lam_{lam}_{acquisition}_classretrain_{retrain}_keepprob{keep}_fromscratch_{from_scratch}.pkl')
    with open(f"{output_dir}bootstrapped_AL_lam_{lam}_{acquisition}_classretrain_{retrain}_keepprob{keep}_fromscratch_{from_scratch}.pkl","wb") as f:
        pickle.dump(output,f)               
    print('Done.')