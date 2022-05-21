import coloredlogs, verboselogs, logging
import os
import copy
import pickle as pkl
from multiprocessing import Pool
#import comet_ml import Experiment
from sklearn.preprocessing import StandardScaler

import pipeline.pipeline_tools as pt
import pipeline.Models as md
from pipeline.run_pipeline import ALpipeline 

from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import torch
from Models import Classifier, ITGDatasetDF, Regressor
import argparse
import pandas as pd
import copy

class CLTaskManager:
    def __init__(self, config_tasks: list = None,  CL_mode: str = 'shrink_perturb', forget: str = None):
        self.config_tasks = config_tasks # --- List of config files for the different tasks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = Classifier(self.device)
        self.regressor = Regressor(self.device,scaler=None, flux=None)
        self.cl_mode = CL_mode
        self.forget_path = forget
        
    def get_data(self,cfg, scale=True):
        PATHS = cfg["data"]
        FLUX = cfg["flux"]
        train_dataset, eval_dataset, test_dataset, scaler = pt.prepare_data(
        PATHS["train"], PATHS["validation"], PATHS["test"], fluxes=FLUX, samplesize_debug=0.1, scale=scale
    ) 
        return train_dataset, eval_dataset, test_dataset,scaler

    def shrink_perturb(self,lam):
        self.classifier.shrink_perturb(lam=lam, scale=0.01, loc=0.0)
        self.regressor.shrink_perturb(lam=lam, scale=0.01, loc=0.0)     

    def EWC(self):
        raise ValueError('Not implemented')

    def OGD(self):
        raise ValueError('Not implemented')

    def run(self):
        models =  {'Classifier':self.classifier,'Regressor':self.regressor}
        test_data = {}
        forgetting = {}
        outputs = {}
        # --- all of this is so ugly it makes me ashamed, but no time for polishing now
        # AL pipeline should be a class that initialises the models by training them the first time, then can be updated with new data (data is a self.) and relative scaler (also a self)
        for j,cfg in enumerate(self.config_tasks):
            self.regressor.flux = cfg['flux'] 
            if self.cl_mode!= 'shrink_perturb':
                cfg['hyperparams']['lambda'] = 1
            if j==0:
                train, val, test, scaler = self.get_data(cfg,scale=True)  
                self.regressor.scaler = scaler
                test_new = test.sample(cfg['hyperparams']['test_size'])
                save = copy.deepcopy(test_new)
                save.scale(scaler, unscale=True)
                test_data.update({f'task{j}': save})
                print(test_data)
                inp = cfg
            else:
                train_new, eval_new, test_new, _ = self.get_data(cfg,scale=False)                 
                
                
                test_new = test_new.sample(cfg['hyperparams']['test_size'])
                save = copy.deepcopy(test_new)
                test_data.update({f'task{j}': save })
                test_new.scale(scaler)

                eval_new.scale(scaler)   
                val_new = eval_new.sample(cfg['hyperparams']['valid_size'])
                eval_new.remove(val_new.data.index)
                unlabelled_pool = eval_new
                
                val.add(val_new)   # val and test have been scaled in the AL pipeline already
                
                
                test.add(test_new) #--- assuming we have validation and testing - this is not always true in practice. In fact, in a real case we have to keep updating them

                inp = [0,cfg,j,{'scaler':scaler,'train':train,'val':val,'test':test,'unlabelled':unlabelled_pool}, models]

            

            # ToDo: =====>> think a bit better about the scaler!! still use the scaler for the first task??
             # ---  train dataset is augmented each time, without removing old data
            train, val, test, output_dict, scaler, models = ALpipeline(inp)
            outputs.update({f'task{j}':output_dict})
            # ToDo: ====>> instead, load each model from task 1 and retrain from same point on task 2 based on the various strategies
            if self.cl_mode == 'shrink_perturb':
                self.shrink_perturb(cfg['hyperparams']['lambda'])
            elif self.cl_mode == 'EWC':
                self.EWC()
            elif self.cl_mode == 'OGD': # ToDo =========>>> implement other frameworks
                self.other()
            else: # --- Few-shot transfer learning, we don't care about forgetting
                pass

            
            for k in test_data.keys():
                test_data[k].scale(scaler)
                _, regr_test_loss = models[cfg['flux'][0]]['Regressor'].predict(test_data[k])
                _, class_test_loss = models[cfg['flux'][0]]['Classifier'].predict(test_data[k])
                forgetting.update({f'regression_{k}_model{j}':regr_test_loss})
                forgetting.update({f'classification_{k}_model{j}':class_test_loss})
                test_data[k].scale(scaler, unscale=True)

        with open(self.forget_path, "wb") as f:
            pickle.dump(forgetting, f)

        return outputs




def ALpipeline_old(cfg: dict, models: dict, j: str, device: torch.device, cl_mode: str, scaler: StandardScaler,
            train_dataset: ITGDatasetDF,
            eval_dataset: ITGDatasetDF,
            test_dataset: ITGDatasetDF) -> None:
    # Create logger object for use in pipeline
    verboselogs.install()
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")#cfg["logging_level"])


    #comet_project_name = "AL-pipeline"
    #experiment = Experiment(project_name = comet_project_name)


    # Logging levels, DEBUG = 10, VERBOSE = 15, INFO = 20, NOTICE = 25, WARNING = 30, SUCCESS = 35, ERROR = 40, CRITICAL = 50

    FLUX = cfg["flux"]
    PRETRAINED = cfg["pretrained"]
    PATHS = cfg["data"]
    SAVE_PATHS = cfg["save_paths"]
    lamb = cfg["hyperparams"]["lambda"]
    lam = 1 # TodO: ======> perhaps also shrink perturb on AL pipeline, but different lambda?
    train_size = cfg["hyperparams"]["train_size"]
    valid_size = cfg["hyperparams"]["valid_size"]
    test_size = cfg["hyperparams"]["test_size"]
    batch_size = cfg["hyperparams"]["batch_size"]
    candidate_size = cfg["hyperparams"]["candidate_size"]
    # Dictionary to store results of the classifier and regressor for later use
    output_dict = pt.output_dict


    # To Do:  explore candidate_size hyperparam, explore architecture, validation loss shouldn't be used as test loss
    # --- holdout set is from the test set
    plot_sample = test_dataset.sample(test_size)  # Holdout dataset
    holdout_set = plot_sample 
    #holdout_set = pt.pandas_to_numpy_data(plot_sample) # Holdout set, remaining validation is unlabeled pool
    # --- validation set is fixed and from the evaluation
    valid_dataset = eval_dataset.sample(valid_size) # validation set
    # --- unlabelled pool is from the evaluation set minus the validation set (note, I'm not using "validation" and "evaluation" as synonyms)
    eval_dataset.remove(valid_dataset.data.index) # 
    unlabelled_pool = eval_dataset

    models['Regressor'].scaler = scaler #ToDo : =============>>>> use proper setter
    # --- Set up saving
    save_dest = os.path.join(SAVE_PATHS["outputs"], FLUX)
    if not os.path.exists(save_dest): os.makedirs(save_dest)
    if not os.path.exists(SAVE_PATHS["outputs"]):
        os.makedirs(SAVE_PATHS["outputs"])

    # ------------------------------------------- Load or train first models ------------------------------------------

    for model in models.keys():

        if j == 0:  # --- Retrain from scratch only for first iteration, then AL on the other task
                   
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
                output_dict["train_loss_init"].append(train_loss)
            if model == "Classifier":  
                train_loss, train_accuracy, validation_losses, val_accuracy = losses
                output_dict['class_train_loss_init'].append(train_loss)
                output_dict["class_train_acc"].append(train_accuracy)
                
    # ---- Losses before the pipeline starts
    _, holdout_loss = models["Regressor"].predict(holdout_set)
    output_dict["test_loss_init"].append(holdout_loss)
    _, holdout_losses = models["Classifier"].predict(holdout_set)
    holdout_class_loss, holdout_class_acc = holdout_losses
    output_dict["class_test_loss_init"].append(holdout_class_loss)
    output_dict["class_test_acc_init"].append(holdout_class_acc)



#    if len(train_sample) > 100_000:
#        logger.warning("Training sample is larger than 100,000, if using a pretrained model make sure to use the same training data")


    logging.info(f"Training for lambda: {lam}")

    # ------------------------------------------------------- AL pipeline ---------------------------------------------
    classifier_buffer = []
    buffer_size = 0

    for i in range(cfg["iterations"]):
        logging.info(f"Iteration: {i+1}\n")
        if j>0: # --- overwrite evaluation and test data
            train_sample = train_dataset
            scaler = StandardScaler()
            scaler.fit_transform(train_sample.drop(["stable_label"], axis=1))  
            eval_dataset.scale(scaler)                 
            test_dataset.scale(scaler)

        if i != 0:
            # reset the output dictionary for each iteration
            output_dict = pt.output_dict

        # --- at each iteration the labelled pool is updated - 10_000 samples are taken out, the most uncertain are put back in
        candidates = unlabelled_pool.sample(candidate_size)  
        # --- remove the sampled data points from the dataset
        unlabelled_pool.remove(candidates.data.index)

        # --- See Issue #37 --- candidates are only those that the classifier selects as unstable.
        candidates = pt.select_unstable_data(
            candidates, batch_size=batch_size, classifier=models["Classifier"], device=device
        )   # It's not the classifier's job to say whether a point is stable or not. This happens at the end of the pipeline when we get the true labels by running Qualikiz.
        

        epochs = cfg["initial_epochs"] * (i + 1)

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
        #(
        #    train_sample_origin,   # --- why is it called different tha train_sample? Are they not exactly the same?
        #    train_uncert_before,
        #    train_uncert_idx,
        #) = pt.regressor_uncertainty(
        #    train_sample,
        #    models["Regressor"],
        #    n_runs=cfg["MC_dropout_runs"],
        #    train_data=True,
        #)

        #prediction_train_origin, loss_train_origin = models["Regressor"].predict(train_sample_origin)

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
                misclassified_dataset = md.ITGDatasetDF(misclassified, FLUX, keep_index=True)

                logging.info("Buffer full, retraining classifier")
                # retrain the classifier on the misclassified points 
                losses, accs = pt.retrain_classifier(
                    misclassified_dataset,
                    train_sample,
                    holdout_set,
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

        # ---  get predictions for enriched train sample before retraining (perhaps not useful?)
        prediction_before, _ = models["Regressor"].predict(train_sample)
        
        # --- validation on holdout set before regressor is retrained (this is what's needed for AL)
        holdout_pred_before, _ = models["Regressor"].predict(holdout_set) 

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


        # --- predictions for the enriched train sample after (is that really needed?)
        enriched_train_prediction_after, _ = models["Regressor"].predict(train_sample)
        # --- validation on holdout set after regressor is retrained
        logging.info("Running prediction on validation data set")
        holdout_pred_after,holdout_loss, holdout_loss_unscaled = models["Regressor"].predict(holdout_set,unscale=True)   

        _, candidates_uncert_after, _, _ = pt.regressor_uncertainty(
            candidates,
            models["Regressor"],
            n_runs=cfg["MC_dropout_runs"],
            keep=cfg["keep_prob"],
            order_idx=data_idx,
        ) # --- uncertainty of newly added points
        #_, train_uncert_after, _ = pt.regressor_uncertainty(
        #    train_sample_origin,
        #    models["Regressor"],
        #    n_runs=cfg["MC_dropout_runs"],
        #    order_idx=train_uncert_idx,
        #    train_data=True,
        #) # --- uncertainty on first training set before points were added (is that really needed?)

        logging.info("Change in uncertainty for most uncertain data points:")
        
        output_dict["d_novel_uncert"].append(
            pt.uncertainty_change(
                x=candidates_uncert_before,
                y=candidates_uncert_after,
                plot_title='Novel data',
                iteration=i,
                save_path=save_dest
            )
        )

        output_dict["novel_uncert_before"].append(candidates_uncert_before)
        output_dict["novel_uncert_after"].append(candidates_uncert_after)

        #logging.info("Change in uncertainty for training data:")
        #output_dict["d_uncert"].append(
        #    pt.uncertainty_change(
        #        x=train_uncert_before,
        #        y=train_uncert_after,
        #        plot_title='Train data',
        #        iteration=i,
        #        save_path=save_dest
        #    )
        #)

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
                train_sample,
                uncertainties=[train_uncert_before, train_uncert_after],
                data="novel",
                save_path = SAVE_PATHS["plots"], 
                iteration=i,
                lam = lam
            )
        except:
            logging.debug("pt.mse_change failed, whatever")

        n_train = len(train_sample)
        output_dict["holdout_pred_before"].append(holdout_pred_before) # these two are probably the only important ones
        output_dict["holdout_pred_after"].append(holdout_pred_after)
        output_dict["holdout_ground_truth"].append(holdout_set.target)
        output_dict["retrain_losses"].append(train_loss)
        output_dict["retrain_test_losses"].append(test_loss)
        output_dict["post_test_loss"].append(holdout_loss)
        output_dict["post_test_loss_unscaled"].append(holdout_loss_unscaled)

        try:
            output_dict["mse_before"].append(train_mse_before) # these three relate to the training MSE, probably not so useful to inspect 
            output_dict["mse_after"].append(train_mse_after)
            output_dict["d_mse"].append(delta_mse)
        except:
            pass
        output_dict["n_train_points"].append(n_train)

        # --- Save at end of iteration
        output_path = os.path.join(save_dest, f"pipeline_outputs_{cl_mode}_lam_{lamb}_iteration_{i}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)
        regressor_path = os.path.join(save_dest, f"regressor_lam_{cl_mode}_{lamb}_iteration_{i}.pkl")
        torch.save(models["Regressor"].state_dict(), regressor_path)
        classifier_path = os.path.join(save_dest, f"classifier_lam_{cl_mode}_{lamb}_iteration_{i}.pkl")
        torch.save(models["Classifier"].state_dict(), classifier_path)

        if j>0: # --- overwrite evaluation and test data
            train_sample = train_dataset
        else:
            train_sample = train_dataset.sample(train_size) 
            scaler = StandardScaler()
            scaler.fit_transform(train_sample.drop(["stable_label"], axis=1))  
            valid_dataset.scale(scaler)                 
            test.scale(scaler)
    return  train_sample, holdout_set, output_dict



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    task1 = cfg['task1']            
    task2 = cfg['task2']
    task3 = cfg['task3']
    task4 = cfg['task4']
    common = cfg['common']
    acquisition = common['acquisition']
    classretrain = common['retrain_classifier']

    task1.update(common)
    task2.update(common)
    task3.update(common)
    task4.update(common)

    config_tasks = [task1,task2,task3,task4]
    CL_mode = cfg['CL_method']
    lam = cfg['common']['hyperparams']['lambda']
    if CL_mode != 'shrink_perturb':
        lam = 1
    # ToDo =====> add following to cfg
    #save_path = /home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject 

    PL = CLTaskManager(config_tasks, CL_mode, forget=f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/forgetting_{CL_mode}_lam_{lam}_{acquisition}_{classretrain}_sigma0.05.pkl')

    outputs = PL.run()
    outputs = {'out':outputs}
    with open(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/bootstrapped_CL_{CL_mode}_lam_{lam}_{acquisition}_{classretrain}_sigma0.05.pkl', 'wb') as f:
        pkl.dump(outputs, f)

