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
from multiprocessing import Pool

from torch.utils.data import DataLoader
from scripts.utils import train_keys
import yaml
import pickle
import numpy as np

import torch
from Models import Classifier, ITGDatasetDF, Regressor
import argparse
import pandas as pd
import copy

def get_data(self,cfg, scale=True):
    PATHS = cfg["data"]
    FLUX = cfg["flux"]
    train_dataset, eval_dataset, test_dataset, scaler = pt.prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], fluxes=FLUX, samplesize_debug=0.1, scale=scale
) 
    return train_dataset, eval_dataset, test_dataset,scaler

def CLPipeline(arg):
    seed = arg[0]
    config_tasks = arg[1]
    CL_mode = arg[2]
    save_plots_path = arg[3]
    save_outputs_path = arg[4]
    mem_replay = arg[5]
    lambda_task = arg[6]
    test_data = {}   # --- saves test data for each task
    forgetting = {}  # --- saves test MSE on previous tasks with updated model
    outputs = {} # --- saves all output losses for each task
    # --- all of this is so ugly it makes me ashamed, but no time for polishing now
    # AL pipeline should be a class that initialises the models by training them the first time, then can be updated with new data (data is a self.) and relative scaler (also a self)
    models = None
    print(config_tasks)
    for j,cfg in enumerate(config_tasks):
        print('=====================================')
        print('=====================================')
        print(f'Task number: {j}')
        print('=====================================')
        print('=====================================')        
        PATHS = cfg["data"]
        FLUX = cfg["flux"]        
        if CL_mode!= 'shrink_perturb':
            cfg['hyperparams']['lambda'] = 1
        if j==0:
            train_, val, test, scaler = pt.prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], fluxes=FLUX, samplesize_debug=1, scale=True
) 
            train = train_.sample(cfg['hyperparams']['train_size'])
            train_.remove(train.data.index)
            unlabelled_pool = train_
            val = val.sample(cfg['hyperparams']['valid_size'])
            test_new = test.sample(cfg['hyperparams']['test_size']) 
            save = copy.deepcopy(test_new)
            save.scale(scaler, unscale=True)
            test_data.update({f'task{j}': save})
            first_iter = True
        else:
            train = train.sample(cfg['hyperparams']['train_size']*mem_replay)  # resample training set ToDo:======>>>>> Memory Replay Size (read paper to get a feel for it)
            scaler = StandardScaler()
            scaler.fit_transform(train.data.drop(["stable_label","index"], axis=1))
            train.scale(scaler)
            for flux in models.keys():
                models[flux]['Regressor'].scaler = scaler            
            train_new, eval_new, test_new, _ = pt.prepare_data(
    PATHS["train"], PATHS["validation"], PATHS["test"], fluxes=FLUX, samplesize_debug=1, scale=False
) 
            test_new = test_new.sample(cfg['hyperparams']['test_size'])
            save = copy.deepcopy(test_new)
            test_data.update({f'task{j}': save })  # --- save UNSCALED data for future testing
            test_new.scale(scaler)
            test.add(test_new) #--- assuming we have validation and testing - this is not always true in practice. In fact, in a real case we have to keep updating them
            eval_new.scale(scaler)   
            val_new = eval_new.sample(cfg['hyperparams']['valid_size'])
            val.add(val_new)   # val and test have been scaled in the AL pipeline already
            unlabelled_pool = train_new
            first_iter = False
            
        inp = [seed,cfg,save_outputs_path,save_plots_path, {'scaler':scaler,'train':train,'val':val,'test':test,'unlabelled':unlabelled_pool}, models, first_iter]

            # ---  train dataset is augmented each time in the pipeline, without removing old data
        inp = {'run_mode':'CL','cfg':inp}
        train, val, test, output_dict, scaler, models = ALpipeline(inp)  # --- this is really ugly as it is a recursvie thing and so should really be in a class itself, but not time to polish
        outputs.update({f'task{j}':output_dict})
        if CL_mode == 'shrink_perturb':
            for flux in models.keys():
                for model in models[flux].keys():
                    models[flux][model].shrink_perturb(lam=lambda_task, scale=0.01, loc=0.0)
        #elif self.cl_mode == 'EWC':
        #    self.EWC()
        #elif self.cl_mode == 'OGD': # ToDo =========>>> implement other frameworks if needed
        #    self.other()
        else: # --- Few-shot transfer learning, we don't care about forgetting. Note: 
            pass

        
        for k in test_data.keys():
            test_data[k].scale(scaler)   # --- scale all test data saved so far with current scaler
            _, regr_test_loss,regr_test_loss_unscaled, regr_test_loss_unscaled_norm = models[cfg['flux'][0]]['Regressor'].predict(test_data[k], unscale=True) # ToDo====>>> generalise to two outputs
            _, class_test_loss = models[cfg['flux'][0]]['Classifier'].predict(test_data[k])
            forgetting.update({f'regression_{k}_model{j}':{'regr_test_loss':regr_test_loss,'regr_test_loss_unscaled':regr_test_loss_unscaled, 'regr_test_loss_unscaled_norm': regr_test_loss_unscaled_norm}})
            forgetting.update({f'classification_{k}_model{j}':class_test_loss})
            test_data[k].scale(scaler, unscale=True) # --- unscale data for future passes


    outputs = {'outputs':outputs, 'forgetting':forgetting}
    return outputs




if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", required=True)
    parser.add_argument("-o", "--output_dir", help="outputs directory", required=False, default=None)
    parser.add_argument("-p", "--plot_dir", help="plots directory", required=False, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


    # TODO ========>>>>>>>>> fix output and plots paths, perhaps each task should have its own. Pass the outputs for each task in .run()
   # SAVE_PATHS = {}
   # if args.output_dir is not None:
   #     SAVE_PATHS["outputs"] = args.output_dir
   # else:
   #     SAVE_PATHS["outputs"] = cfg["save_paths"]["outputs"]à#

   # if args.plot_dir is not None:
   #     SAVE_PATHS["plots"] = args.plot_dir
   # else:
   #     SAVE_PATHS["plots"] = cfg["save_paths"]["plots"]
    args = parser.parse_args()

    #ToDo: ===>>> add possibility to declare paths from CLI

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    task1 = cfg['task1']            
    task2 = cfg['task2']
    task3 = cfg['task3']
    task4 = cfg['task4']
    common = cfg['common']
    acquisition = common['acquisition']
    classretrain = common['retrain_classifier']
    Nbootstraps = common['Nbootstraps']
    task1.update(common)
    task2.update(common)
    task3.update(common)
    task4.update(common)

    config_tasks = [task1,task2,task3,task4]
    CL_mode = cfg['CL_method']
    mem_replay = cfg['mem_replay']
    lambda_task = cfg['lambda_task']
    if CL_mode != 'shrink_perturb':
        lam = 1

    seeds = [np.random.randint(0,2**32-1) for i in range(Nbootstraps)]
    inp = []
    for s in seeds:
        inp.append([s, config_tasks,CL_mode,f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/', 
        f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/', mem_replay, lambda_task])

    with Pool(Nbootstraps) as p:
        outputs = p.map(CLPipeline,inp)
   
    with open(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/bootstrapped_CL_{CL_mode}_lam_{lambda_task}_{acquisition}_replaysize_{mem_replay}.pkl', 'wb') as f:
        pkl.dump(outputs, f)

