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



    
def downsample(cfg,data,mem_replay):
    print('==============')
    print('==============')
    print('==============')
    print('==============')
    print('==============')
    print('==============')
    print('before', cfg['hyperparams']['train_size'],cfg['hyperparams']['valid_size'],cfg['hyperparams']['test_size'] )
    train_size = int(cfg['hyperparams']['train_size']*mem_replay)
    val_size = int(cfg['hyperparams']['valid_size']*mem_replay)   # TODO replace with holdout size
    test_size = int(cfg['hyperparams']['test_size']*mem_replay)
    print('==============')
    print('==============')
    print('==============')
    print('==============')
    print('==============')
    print('==============')    
    print('after', train_size, val_size,test_size, 'holdout',len(data[4]))
    train_sample = data[0].sample(train_size)
    train_classifier = data[1].sample(train_size)
    valid_dataset =  data[2].sample(val_size)
    valid_classifier =  data[3].sample(val_size)
    holdout_set =  data[4].sample(int(len(data[4])*mem_replay))
    holdout_classifier =  data[5].sample(int(len(data[5])*mem_replay))
    return train_sample, train_classifier, valid_dataset, valid_classifier,  holdout_set, holdout_classifier

def CLPipeline(arg):
    seed = arg[0]
    config_tasks = arg[1]
    CL_mode = arg[2]
    save_plots_path = arg[3]
    save_outputs_path = arg[4]
    mem_replay = arg[5]
    lambda_task = arg[6]
    saved_test_data = {}   # --- saves test data for each task
    forgetting = {}  # --- saves test MSE on previous tasks with updated model
    outputs = {} # --- saves all output losses for each task
    # --- all of this is so ugly it makes me ashamed, but no time for polishing now
    # AL pipeline should be a class that initialises the models by training them the first time, then can be updated with new data (data is a self.) and relative scaler (also a self)
    models = None
    for j,cfg in enumerate(config_tasks):
        print('=====================================')
        print('=====================================')
        print(f'Task number: {j}')
        print('=====================================')
        print('=====================================')        

        if CL_mode!= 'shrink_perturb':
            cfg['hyperparams']['lambda'] = 1
        if j==0:
            train_sample, train_classifier, valid_dataset, valid_classifier, unlabelled_pool, holdout_set, holdout_classifier, saved_tests, scaler = pt.get_data(cfg,j=0)
            saved_test_data.update(saved_tests)
            # ToDo CHANGE THIS later on! ==== >>> test_new = test.sample(cfg['hyperparams']['test_size'])  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ======================= 
            first_iter = True  # -- ok, retain

        else:
            
            _, _, valid_dataset_new, valid_classifier_new, unlabelled_pool_new, holdout_set_new, holdout_classifier_new, saved_tests,scaler = pt.get_data(cfg, scaler=scaler,j=j)  # --- scaler gets updated during pipeline and passed here
            saved_test_data.update(saved_tests)

            # --- downsample data from previous tasks, otherwise too much imbalance. 
            data = [train_sample, train_classifier, valid_dataset, valid_classifier, holdout_set, holdout_classifier]
            train_sample, train_classifier, valid_dataset, valid_classifier, holdout_set, holdout_classifier =  downsample(cfg, data, mem_replay)

            # --- add new task to previous tasks : NB, train set is NOT updated here, it will get updated in AL progressively through the unlabelled pool
            # TODO ==========>>>>>>>>> MAKE SURE CL IN ALPipeline IS ABLE TO HANDLE THIS WITH NEW TRAIN DEFINITIONS!!!!!!!!

            #train_sample.add(train_sample_new)
            #train_classifier.add(train_classifier_new)
            valid_dataset.add(valid_dataset_new)   # --- in a real world situation where data is streaming this is unavailable. Need to update dynamically withi the AL pipeline.
            valid_classifier.add(valid_classifier_new)
            holdout_set.add(holdout_set_new)
            holdout_classifier.add(holdout_classifier_new)
            unlabelled_pool = unlabelled_pool_new # --- not added, only data from the new task arrives

            for flux in models.keys():
                models[flux]['Regressor'].scaler = scaler            
                
            first_iter = False
            
        inp = [seed,cfg,save_outputs_path,save_plots_path, 
            {'scaler':scaler,
            'train':{'train_class':train_classifier,'train_regr':train_sample},
            'val':{'val_class': valid_classifier, 'val_regr':valid_dataset},
            'test':{'test_class': holdout_classifier, 'test_regr': holdout_set},
            'unlabelled':unlabelled_pool}, models, first_iter]

        inp = {'run_mode':'CL','cfg':inp}

        train_sample, train_classifier, valid_dataset, valid_classifier, holdout_set, holdout_classifier, output_dict, scaler, models = ALpipeline(inp)  # --- this is really ugly as it is a recursvie thing and so should really be in a class itself, but not time to polish
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

        
        for k in saved_test_data.keys():
            saved_test_data[k]['save_regr'].scale(scaler)   # --- scale all test data saved so far with current scaler
            saved_test_data[k]['save_class'].scale(scaler)   # --- scale all test data saved so far with current scaler
            _, regr_test_loss,regr_test_loss_unscaled, regr_test_loss_unscaled_norm = models[cfg['flux'][0]]['Regressor'].predict(saved_test_data[k]['save_regr'], unscale=True) # ToDo====>>> generalise to two outputs
            _, holdout_class_losses = models[cfg['flux'][0]]['Classifier'].predict(saved_test_data[k]['save_class']) # ToDo|!!!! Change to account for different class metrics
            forgetting.update({f'regression_{k}_model{j}':{'regr_test_loss':regr_test_loss,
                                                           'regr_test_loss_unscaled':regr_test_loss_unscaled, 
                                                           'regr_test_loss_unscaled_norm': regr_test_loss_unscaled_norm}
                                                           })
            forgetting.update({f'classification_{k}_model{j}': {'holdout_class_loss':holdout_class_losses[0],
                                                                'holdout_class_acc':holdout_class_losses[1],
                                                                'holdout_class_precision':holdout_class_losses[2],
                                                                'holdout_class_recall': holdout_class_losses[3],
                                                                'holdout_class_f1': holdout_class_losses[4],
                                                                'holdout_class_auc': holdout_class_losses[5]}
                                                            })
            
            saved_test_data[k]['save_regr'].scale(scaler, unscale=True) # --- unscale data for future passes
            saved_test_data[k]['save_class'].scale(scaler, unscale=True) # --- unscale data for future passes


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
    retrain = cfg['common']['retrain_classifier']
    if CL_mode != 'shrink_perturb':
        lam = 1

    seeds = [np.random.randint(0,2**32-1) for i in range(Nbootstraps)]
    inp = []
    for s in seeds:
        inp.append([s, config_tasks,CL_mode,f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/', 
        f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/', mem_replay, lambda_task])

    with Pool(Nbootstraps) as p:
        outputs = p.map(CLPipeline,inp)
   
    with open(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/bootstrapped_CL_{CL_mode}_lam_{lambda_task}_{acquisition}_replaysize_{mem_replay}_retrainclass_{retrain}.pkl', 'wb') as f:
        pkl.dump(outputs, f)

