#!/usr/bin/env python  
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar

import sys
import pickle
from utils import train_keys, target_keys, prepare_model, callbacks
from Classifier import Classifier, ClassifierDataset

num_gpu = 3 # Make sure to request this in the batch script
accelerator = 'gpu'

run = "1"

train_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
val_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"

def main():
    parameters = {
    'nodes': [128],
    'layers': [3]
    }

    hyper_parameters = {
        'batch_size': 4096,
        'epochs': 100,
        'learning_rate': 0.001,
    }

    train_data = ClassifierDataset(train_data_path, xcolumns=train_keys, ycolumns=['target'], train = True)
    val_data = ClassifierDataset(val_data_path, xcolumns = train_keys, ycolumns =['target'])

    train_data.scale()
    val_data.scale()

    train_loader = DataLoader(train_data, batch_size = hyper_parameters['batch_size'], shuffle = True, num_workers = 6)
    val_loader = DataLoader(val_data, batch_size = hyper_parameters['batch_size'], shuffle = False, num_workers = 6)


    def grid_search(parameters, train_loader, val_loader, inshape =15): 
        '''
        Inputs: 
            build_fn: a function that will be used to build the neural network
            parameters: a dictionary of model parameters
            train_data: 
            val_data
        '''
        
        results_dict = {}
        
        
        counter = 0
        
        best_val_loss = sys.float_info.max
        
        for i in parameters['layers']:
            
            

            #List of possible node combinations
            n = i 

            combs = [[nodesize]*i for nodesize in parameters['nodes']]
            
            for node in combs:

                # build model
                model = Classifier()
                early_stopping = EarlyStopping('loss', patience = 10)
                progress_bar = ProgressBar()

                model.build_classifier(i, node, inshape)

                trainer = Trainer(
                    max_epochs = hyper_parameters['epochs'],
                    accelerator = accelerator,
                    strategy = DDPPlugin(find_unused_parameters = False),
                    devices = num_gpu,
                    callbacks = [early_stopping, progress_bar]

                )

                history = trainer.fit(model, train_loader)
                evaluate = trainer.validate(model, val_loader)

                print(f'Fit Output type: {type(history)}')
                print(f'Validate Output type: {type(evaluate)}')



                trial_dict = {
                    'layers': i,
                    'nodes': node,
                    'history': history.history, 
                    'perfomance': evaluate
                }
                
                # if evaluate[1] < best_val_loss: 
                #     results_dict['best_model'] = trial_dict
            
                
                results_dict['trial_'+str(counter)] = trial_dict
                
                file_name = f'/home/tmadula/grid_search/trial_test{str(counter)}.pkl'
                with open(file_name, 'wb') as file:
                    pickle.dump(trial_dict, file)    

                counter += 1
        return results_dict

    grid_dict = grid_search(parameters,train_loader, val_loader)


if __name__ == "__main__":
    main()


