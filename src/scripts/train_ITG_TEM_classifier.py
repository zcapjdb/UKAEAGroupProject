import sys
path = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/'
path2 = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/scripts'
sys.path.append(path)
sys.path.append(path2)
import comet_ml
from pytorch_lightning.loggers import CometLogger

import numpy as np 
import pandas as pd
import h5py as h5
import seaborn as sns
import os
import torch


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from scripts.utils import train_keys, callbacks
from scripts.Classifier import Classifier, ClassifierDataset
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin
from sklearn.preprocessing import StandardScaler



num_gpu = 1  # Make sure to request this in the batch script
accelerator = "gpu"

exp = "correct_checkpoints"
datapath = "/lustre/home/pr5739/qualikiz/UKAEAGroupProject"

def main():
    
    train = pd.read_pickle(f'{datapath}/data/ITG_TEM/train.pkl')
    valid = pd.read_pickle(f'{datapath}/data/ITG_TEM/valid.pkl')
    test = pd.read_pickle(f'{datapath}/data/ITG_TEM/test.pkl')    
    
    X_train = train[train_keys].values
    Y_train = train['turb_type'].values

    ITG = len(Y_train[Y_train==0])
    TEM = len(Y_train[Y_train==1])
    TEM_ITG = len(Y_train[Y_train==2])
    n_samples = len(Y_train)
    n_classes = 3
    weights = n_samples / (n_classes * np.bincount(Y_train)) # --- https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html, actual sklearn implementation not used due to conda conflicts
    np.savetxt(f'{datapath}/classweights.txt',weights.T)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_val = valid[train_keys].values
    Y_val = valid['turb_type'].values
    X_val = scaler.transform(X_val)

    X_test = test[train_keys].values
    Y_test = test['turb_type'].values
    X_test = scaler.transform(X_test)    

    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=256, shuffle=True, drop_last=True, num_workers=16)

    ds_val = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=16)

    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long())
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=16)
    
    
    nodes = [128,256,256,128]
    n_layers = 4
    inshape = 15
    outshape = 3
    model = Classifier(n_layers, nodes, inshape,multiclass=True, outshape=outshape, weights=weights)
    
    comet_api_key = os.environ["COMET_API_KEY"]
    comet_workspace = os.environ["COMET_WORKSPACE"]

    comet_logger = CometLogger(
        api_key=comet_api_key,
        project_name='ITG_TEM_classifier',
        experiment_name=exp,
        workspace=comet_workspace,
    )
    
    cbacks=callbacks(directory='ITG_TEM',run=1,experiment_name=exp,top_k=2, monitor='val_acc')
    trainer = Trainer(
    max_epochs=100,
    logger=comet_logger,
#    accelerator=accelerator,
    #strategy=DDPPlugin(find_unused_parameters=False),
#    devices=num_gpu,
    callbacks=cbacks,
    log_every_n_steps=1,
    )
    

    trainer.fit(
    model=model, train_dataloaders=dl_train, val_dataloaders=dl_val
        )
    comet_logger.log_graph(model)

    trainer.test(dataloaders=test_loader)
    

    
                             
    
if __name__=='__main__':
    main()
