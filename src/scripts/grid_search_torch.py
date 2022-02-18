import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.plugins import DDPPlugin

import sys
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scripts.utils import train_keys, target_keys, prepare_model, callbacks

hyper_parameters = {
    'batch_size': 4096,
    'epochs': 100,
    'learning_rate': 0.001,
}

num_gpu = 3 # Make sure to request this in the batch script
accelerator = 'gpu'

run = "1"

train_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"
val_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
test_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/test_data_clipped.pkl"

def main():
    


