import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import pytorch_lightning as pl

import glob
import json
from tqdm import tqdm
import pickle

from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from scripts.QLKNN import QLKNN, QLKNNDataset, QLKNN_Big
from scripts.utils import train_keys, target_keys, ScaleData
from sklearn.preprocessing import StandardScaler

test_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/valid_data_clipped.pkl"
train_data_path = "/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl"

target_dict = {}


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


for target in target_keys:
    print(f"Evaluating on {target}")
    path = glob.glob(
        f"/share/rcifdata/jbarr/UKAEAGroupProject/logs/QLKNN-Regressor/Run-8-{target}/*.ckpt"
    )[-1]
    model = QLKNN_Big.load_from_checkpoint(
        path, n_input=15, batch_size=2048, epochs=25, learning_rate=0.002
    )

    train_data = QLKNNDataset(
        train_data_path, columns=train_keys + [target], train=True
    )
    train_data.scale()

    test_data = QLKNNDataset(test_data_path, columns=train_keys + [target], train=False)
    # test_data.data = test_data.data.sample(100_000)
    used_idx = test_data.data.index.tolist()
    test_data.scale()

    test_loader = DataLoader(test_data, batch_size=10_000, shuffle=False, num_workers=8)

    eval_model = model.eval()
    enable_dropout(eval_model)

    runs = []
    hist_list = []
    for i in tqdm(range(100)):
        step_list = []
        print(f"Run {i}")
        for step, (x, y) in enumerate(test_loader):
            predictions = eval_model(x).detach().numpy()
            step_list.append(predictions)

        flattened_predictions = np.array(step_list).flatten()
        runs.append(flattened_predictions)
        hist_list.append(flattened_predictions[0])

    output_mean = np.mean(np.array(runs), axis=0)

    out_std = np.std(np.array(runs), axis=0)

    params = {"indices": used_idx, "means": output_mean, "std": out_std}

    target_dict[target] = params

with open("saved_dictionary_big.pkl", "wb") as f:
    pickle.dump(target_dict, f)
