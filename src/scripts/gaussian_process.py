import math
import torch
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt
import gpytorch

import pandas as pd
import numpy as np
import pickle

from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

# Load the training data
train_data = pd.read_pickle("/share/rcifdata/jbarr/UKAEAGroupProject/data/train_data_clipped.pkl")

train_keys = [
    "ane",
    "ate",
    "autor",
    "machtor",
    "x",
    "zeff",
    "gammae",
    "q",
    "smag",
    "alpha",
    "ani1",
    "ati0",
    "normni1",
    "ti_te0",
    "lognustar",
]

target_keys = [
    "dfeitg_gb_div_efiitg_gb",
    "dfetem_gb_div_efetem_gb",
    "dfiitg_gb_div_efiitg_gb",
    "dfitem_gb_div_efetem_gb",
    "efeetg_gb",
    "efeitg_gb_div_efiitg_gb",
    "efetem_gb",
    "efiitg_gb",
    "efitem_gb_div_efetem_gb",
    "pfeitg_gb_div_efiitg_gb",
    "pfetem_gb_div_efetem_gb",
    "pfiitg_gb_div_efiitg_gb",
    "pfitem_gb_div_efetem_gb",
    "vceitg_gb_div_efiitg_gb",
    "vcetem_gb_div_efetem_gb",
    "vciitg_gb_div_efiitg_gb",
    "vcitem_gb_div_efetem_gb",
    "vfiitg_gb_div_efiitg_gb",
    "vfitem_gb_div_efetem_gb",
    "vriitg_gb_div_efiitg_gb",
    "vritem_gb_div_efetem_gb",
    "vteitg_gb_div_efiitg_gb",
    "vtiitg_gb_div_efiitg_gb",
]

# larger training loop
def main():
    predictions_dict = {}
    for target in tqdm(target_keys, total=len(target_keys)):

        # Data processing:
        scaler = StandardScaler()

        joint_keys = train_keys + [target]

        drop_data = train_data[joint_keys].dropna()

        drop_data = scaler.fit_transform(drop_data)

        x_train_data = drop_data[:,:-1]
        y_train_data = drop_data[:,-1:]
       
        assert x_train_data.shape[0] == y_train_data.shape[0]
        assert x_train_data.shape[1] == len(joint_keys)-1

        n = 5_000
        idx = np.random.permutation(n)

        x_train_data = torch.tensor(x_train_data)[idx]
        y_train_data = torch.tensor(y_train_data)[idx]

        x_train_data = x_train_data.unsqueeze(0)
        y_train_data = y_train_data.unsqueeze(0)

        x_min0, x_max0 = x_train_data.min(), x_train_data.max()

        # Gaussian Process and Model fit:
        gp = SingleTaskGP(x_train_data, y_train_data)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll);

        # Testing trained gaussian process:
        test_data = pd.read_pickle("/share/rcifdata/jbarr/UKAEAGroupProject/data/test_data_clipped.pkl")
        drop_data_test = test_data[joint_keys].dropna()

        drop_data_test = scaler.transform(drop_data_test)

        x_test_data = drop_data_test[:,:-1]
        y_test_data = drop_data_test[:,-1:]

        x_test_data = scaler.transform(x_test_data)


        assert x_test_data.shape[0] == y_test_data.shape[0]
        assert x_test_data.shape[1] == len(joint_keys)-1


        x_test_data = torch.tensor(x_test_data)
        y_test_data = torch.tensor(y_test_data)

        x_test_data = x_test_data.unsqueeze(0)
        y_test_data = y_test_data.unsqueeze(0)

        x_min0, x_max0 = x_test_data.min(), x_test_data.max()

        # Get into evaluation (predictive posterior) mode
        gp.eval()
        gp.likelihood.eval()


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            
            observed_pred = gp.likelihood(gp(x_test_data))
            
            mean = observed_pred.mean
            variance = observed_pred.variance
        
        output = mean.detach().numpy()
        out_var = variance.detach().numpy().squeeze()
        output = output.squeeze()

        predictions_dict[target] = {'means': output, 'variances': out_var, 'scaler': scaler}
    
    file_name = '/home/tmadula/submit/errors/gp_outputs.pkl'
    
    with open(file_name, "wb") as file:
                pickle.dump(predictions_dict, file)

if __name__ == "__main__":
    main()


