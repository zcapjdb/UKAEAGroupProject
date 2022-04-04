import sys
path = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/'
#path2 = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/scripts'
sys.path.append(path)
#sys.path.append(path2)
import numpy as np 
import pandas as pd
import h5py as h5
import seaborn as sns
from scripts.utils import train_keys
import pickle


def main():
    
    # --- load data
    datapath = "/lustre/home/pr5739/qualikiz/UKAEAGroupProject"
    train_data_path = f"{datapath}/data/train_data_clipped.pkl"
    val_data_path = f"{datapath}/data/valid_data_clipped.pkl"
    test_data_path = f"{datapath}/data/test_data_clipped.pkl"
    
    # --- concatenate previously split data
    train = pd.read_pickle(train_data_path)
    val = pd.read_pickle(val_data_path)
    test = pd.read_pickle(test_data_path)
    df = pd.concat((train,val,test))

    # --- get only unstable and invalid points
    df.loc[:,'invalid_unstable'] = 0 # 0 will be invalid
    unstable = (df['efeetg_gb']>0) | (df['efetem_gb']>0) | (df['efiitg_gb']>0)
    invalid = df['target'] == 0
    df_unstable = df[unstable]
    df_invalid = df[invalid]
    df_unstable.loc[:,'invalid_unstable'] = 1
    df_invalid_unstable = pd.concat([df_unstable, df_invalid])

    # --- split train/val/test
    train, tmp = train_test_split(df_invalid_unstable, test_size = 0.2, random_state = 42)
    valid, test = train_test_split(tmp, test_size = 0.5, random_state = 42)

    # --- save
    train.to_pickle(f'{datapath}/data/valid_unstable/train.pkl')
    valid.to_pickle(f'{datapath}/data/valid_unstable/valid.pkl')
    test.to_pickle(f'{datapath}/data/valid_unstable/test.pkl')
    
    
if __name__=='__main__':
    main()