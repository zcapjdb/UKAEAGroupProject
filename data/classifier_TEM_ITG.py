import sys
path = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src'
path2 = '/lustre/home/pr5739/qualikiz/UKAEAGroupProject/src/scripts'
sys.path.append(path)
#sys.path.append(path2)
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
#from scripts.utils import train_keys ## --- FIX THIS 
import pickle

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
    del train, val, test
    
    df = df[train_keys+['efetem_gb','efiitg_gb']]
    
    # --- get only unstable points
   # df.loc[:,'ITG'] = 0 # 0 will be no ITG
   # df.loc[:,'TEM'] = 0 # 0 will be no TEM
   # df.loc[:,'ITG_and_TEM'] = 0 # will be 0 for no mode occurring
    df.loc[:,'turb_type'] = 0
    ITG = df['efiitg_gb']>0
    TEM = df['efetem_gb']>0
    df.loc[ITG,'turb_type'] = 0
    df.loc[TEM,'turb_type'] = 1
    # --- overwrites where both occur
    df.loc[ITG & TEM,'turb_type'] = 2
    df_out = df[ ITG | TEM ] 


    # --- split train/val/test
    train, tmp = train_test_split(df_out, test_size = 0.2, random_state = 42)
    valid, test = train_test_split(tmp, test_size = 0.5, random_state = 42)

    # --- save
    train.to_pickle(f'{datapath}/data/ITG_TEM/train.pkl')
    valid.to_pickle(f'{datapath}/data/ITG_TEM/valid.pkl')
    test.to_pickle(f'{datapath}/data/ITG_TEM/test.pkl')
    
    
if __name__=='__main__':
    main()