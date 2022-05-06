import pstats
import numpy as np
import h5py as h5
import pandas as pd

from sklearn.model_selection import train_test_split

from scripts.utils import train_keys, target_keys, jet_keys

path = "/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/data"
def main(required: str = 'LH'):
    '''
    required: 'LH' (for LH transition), 'CB' (for carbon-berillium wall), 'all' (for no distinction)    
    '''
    print('readin inputs...')
    df_in = pd.read_hdf(f"{path}/qlk_jetexp_nn_training_database_minimal.h5",key='input')
    df_in =  df_in.rename(columns= {c:c.lower() for c in df_in.columns })
    print('readin outputs...')
    df_out = pd.read_hdf(f"{path}/qlk_jetexp_nn_training_database_minimal.h5",key='output')
    df_out = df_out.rename(columns= {c:c.lower() for c in df_out.columns })
    print('readin labels...')
    df_labels = pd.read_hdf(f"{path}/qlk_jetexp_nn_training_database_labels.h5")
    df_labels = df_labels.rename(columns= {c:c.lower() for c in df_labels.columns })
    print('running...')
    if required == 'all':
        df_labels = df_labels[jet_keys]
    elif required == 'CB':
        df_labels = df_labels['wall_material_index']
    elif required == 'LH':
        df_labels = df_labels[['discharge_phase_index', 'is_hmode']]
    df_out = df_out[target_keys]    

    idx_in = df_in.index.values.tolist()
    idx_out = df_out.index.values.tolist()
    intersect = np.intersect1d(idx_in, idx_out)

    df_out.reindex(range(df_in.index[0], df_in.index[-1] + 1), fill_value = np.nan)
    df = pd.concat([df_in, df_out], axis = 1)
    #Give a class label of 1 to inputs which have an output and 0 to inputs without a corresponding QuaLiKz output
    y = np.where(np.in1d(idx_in, idx_out), 1, 0)
    df['target'] = y   
    # --- clip DF
    Q1 = df.quantile(0.01)
    Q9 = df.quantile(0.99)
    # Last value is categorial target which takes 0 or 1, no outliers here so make sure all values are accepted
    Q1[-1] = -1
    Q9[-1] = 2
    df = df[~((df < Q1) | (df > Q9)).any(axis = 1) ]    

    df = pd.merge(df,df_labels, left_index=True, right_index=True)

    if required == 'all':
        train, tmp = train_test_split(df, test_size = 0.2, random_state = 42)
        valid, test = train_test_split(tmp, test_size = 0.5, random_state = 42)    
        train.to_pickle(f"{path}/train_data_clipped.pkl")
        valid.to_pickle(f"{path}/valid_data_clipped.pkl")
        test.to_pickle(f"{path}/test_data_clipped.pkl")
    elif required == 'CB':
        train_C, tmp = train_test_split(df.query('wall_material_index==0'), test_size = 0.2, random_state = 42)
        valid_C, test_C = train_test_split(tmp, test_size = 0.5, random_state = 42)    
        train_B, tmp = train_test_split(df.query('wall_material_index==1'), test_size = 0.2, random_state = 42)
        valid_B, test_B = train_test_split(tmp, test_size = 0.5, random_state = 42)    
        train_C.to_pickle(f"{path}/carbonwall/train_data_clipped.pkl")
        valid_C.to_pickle(f"{path}/carbonwall/valid_data_clipped.pkl")
        test_C.to_pickle(f"{path}/carbonwall/test_data_clipped.pkl")
        train_B.to_pickle(f"{path}/berilliumwall/train_data_clipped.pkl")
        valid_B.to_pickle(f"{path}/berilliumwall/valid_data_clipped.pkl")
        test_B.to_pickle(f"{path}/berilliumwall/test_data_clipped.pkl")
    elif required == 'LH':
        train_L, tmp = train_test_split(df.query('is_hmode==0 & discharge_phase_index==0'), test_size = 0.2, random_state = 42)
        valid_L, test_L = train_test_split(tmp, test_size = 0.5, random_state = 42)    
        train_H, tmp = train_test_split(df.query('is_hmode==1 & discharge_phase_index==0'), test_size = 0.2, random_state = 42)
        valid_H, test_H = train_test_split(tmp, test_size = 0.5, random_state = 42)    
        train_L.to_pickle(f"{path}/Lmode/train_data_clipped.pkl")
        valid_L.to_pickle(f"{path}/Lmode/valid_data_clipped.pkl")
        test_L.to_pickle(f"{path}/Lmode/test_data_clipped.pkl")
        train_H.to_pickle(f"{path}/Hmode/train_data_clipped.pkl")
        valid_H.to_pickle(f"{path}/Hmode/valid_data_clipped.pkl")
        test_H.to_pickle(f"{path}/Hmode/test_data_clipped.pkl")

if __name__=='__main__':
    main('LH')