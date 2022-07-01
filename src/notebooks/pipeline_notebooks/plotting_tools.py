import pickle as pkl
import matplotlib.pylab as plt
import pandas as pd
import numpy as np


def get_data(path, train_key, val_key, test_key):
    out_ = {}
    with open(path, "rb") as f:
        dic = pkl.load(f)
    L = len(dic['out'])            
    out_[train_key] = [dic['out'][i][train_key]for i in range(L)] # L models
    out_[val_key] = [dic['out'][i][val_key] for i in range(len(dic['out']))]
    out_[test_key] = [dic['out'][i][test_key] for i in range(len(dic['out']))]

#    out_['class_train_acc'] = [dic['out'][i]['class_train_acc'] for i in range(len(dic['out']))]
#    out_['class_val_acc'] = [dic['out'][i]['class_val_acc'] for i in range(len(dic['out']))]
    return out_        

def get_forgetting(master_dic,model='regressor'):
    
    forget_task = {key:{'regr_test_loss_unscaled':[],'holdout_class_f1':[],'holdout_class_precision':[],'holdout_class_recall':[]} for key in master_dic[0]['forgetting'] }
    for dic in master_dic:
        for key in dic['forgetting'].keys():   # --- key are: regressor_task0_model1 etc..
            if key.startswith('regression'):
                forget_task[key]['regr_test_loss_unscaled'].append(dic['forgetting'][key]['regr_test_loss_unscaled'])
            else:
                forget_task[key]['holdout_class_f1'].append(dic['forgetting'][key]['holdout_class_f1'])
                forget_task[key]['holdout_class_precision'].append(dic['forgetting'][key]['holdout_class_precision'])
                forget_task[key]['holdout_class_recall'].append(dic['forgetting'][key]['holdout_class_recall'])

    stats = {key:{'regr_test_loss_unscaled':{},'holdout_class_f1':{},'holdout_class_precision':{},'holdout_class_recall':{}} for key in master_dic[0]['forgetting'] }
    for key in forget_task.keys():
        for key2 in forget_task[key].keys():
            try:
                stats[key][key2]['mean'] = np.format_float_positional(np.mean(forget_task[key][key2]), precision=3, unique=False, fractional=False, trim='k')
                stats[key][key2]['std'] = np.format_float_positional(np.std(forget_task[key][key2]), precision=3,unique=False, fractional=False, trim='k')
            except:
                pass

    return stats
        

def forget_tables(dic):
    stats = get_forgetting(dic)

    df_regr = pd.DataFrame()
    df_class_f1 = pd.DataFrame()
    tasks = [f'task{i}' for i in [0,1,2,3]]
    models = [f'model{i}' for i in [0,1,2,3]]
    keys_regr = [f'regression_{i}_{j}' for i in tasks for j in models]
    keys_class = [f'classification_{i}_{j}' for i in tasks for j in models]

    for task in tasks:
        for model in models:
            key_regr = f'regression_{task}_{model}'
            try:
                df_regr.loc[task,model] = stats[key_regr]['regr_test_loss_unscaled']['mean'] #} pm {stats[key_regr]['regr_test_loss_unscaled_norm']['std']}"
            except:
                df_regr.loc[task,model]  = '-'

            key_class = f'classification_{task}_{model}'
            try:
                df_class_f1.loc[task,model] = stats[key_class]['holdout_class_f1']['mean'] #} pm {stats[key_class]['holdout_class_f1']['std']}"
            except:
                df_class_f1.loc[task,model]  = '-'    

    return df_regr, df_class_f1    

def final_forgetting(df):
    forgot = {}
    avg_forget = []
    for i in range(3):
        m3 = df.loc[f'task{i}', 'model3']
        mprev = df.loc[f'task{i}',f'model{i}']
        forget = float(mprev)-float(m3)
        forgot.update({f'task{i}':forget})
        avg_forget.append(forget)
    cum_forget = float(np.cumsum(avg_forget)[-1])
    avg_forget = float(np.mean(avg_forget))
   
    return forgot, cum_forget, avg_forget

def get_losses_nonbootstrapped(dic):
    train = []
    for d in dic[train_key]:
        train.extend(d[0])
    val = []
    for d in dic[val_key]:
        val.extend(d[0])
    test  = np.array(dic[test_key]).flatten()
    return train, val, test


def flatten_data_regr(master_dict, train_key, val_key, test_key, out_flux: int=0, unscale=False):
    train_losses = []
    val_losses = []
    test_losses =[]

    if unscale:
        train_key = f'{train_key}_unscaled'
        val_key = f'{val_key}_unscaled'
        test_key = f'{test_key}_unscaled'
    for i in range(len(master_dict)):
        
        dic = master_dict[i]['outputs']['task3']
           
        outs = []
        for o in dic[train_key]:
            app = o[out_flux]
            outs.extend(np.append(np.array(app),np.zeros(10000-len(app))+np.nan))
                
        train_losses.append(np.array(outs))

        outs = []
        for o in dic[val_key]:
            app = o[out_flux]
            outs.extend(np.append(np.array(app),np.zeros(10000-len(app))+np.nan))
            
        val_losses.append(np.array(outs))
        
        app = np.array(dic[test_key])[:,out_flux]
        test_losses.append(np.array(app))

    return train_losses, val_losses, test_losses


def flatten_data_cls(master_dict, train_key, val_key, test_key):
    train_losses = []
    val_losses = []
    test_losses =[]


    for i in range(len(master_dict)):
        
        dic = master_dict[i]['outputs']['task3']
           
        outs = []
        for o in dic[train_key]:
            app = o
            outs.extend(np.append(np.array(app),np.zeros(10000-len(app))+np.nan))
                
        train_losses.append(np.array(outs))

        outs = []
        for o in dic[val_key]:
            app = o
            outs.extend(np.append(np.array(app),np.zeros(10000-len(app))+np.nan))
            
        val_losses.append(np.array(outs))
        
        app = np.array(dic[test_key])
        test_losses.append(np.array(app))

    return train_losses, val_losses, test_losses    

def get_losses_CL(master_dict,  train_key, val_key, test_key,out_flux: int = 0, unscale=False, type='regr'):
    '''
    out_flux should be 0 for the leading flux and 1 for the derived flux
    '''

    if type == 'regr':
        train_losses, val_losses, test_losses = flatten_data_regr(master_dict, train_key, val_key, test_key,out_flux=out_flux,unscale=unscale)
    elif type == 'cls':
        train_losses, val_losses, test_losses = flatten_data_cls(master_dict, train_key, val_key, test_key)
    train_mean = np.nanmean(np.array(train_losses), axis=0)
    train_std = np.nanstd(np.array(train_losses),axis=0)
    val_mean =  np.nanmean(np.array(val_losses), axis=0)
    val_std = np.nanstd(np.array(val_losses),axis=0) 
    test_mean =  np.nanmean(np.array(test_losses), axis=0)
    test_std = np.nanstd(np.array(test_losses),axis=0) 
    return train_mean ,train_std, val_mean, val_std, test_mean , test_std  #      


def get_losses(dic, train_key, val_key, test_key,):
    train_losses = []

    if len(np.shape(dic[train_key])) == 3:
        dic[train_key] = np.squeeze(dic[train_key])
    if len(np.shape(dic[val_key])) == 3:
        dic[val_key] = np.squeeze(dic[val_key])     
    if len(np.shape(dic[test_key])) == 3:
        dic[test_key] = np.squeeze(dic[test_key])     
    for o in dic[train_key]:
        L = 0
        outs = []
        for oo  in o:
            outs.extend(oo)
            
        train_losses.append(np.append(np.array(outs),np.zeros(2000-len(outs))+np.nan))

    val_losses = []
    for o in dic[val_key]:
        L = 0
        outs = []
        for oo  in o:
            outs.extend(oo)
        
        val_losses.append(np.append(np.array(outs),np.zeros(2000-len(outs))+np.nan))

    test_losses =[]
    for o in dic[test_key]:
        test_losses.append(np.append(np.array(o),np.zeros(2000-len(o))+np.nan))
    train_mean = np.nanmean(np.array(train_losses), axis=0)
    train_std = np.nanstd(np.array(train_losses),axis=0)
    val_mean =  np.nanmean(np.array(val_losses), axis=0)
    val_std = np.nanstd(np.array(val_losses),axis=0) 
    test_mean =  np.nanmean(np.array(test_losses), axis=0)
    test_std = np.nanstd(np.array(test_losses),axis=0) 
    return train_mean ,train_std, val_mean, val_std, test_mean , test_std  #                    