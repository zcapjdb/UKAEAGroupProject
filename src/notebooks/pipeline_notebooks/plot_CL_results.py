import pickle as pkl
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from plotting_tools import get_losses_CL,  forget_tables, final_forgetting
import matplotlib.pylab as plt
import matplotlib as mpl
import yaml

mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
#mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3.
mpl.rcParams['axes.titlepad'] = 20
#plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

path ='/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/paper_results_DONT_TOUCH'


names = ['bootstrapped_CL_shrink_perturb_lam_0.5_random_replaysize_0.5',
        'bootstrapped_CL_shrink_perturb_lam_0.5_random_replaysize_1',
        'bootstrapped_CL_shrink_perturb_lam_1_random_replaysize_0.5',
        'bootstrapped_CL_shrink_perturb_lam_1_random_replaysize_1',
        'bootstrapped_CL_shrink_perturb_lam_0.5_individual_replaysize_0.5',
        'bootstrapped_CL_shrink_perturb_lam_0.5_individual_replaysize_1',
        'bootstrapped_CL_shrink_perturb_lam_1_individual_replaysize_0.5',
        'bootstrapped_CL_shrink_perturb_lam_1_individual_replaysize_1']

#names = ['bootstrapped_CL_shrink_perturb_lam_0.5_random_replaysize_0.5_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_0.5_random_replaysize_1_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_1_random_replaysize_0.5_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_1_random_replaysize_1_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_0.5_individual_replaysize_0.5_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_0.5_individual_replaysize_1_retrainclass_False',#
#        'bootstrapped_CL_shrink_perturb_lam_1_individual_replaysize_0.5_retrainclass_False',
#        'bootstrapped_CL_shrink_perturb_lam_1_individual_replaysize_1_retrainclass_False']

cmap_1 = plt.get_cmap('viridis')
colors_1 = [cmap_1(i*80) for i in range(4)]
cmap_2 = plt.get_cmap('autumn')
colors_2 = [cmap_2(i*80) for i in range(4)]
fig, ax = plt.subplots(1,1, figsize=(16,16))
fig1,ax1 = plt.subplots(1,1, figsize=(16,16))
dict_forget = {'class':{},'regr':{}}
for i,name in enumerate(names):
    try:
        with open(path+name+'.pkl','rb') as f:
            dic = pkl.load(f)
    #    fig, ax = plt.subplots(1,2, figsize=(20,10))
        train_mean ,train_std, val_mean, val_std, test_mean , test_std = get_losses_CL(dic,unscale=True)
        train_mean = train_mean[~np.isnan(train_mean)]
        train_std = train_std[~np.isnan(train_std)]
        val_mean = val_mean[~np.isnan(val_mean)]
        val_std = val_std[~np.isnan(val_std)]
        test_mean = test_mean[~np.isnan(test_mean)]
        test_std = test_std[~np.isnan(test_std)]

        lab = name.split('_')
        lam = lab[5]
        acquis = lab[6]
        repl = lab[8]
        label=rf'$\lambda={lam}$, {acquis}, membuff={repl}'
        #ax[0].plot(range(len(train_mean)),train_mean, label='train ',color='red',lw=4,)
        #ax[0].plot(range(len(val_mean)),val_mean, label='val',color='blue',lw=4)
        if acquis == 'random':
            ls = '--'
            color = colors_1[i]
            marker = 'd'
        else:
            ls='-'
            color=colors_2[i-4] 
            marker ='*'
        ax.plot(range(len(test_mean)),test_mean,lw=4,label=label, color=color,ls=ls)   

        #ax[0].fill_between(range(len(train_mean)),train_mean-train_std, train_mean+train_std, facecolor='red', alpha=0.4)
        #ax[0].fill_between(range(len(val_mean)),val_mean-val_std,val_mean+val_std,facecolor='blue', alpha=0.4)

        #ax[1].fill_between(range(len(test_mean)),(test_mean-test_std)*100, (test_mean+test_std)*100,facecolor='red',alpha=0.4)   
        ax.set_ylabel('MSE')
        #ax[1].set_ylabel('MSE')

        #ax[0].set_xlabel('epochs')
        ax.set_xlabel('AL iterations')
        #ax[1].legend(fontsize=20)
        ax.legend(fontsize=22)
        #fig.subplots_adjust(hspace=0.3, wspace=0.5)    
        #fig.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/{name}.png')
        #fig.clf()

        df_regr, df_class_f1 = forget_tables(dic)
    #    df_regr.to_csv(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/{name}.csv')
        forgot_regr, cum_regr, avg_regr = final_forgetting(df_regr)
        forgot_class, cum_class, avg_class = final_forgetting(df_class_f1)
        dict_forget['class'].update({label: {'cumul':cum_class,'avg':avg_class}})
        dict_forget['regr'].update({label: {'cumul':cum_regr,'avg':avg_regr}})
        
    except:
        pass
    ax1.scatter([i],[dict_forget['regr'][label]['avg']], color=color, s=1000, marker=marker, label=label)

with open('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/final_forgetting.yaml','w') as f:
    yaml.dump(dict_forget,f)
for line in [10,20,30]:
    ax.axvline(line, color='black', lw=1, ls=':')
ax.set_ylim(30,105)
fig.tight_layout()
fig.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/all_overplotted_MSE.png')
fig.clf()
ax1.set_xticks([])
ax1.set_ylabel('avg MSE forgetting')
fig1.tight_layout()
fig1.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/forgetting.png')
