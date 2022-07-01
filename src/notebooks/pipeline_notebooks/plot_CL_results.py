import pickle as pkl
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from plotting_tools import get_losses_CL,  forget_tables, final_forgetting
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


path ='/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/'#paper_results_DONT_TOUCH/'

def losses_and_forgetting():

    useAL = True
 
    lambdas = [0.6,0.8,1]
    replays = [0.2,0.4,0.6,0.8,1]
    acquis =  ['individual']
    names = [f'bootstrapped_CL_shrink_perturb_lam_{lam}_{acq}_replaysize_{mem}_useAL_{useAL}' for acq in acquis for lam in lambdas for mem in replays]
    cmap_1 = plt.get_cmap('Greens')

    split = int(255/len(names))
    colors_1 = [cmap_1(i*split) for i in range(len(names))]
    cmap_2 = plt.get_cmap('Purples')
    colors_2 = [cmap_2(i*split) for i in range(len(names))]
    cmap3 = plt.get_cmap('jet')
    colors_3 = [cmap3(i*split) for i in range(len(names))]

    fig, ax = plt.subplots(1,1, figsize=(16,16))
    fig1,ax1 = plt.subplots(1,1, figsize=(16,16)) # ---ignore 
    fig2,ax2 = plt.subplots(1,1, figsize=(16,16))
    dict_forget = {'class':{},'regr':{}}

    for acq in acquis:
        fig3, ax3 = plt.subplots(1,4, figsize=(40,10)) # --- acquisition-specific
        fig4,ax4 = plt.subplots(1,4, figsize=(40,10))

        L = []
        R =[] 
        task0 =[]
        task1 = []
        task2 = []
        task3 =[]

        task0_cls =[]
        task1_cls = []
        task2_cls = []
        task3_cls =[]        
        i =0 
        for lam in lambdas:
            for mem in replays:
                name = f'bootstrapped_CL_shrink_perturb_lam_{lam}_{acq}_replaysize_{mem}_useAL_{useAL}'
            
                try:
                    with open(path+name+'.pkl','rb') as f:
                        dic = pkl.load(f)
                    print( name)

                #    fig, ax = plt.subplots(1,2, figsize=(20,10))
                    train_mean ,train_std, val_mean, val_std, test_mean , test_std = get_losses_CL(dic,
                                                                                        train_key='retrain_losses',
                                                                                        val_key='retrain_val_losses',
                                                                                        test_key = 'post_test_loss',
                                                                                        unscale=True, type = 'regr')
                    train_mean = train_mean[~np.isnan(train_mean)]
                    train_std = train_std[~np.isnan(train_std)]
                    val_mean = val_mean[~np.isnan(val_mean)]
                    val_std = val_std[~np.isnan(val_std)]
                    test_mean = test_mean[~np.isnan(test_mean)]
                    test_std = test_std[~np.isnan(test_std)]

                    # ToDo====>>> --- currently does not handle the case of useAL=False, as F1 score is not returned by the training routines 
                    train_mean_cls ,train_std_cls, val_mean_cls, val_std_cls, test_mean_cls , test_std_cls = get_losses_CL(dic,
                                                                                        train_key='class_train_acc',
                                                                                        val_key='class_val_acc',
                                                                                        test_key = 'holdout_class_precision',
                                                                                        type = 'cls'
                                                                                        )    
                    print('here')                
                    train_mean_cls = train_mean_cls[~np.isnan(train_mean_cls)]
                    train_std_cls = train_std_cls[~np.isnan(train_std_cls)]
                    val_mean_cls = val_mean_cls[~np.isnan(val_mean_cls)]
                    val_std_cls = val_std_cls[~np.isnan(val_std_cls)]
                    test_mean_cls = test_mean_cls[~np.isnan(test_mean_cls)]
                    test_std_cls = test_std_cls[~np.isnan(test_std_cls)]

                    lab = name.split('_')
                    lam = lab[5]
                    acquis = lab[6][:3]
                    repl = lab[8][:3]
                    label=rf'$\lambda={lam}$, {acquis}, mem={repl}'
                    #ax[0].plot(range(len(train_mean)),train_mean, label='train ',color='red',lw=4,)
                    if acquis == 'ran':
                        ls = '--'
                        color = colors_1[i]
                        marker = 'd'
                    elif acquis == 'ind':
                        ls='-'
                        color=colors_2[i-4] 
                        marker ='*'
                    elif acquis == 'Non':
                        ls = '-'
                        color = colors_3[i]
                        marker = 'o'
                
                    print('LENGTH',len(test_mean))
                    if len(test_mean)>4: # --- hack to account for fact that only when including AL you have complete loss curves
                        task0.append(test_mean[24])
                        task1.append(test_mean[49])
                        task2.append(test_mean[74])
                        task3.append(test_mean[99])
                    # ToDo====>>> --- currently does not handle the case of useAL=False, as F1 score is not returned by the training routines 
                        task0_cls.append(test_mean_cls[24])
                        task1_cls.append(test_mean_cls[49])
                        task2_cls.append(test_mean_cls[74])
                        task3_cls.append(test_mean_cls[99])  
                        xlabel_ax2 = 'AL iterations'                     
                    else:
                        task0.append(test_mean[0])
                        task1.append(test_mean[1])
                        task2.append(test_mean[2])
                        task3.append(test_mean[3])
                    # ToDo====>>> --- currently does not handle the case of useAL=False, as F1 score is not returned by the training routines 
                        task0_cls.append(test_mean_cls[0])
                        task1_cls.append(test_mean_cls[1])
                        task2_cls.append(test_mean_cls[2])
                        task3_cls.append(test_mean_cls[3])     
                        xlabel_ax2 = 'CL iterations'

                    ax.plot(range(len(test_mean)),test_mean, label=label,color=color,lw=4)
                
                    ax.set_ylabel('MSE')
                    ax.set_xlabel(xlabel_ax2)
                    ax.legend(fontsize=15, ncol=4)

                    ax2.plot(range(len(test_mean_cls)),test_mean_cls, label=label,color=color,lw=4)
                
                    ax2.set_ylabel('F1 score')
                    ax2.set_xlabel(xlabel_ax2)
                    ax2.legend(fontsize=15, ncol=4)                    

                except:
                    print(name,'  HAS FAILED!!')
                    task0.append(np.nan)
                    task1.append(np.nan)
                    task2.append(np.nan)
                    task3.append(np.nan)
                    task0_cls.append(np.nan)
                    task1_cls.append(np.nan)
                    task2_cls.append(np.nan)
                    task3_cls.append(np.nan)                    
                L.append(lam)
                R.append(mem)  
            i+=1                 
        task0 = np.array(task0).reshape((len(lambdas),len(replays)))
        task1 = np.array(task1).reshape((len(lambdas),len(replays)))
        task2 = np.array(task2).reshape((len(lambdas),len(replays)))
        task3 = np.array(task3).reshape((len(lambdas),len(replays)))
        tasknames = ['C-L','C-H','ILW-L','ILW-H']
        for i,(t,a) in enumerate(zip([task0,task1,task2,task3], ax3)):
            print(t)
            im = a.imshow(np.flipud(t.T), origin='upper', extent=(0.4,1,0.4,1))
            divider = make_axes_locatable(a)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label('test MSE')
            ticks = np.linspace(np.nanmin(t.flatten()),np.nanmax(t.flatten()),5).astype(int)
            print(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)
            a.set_xlabel(r'$\lambda$')
            a.set_ylabel(r'$\alpha$')
            a.set_title(tasknames[i],fontsize=45)
            a.set_xticks([0.4,0.6,0.8,1])
            a.set_yticks([0.4,0.6,0.8,1])
            a.minorticks_off()
        fig3.tight_layout()
        fig3.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/losses_heatmap_useAL_{useAL}_{acq}.png')
        fig3.clf()    

        task0_cls = np.array(task0_cls).reshape((len(lambdas),len(replays)))
        task1_cls = np.array(task1_cls).reshape((len(lambdas),len(replays)))
        task2_cls = np.array(task2_cls).reshape((len(lambdas),len(replays)))
        task3_cls = np.array(task3_cls).reshape((len(lambdas),len(replays)))
        tasknames = ['C-L','C-H','ILW-L','ILW-H']
        for i,(t,a) in enumerate(zip([task0_cls,task1_cls,task2_cls,task3_cls], ax4)):
            print(f'task{i},',t)
            im = a.imshow(np.flipud(t.T), origin='upper', extent=(0.4,1,0.4,1))
            divider = make_axes_locatable(a)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label('test F1')
            ticks = np.around(np.linspace(np.nanmin(t.flatten()),np.nanmax(t.flatten()),5),decimals=2)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)
            a.set_xlabel(r'$\lambda$')
            a.set_ylabel(r'$\alpha$')
            a.set_title(tasknames[i],fontsize=45)
            a.set_xticks([0.4,0.6,0.8,1])
            a.set_yticks([0.4,0.6,0.8,1])
            a.minorticks_off()
        fig4.tight_layout()
        fig4.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/losses_cls_heatmap_useAL_{useAL}_{acq}.png')
        fig4.clf()            

#    with open(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/bootstrap/final_forgetting_20iter_useAL_{useAL}.yaml','w') as f:
 #       yaml.dump(dict_forget,f)
    for line in [1,2,3]:
        ax.axvline(line, color='black', lw=1, ls=':')
    ax.set_ylim(0,100)
    ax.set_yticks(np.arange(0,110,10))
    fig.tight_layout()
    fig.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/all_overplotted_MSE_20iter_useAL_{useAL}.png')
    fig.clf()

    ax2.set_ylim(0.5,1)
    ax2.set_yticks(np.arange(0.5,1.1,0.1))
    fig2.tight_layout()
    fig2.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/all_overplotted_F1_20iter_useAL_{useAL}.png')
    fig2.clf()

    ax1.set_xticks([])
    ax1.legend(fontsize=15, ncol=4)
    ax1.set_ylabel('avg MSE forgetting')
    fig1.tight_layout()
    fig1.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/forgetting_20iter_useAL_{useAL}.png')



def get_forgetting_scatter_plot():
    useAL = True

    lambdas = [0.6,0.8,1]
    replays = [0.2,0.4,0.6,0.8,1]
    names = [ i*j for i in lambdas for j in replays]

    dict_forget = {'class':{},'regr':{}}

    cmap_1 = plt.get_cmap('viridis')
    split = int(255/len(names))
    colors_1 = [cmap_1(i*split) for i in range(len(names))]


    acquis = ['individual']#,'random']
    for acq in acquis:
        avg_regr = []
        avg_cls = []
        L =[]
        R =[]        
        fig, ax = plt.subplots(1,1)
        fig1,ax1 = plt.subplots(1,1)        
        for lam in lambdas:
            for mem in replays:
                name = f'bootstrapped_CL_shrink_perturb_lam_{lam}_{acq}_replaysize_{mem}_useAL_{useAL}'

                try:
                    with open(path+name+'.pkl','rb') as f:
                        dic = pkl.load(f)
                    print( name)
                    df_regr, df_class_f1 = forget_tables(dic)
                #    df_regr.to_csv(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/outputs/CL/{name}.csv')
                    forgot_regr, cum_regr, avg_regr_ = final_forgetting(df_regr)
                    forgot_class, cum_class, avg_cls_ = final_forgetting(df_class_f1)
                    avg_regr.append(avg_regr_)
                    avg_cls.append(avg_cls_)

                except:
                    print(name,' HAS FAILED!!!')
                    avg_regr.append(np.nan)
                    avg_cls.append(np.nan)
                L.append(lam)
                R.append(mem)
        avg_regr = np.array(avg_regr).reshape((len(lambdas),len(replays)))
        avg_cls = np.array(avg_cls).reshape((len(lambdas),len(replays)))
        im = ax.imshow(np.flipud(avg_regr.T), origin='upper', extent=(0.4,1,0.4,1))
        im1 = ax1.imshow(np.flipud(avg_cls.T), origin='upper', extent=(0.4,1,0.4,1))

        ax.set_xticks([0.4,0.6,0.8,1])
        ax.set_yticks([0.4,0.6,0.8,1])
        ax1.set_xticks([0.4,0.6,0.8,1])
        ax1.set_yticks([0.4,0.6,0.8,1])
        ax.minorticks_off()
        ax1.minorticks_off()
    # norm = plt.Normalize(np.min(avg_regr), np.max(avg_regr))
    # ax.scatter(L,R,c=avg_regr, cmap='viridis',norm=norm, s=5000)
    # smap = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        cbar = fig.colorbar(im)
        cbar.set_label('avg forgetting')
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\alpha$')
        fig.tight_layout()
        fig.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/forgetting_scatterplot_useAL_{useAL}_{acq}.png')
        fig.clf()

        cbar1 = fig1.colorbar(im1)
        cbar1.set_label('avg forgetting')
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(r'$\alpha$')
        fig1.tight_layout()
        fig1.savefig(f'/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/plots/CL/forgetting_cls_scatterplot_useAL_{useAL}_{acq}.png')
        fig1.clf()

if __name__=='__main__':
    losses_and_forgetting()
   # get_forgetting_scatter_plot()