# Discrete time observation inference
# packages
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, "C:\\Users\\Liam\\Desktop\\Master\\code")
#import EPISC_MCMC
#import EPISC_ABC
import EPISC_ABC_simple


def prior_U(par):
    return 1

# %% create data fom model X

# simulation parameter
pop0 = np.array([1,0,0,0,0,0,0,0])
T = 1
N = 1000
marker_unobs = 'S'
T_marker = 1

# choose depending on 'full' or 'simple' model
par_true = np.exp(np.random.uniform(np.log(0.1), np.log(3), 7))
#par_true = np.array([2,1,0.1,0.1,1,2,1])
#par_true = np.array([2,1,0.2,0.2,1,0.5,0.8])

# generate trajectory (transition data)
pop1 = np.zeros((0,4))
for i in range(0,N):
    traj = EPISC_ABC_simple.trajectory_EPISC_X(par_true, pop0, T)
    pop1 = np.vstack([pop1,EPISC_ABC_simple.project_population(traj,marker_unobs)])

stat_true = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
print(stat_true[0])



# %% create data fom model Y

# simulation parameter
pop0 = np.array([1,0,0,0,0,0,0,0])
T = 1
N = 1000
marker_unobs = 'S'
T_marker = 1

# choose depending on 'full' or 'simple' model
par_true = np.exp(np.random.uniform(np.log(0.01), np.log(3), 7))
#par_true = np.array([2,1,0.1,0.1,1,2,1])
#par_true = np.array([2,1,0.2,0.2,1,0.5,0.8])

# generate trajectory (transition data)
pop1 = np.zeros((0,4))
for i in range(0,N):
    traj = EPISC_ABC_simple.trajectory_EPISC_Y(par_true, pop0, T)
    pop1 = np.vstack([pop1,EPISC_ABC_simple.project_population(traj,marker_unobs)])

stat_true = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
print(stat_true[0])



# %% load experimental data
xls = pd.ExcelFile('C:/Users/Liam/Desktop/Master/inference/raw_clonal_data_adj.xlsx')

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'S'
T=1


#Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
#pop1 = Tp_TF.values
#T_marker = 1
#marker_unobs = 'F'
#T=1


#Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
#pop1 = Tp_TF.values
#T_marker = 0
#marker_unobs = 'S'
#T=1


#Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
#pop1 = Tp_TF.values
#T_marker = 0
#marker_unobs = 'F'
#T=1

stat_true = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
print(stat_true[0])




# %% ABC MCMC Parameter estimation

PAR_MIN = 0.01
PAR_MAX = 3
N_MC = 50000
eps = 0.5

# Estimation in model X
par_ls = EPISC_ABC_simple.get_ABCMCMC_X(stat_true,T_marker,marker_unobs, T, N_MC, PAR_MIN, PAR_MAX, eps)

# Estimation in model Y
par_ls = EPISC_ABC_simple.get_ABCMCMC_Y(stat_true,T_marker,marker_unobs, T, N_MC, PAR_MIN, PAR_MAX, eps)



# par = [0,T-,T+,S-,S+,F-,F+]
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
for i in [0,1,2,3,4,5,6]:
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)

    x = np.arange(PAR_MIN, PAR_MAX, 0.001)
    kde = gaussian_kde(np.log(par_ls[3000:, i]),0.25)
    ax.plot(x, kde(np.log(x)))
    ax.fill_between(x, kde(np.log(x)), interpolate=True, color='blue', alpha=0.25)
    ax.set_xscale('log')
    ax.set_xlim(0.01,3)

    #plt.vlines(par_true[i], 0, max(kde(x)))
    ax.set_xlabel(par_name[i])
    ax.set_ylabel('p('+par_name[i]+')')

    #plt.show()
    plt.savefig("C:/Users/Liam/Desktop/Master/exp_Y_logu-001-3-"+str(i)+".png", bbox_inches="tight", dpi=600)




# %% ABC rejection sampling: Bayes factor estimation

PAR_MIN = 0.01
PAR_MAX = 3
N_MC = 1000

[output_parX,output_parY] = EPISC_ABC_simple.get_ABC_BF(stat_true, T_marker, marker_unobs, T, N_MC, PAR_MIN, PAR_MAX)

print(len(output_parX)/len(output_parY))
