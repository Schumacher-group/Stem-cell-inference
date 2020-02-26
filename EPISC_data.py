
# Discrete time observation inference
# packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# INSERT HERE YOUR LOCAL PATH TO PYTHON SCRIPTS
pathabs = "/home/linus/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/"
sys.path.insert(0, pathabs+"Stem-cell-inference")

import EPISC_ABC_simple


def prior_U(par):
    return 1



#%% 1.1 ########### ABC rejection sampling CHIR

# sampling limits for parameter proposals
PAR_MIN = 0.001
PAR_MAX = 1.0
N_MC = 100

# times, unobserved markers and initial T marker of data-sets in the order shown below
T_ls = [2,2,2,2,3,3,3,3]
marker_unobs_ls = ['S','F','S','F','S','F','S','F']
T_marker_ls = [1,1,0,0,1,1,0,0]

# INSERT PATH TO YOUR LOCAL DATA FILE
xls = pd.ExcelFile(pathabs+'raw_clonal_data_adj.xlsx')
Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D2')
pop1 = Tp_TF.values
stat_true1 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D2')
pop1 = Tp_TF.values
stat_true2 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D2')
pop1 = Tp_TF.values
stat_true3 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D2')
pop1 = Tp_TF.values
stat_true4 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D3')
pop1 = Tp_TF.values
stat_true5 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D3')
pop1 = Tp_TF.values
stat_true6 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D3')
pop1 = Tp_TF.values
stat_true7 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D3')
pop1 = Tp_TF.values
stat_true8 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

# combine data-setsY
stat_true_ls = [stat_true1,stat_true2,stat_true3,stat_true4,stat_true5,stat_true6,stat_true7,stat_true8]

# # execute inference (epsilon values and/or summary statistics can be changed in the EPISC_ABC_simple script)
# [output_parX,output_parY] = EPISC_ABC_simple.get_ABC_BF_comb(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, N_MC, PAR_MIN, PAR_MAX)

# functions for parallel computing (need to be used with multiprocessing python library)
#
# EXAMPLE:
#
from joblib import Parallel, delayed
from tqdm import tqdm

N_MC = 4000000
num_cores = 16
eps1 = 40

results = np.stack(Parallel(n_jobs=num_cores)(delayed(EPISC_ABC_simple.get_ABC_BF_comb_paralx)(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, PAR_MIN, PAR_MAX, eps1) for i in tqdm(range(N_MC),position=0, leave=True) ))
X = results[results[:,0]!=0,:]
np.save(pathabs+'CHIR_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps1)+'_ModelX',X)

results = np.stack(Parallel(n_jobs=num_cores)(delayed(EPISC_ABC_simple.get_ABC_BF_comb_paraly)(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, PAR_MIN, PAR_MAX, eps1) for i in tqdm(range(N_MC),position=0, leave=True) ))
Y = results[results[:,0]!=0,:]
np.save(pathabs+'CHIR_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps1)+'_ModelY',Y)

#%% 1.2 ########### Obtain Posterior Distributions from ABC output

# # If needed: Load ABC results which were saved earlier. Example: ABC outputs
# # calculated with an N*epsilon value of 55 for CHIR dataset
# X = np.load(pathabs+'Test55_ModelX.npy')
# Y = np.load(pathabs+'Test55_ModelY.npy')

# KDE of sampled parameters
x = np.logspace(-3,0.5,500)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X[:,idx]))
    ax.plot(np.log10(x),k1(np.log10(x)),label=par_name[idx])
    ax.fill_between(np.log10(x), k1(np.log10(x)), interpolate=True,alpha=0.25)
plt.title('Model X: parameter posteriors')
plt.xlim(-3,0.5)
#plt.ylim(0,28)
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.savefig('modelXparampostisCHIR.pdf')


x = np.logspace(-3,0.5,500)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(Y[:,idx]))
    ax.plot(np.log10(x),k1(np.log10(x)),label=par_name[idx])
    ax.fill_between(np.log10(x), k1(np.log10(x)), interpolate=True,alpha=0.25)
plt.title('Model Y: parameter posteriors')
plt.xlim(-2.5,0.5)
#plt.ylim(0,3.0)
plt.xlabel(r'$log_{10}$ reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.savefig('modelYparampostisCHIR.pdf')




#%% 2 ########### ABC rejection sampling EPISC

PAR_MIN = 0.001
PAR_MAX = 1.0
N_MC = 4000000

T_ls = [3,3,3,3]
marker_unobs_ls = ['S','F','S','F']
T_marker_ls = [1,1,0,0]


xls = pd.ExcelFile(pathabs+'raw_clonal_data_adj.xlsx')

Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TF-D3')
pop1 = Tp_TF.values
stat_true1 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TS-D3')
pop1 = Tp_TF.values
stat_true2 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TF-D3')
pop1 = Tp_TF.values
stat_true3 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TS-D3')
pop1 = Tp_TF.values
stat_true4 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]

stat_true_ls = [stat_true1,stat_true2,stat_true3,stat_true4]




# [output_parX,output_parY] = EPISC_ABC_simple.get_ABC_BF_comb_EPISC(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, N_MC, PAR_MIN, PAR_MAX)

# functions for parallel computing (need to be used with multiprocessing python library)
eps2 = 20

results = np.stack(Parallel(n_jobs=num_cores)(delayed(EPISC_ABC_simple.get_ABC_BF_comb_paralx_EPISC)(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, PAR_MIN, PAR_MAX, eps2) for i in tqdm(range(N_MC),position=0, leave=True) ))
X = results[results[:,0]!=0,:]
np.save(pathabs+'EPISC_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps2)+'_ModelX',X)

results = np.stack(Parallel(n_jobs=num_cores)(delayed(EPISC_ABC_simple.get_ABC_BF_comb_paraly_EPISC)(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, PAR_MIN, PAR_MAX, eps2) for i in tqdm(range(N_MC),position=0, leave=True) ))
Y = results[results[:,0]!=0,:]
np.save(pathabs+'EPISC_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps2)+'_ModelY',Y)



#%% 3 ########### Compare CHIR/EPISC Posteriors

from scipy.integrate import simps
x = np.logspace(-3,1,500)

# Example: Model X ABC outputs with a N*epsilon value of 50 for CHIR dataset
X = np.load(pathabs+'CHIR_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps1)+'_ModelX.npy')

stat_CHIR = np.zeros((7,2))
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X[:,idx]))
    pdf = k1(np.log10(x))
    pdf = pdf/simps(pdf,dx=0.0001)
    stat_CHIR[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]


# Example: Model X ABC outputs with a N*epsilon value of 25 for EPISC dataset
X = np.load(pathabs+'EPISC_Test_nmc'+str(N_MC)+'_nc'+str(num_cores)+'_eps'+str(eps2)+'_ModelX.npy')

stat_EPISC = np.zeros((7,2))
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X[:,idx]))
    pdf = k1(np.log10(x))
    pdf = pdf/simps(pdf,dx=0.0001)
    stat_EPISC[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]



# Abs Bar plot

# width of the bars
barWidth = 0.3

# Choose the height of bars
bars1 = 10**stat_CHIR[:,0]
bars2 = 10**stat_EPISC[:,0]

# Choose the height of the error bars (use +/- std deviation)
yer1 = [10**(stat_CHIR[:,0]-np.sqrt(stat_CHIR[:,1])), 10**(stat_CHIR[:,0]+np.sqrt(stat_CHIR[:,1]))]
yer2 = [10**(stat_EPISC[:,0]-np.sqrt(stat_EPISC[:,1])), 10**(stat_EPISC[:,0]+np.sqrt(stat_EPISC[:,1]))]

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Create bars
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax.bar(r1, bars1, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='CHIR')
ax.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='EPISC')
ax.vlines(r1, yer1[0], yer1[1])
ax.vlines(r2, yer2[0], yer2[1])

# general layout
ax.set_yscale('log')
#ax.set_ylim(0.005,2)
par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
ax.set_xticks([r + barWidth for r in range(len(bars1))])
ax.set_xticklabels(par_name)
ax.set_ylabel('Rate [1/day]')
plt.legend()
plt.savefig('relativeratesX.pdf')
