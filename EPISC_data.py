# Discrete time observation inference
# packages
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, "/home/ruske/Desktop/Stem-cell-inference-master/code")
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
xls = pd.ExcelFile('/home/ruske/Desktop/Stem-cell-inference-master/inference/raw_clonal_data_adj.xlsx')


# Day 2 data

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'S'
T=2


Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'F'
T=2


Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'S'
T=2


Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D2')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'F'
T=2

# Day 3 data

Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'S'
T=3


Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'F'
T=3


Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'S'
T=3


Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'F'
T=3




# Day 3 data EPISC

Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TF-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'S'
T=3


Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TS-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 1
marker_unobs = 'F'
T=3


Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TF-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'S'
T=3


Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TS-D3')
# remove headers. Order: T+F+,T+F-,T-F+,T-F-
pop1 = Tp_TF.values
T_marker = 0
marker_unobs = 'F'
T=3



stat_true = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
print(stat_true[0])




# %% ABC MCMC Parameter estimation

PAR_MIN = 0.005
PAR_MAX = 2
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
    plt.savefig("/home/ruske/Desktop/Stem-cell-inference-master/exp_Y_logu-001-3-"+str(i)+".png", bbox_inches="tight", dpi=600)




# %% ABC rejection sampling: Bayes factor estimation

PAR_MIN = 0.005
PAR_MAX = 2
N_MC = 50000

[output_parX,output_parY] = EPISC_ABC_simple.get_ABC_BF(stat_true, T_marker, marker_unobs, T, N_MC, PAR_MIN, PAR_MAX)


np.save('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tm_nomF_D3',output_parX)
np.save('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_Y_par_Tm_nomF_D3',output_parY)






















# %% ABC rejection sampling Linus

PAR_MIN = 0.005
PAR_MAX = 1.0
N_MC = 50000

T_ls = [2,2,2,2,3,3,3,3]
marker_unobs_ls = ['S','F','S','F','S','F','S','F']
T_marker_ls = [1,1,0,0,1,1,0,0] 


xls = pd.ExcelFile('/home/ruske/Desktop/Stem-cell-inference-master/inference/raw_clonal_data_adj.xlsx')
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

stat_true_ls = [stat_true1,stat_true2,stat_true3,stat_true4,stat_true5,stat_true6,stat_true7,stat_true8]





[output_parX,output_parY] = EPISC_ABC_simple.get_ABC_BF_comb(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, N_MC, PAR_MIN, PAR_MAX)


X = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Test_ModelX.npy')
Y = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Test_ModelY.npy')



x = np.logspace(-3,0.5,500)
fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X[:,idx]))
    ax.plot(np.log10(x),k1(np.log10(x)),label=par_name[idx])
    ax.fill_between(np.log10(x), k1(np.log10(x)), interpolate=True,alpha=0.25)
plt.title('Model X: parameter posteriors')
plt.xlim(-2.5,0.5)
#plt.ylim(0,28)
plt.xlabel('reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()


x = np.logspace(-3,0.5,500)
fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(Y[:,idx]))
    ax.plot(np.log10(x),k1(np.log10(x)),label=par_name[idx])
    ax.fill_between(np.log10(x), k1(np.log10(x)), interpolate=True,alpha=0.25)
plt.title('Model Y: parameter posteriors')
plt.xlim(-2.5,0.5)
#plt.ylim(0,28)
plt.xlabel('reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()















# %% Plot that X CHIR
from scipy.integrate import simps
# log-uniform PDF between PAR_MIN and PAR_MAX
def log_unif(x, PAR_MIN, PAR_MAX):
    return 1 / (x * (np.log(PAR_MAX) - np.log(PAR_MIN)))
x = np.logspace(-3,0.5,500)

X1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomF_D2.npy')
X2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomS_D2.npy')
X3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomF_D2.npy')
X4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomS_D2.npy')
X5 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomF_D3.npy')
X6 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomS_D3.npy')
X7 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomF_D3.npy')
X8 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomS_D3.npy')


for idx in range(0,7):
    k1 = gaussian_kde(X1[:,idx])
    k2 = gaussian_kde(X2[:,idx])
    k3 = gaussian_kde(X3[:,idx])
    k4 = gaussian_kde(X4[:,idx])
    k5 = gaussian_kde(X5[:,idx])
    k6 = gaussian_kde(X6[:,idx])
    k7 = gaussian_kde(X7[:,idx])
    k8 = gaussian_kde(X8[:,idx])

    fig = plt.figure(figsize=(12,4))        
    ax = plt.subplot(111)
    ax.semilogx(x,k1(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k2(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k3(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k4(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k5(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k6(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k7(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k8(x)/log_unif(x, 0.005, 2))


fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X1[:,idx]))
    k2 = gaussian_kde(np.log10(X2[:,idx]))  
    k3 = gaussian_kde(np.log10(X3[:,idx]))
    k4 = gaussian_kde(np.log10(X4[:,idx]))
    k5 = gaussian_kde(np.log10(X5[:,idx]))
    k6 = gaussian_kde(np.log10(X6[:,idx]))
    k7 = gaussian_kde(np.log10(X7[:,idx]))
    k8 = gaussian_kde(np.log10(X8[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x))*k5(np.log10(x))*k6(np.log10(x))*k7(np.log10(x))*k8(np.log10(x))#/log_unif(x, 0.005, 2)**7
    pdf = pdf/simps(pdf,dx=0.001)
    ax.semilogx(x,pdf,label=par_name[idx])
    ax.fill_between(x, pdf, interpolate=True,alpha=0.25)
plt.title('Model X: parameter posteriors')
plt.xlim(0.005,2)
#plt.ylim(0,28)
plt.xlabel(r'Reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()



# %% Plot that Y CHIR
from scipy.integrate import simps
x = np.logspace(-3,0.5,200)

Y1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tp_nomF_D2.npy')
Y2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tp_nomS_D2.npy')
Y3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tm_nomF_D2.npy')
Y4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tm_nomS_D2.npy')
Y5 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tp_nomF_D3.npy')
Y6 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tp_nomS_D3.npy')
Y7 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tm_nomF_D3.npy')
Y8 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/Y_par_Tm_nomS_D3.npy')


for idx in range(0,7):
    k1 = gaussian_kde(Y1[:,idx])
    k2 = gaussian_kde(Y2[:,idx])
    k3 = gaussian_kde(Y3[:,idx])
    k4 = gaussian_kde(Y4[:,idx])
    k5 = gaussian_kde(Y5[:,idx])
    k6 = gaussian_kde(Y6[:,idx])
    k7 = gaussian_kde(Y7[:,idx])
    k8 = gaussian_kde(Y8[:,idx])

    fig = plt.figure(figsize=(12,4))        
    ax = plt.subplot(111)
    ax.semilogx(x,k1(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k2(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k3(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k4(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k5(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k6(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k7(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k8(x)/log_unif(x, 0.005, 2))


fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(Y1[:,idx]))
    k2 = gaussian_kde(np.log10(Y2[:,idx]))  
    k3 = gaussian_kde(np.log10(Y3[:,idx]))
    k4 = gaussian_kde(np.log10(Y4[:,idx]))
    k5 = gaussian_kde(np.log10(Y5[:,idx]))
    k6 = gaussian_kde(np.log10(Y6[:,idx]))
    k7 = gaussian_kde(np.log10(Y7[:,idx]))
    k8 = gaussian_kde(np.log10(Y8[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x))*k5(np.log10(x))*k6(np.log10(x))*k7(np.log10(x))*k8(np.log10(x))#/log_unif(x, 0.005, 2)**7
    pdf = pdf/simps(pdf,dx=0.001)
    ax.plot(np.log10(x),pdf,label=par_name[idx])
    ax.fill_between(np.log10(x), pdf, interpolate=True,alpha=0.25)
plt.title('Model Y: parameter posteriors')
plt.xlim(-2,0.3)
#plt.ylim(0,28)
plt.xlabel(r'$log_{10}$ reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()





















# %% Plot that X EPISC
from scipy.integrate import simps
x = np.logspace(-3,0.5,500)

# exclude X1 for now because too few data points
X1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tp_nomF_D3.npy')
X2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tp_nomS_D3.npy')
X3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tm_nomF_D3.npy')
X4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tm_nomS_D3.npy')

"""
for idx in range(0,7):
    k1 = gaussian_kde(X1[:,idx])
    k2 = gaussian_kde(X2[:,idx])
    k3 = gaussian_kde(X3[:,idx])
    k4 = gaussian_kde(X4[:,idx])

    fig = plt.figure(figsize=(12,4))        
    ax = plt.subplot(111)
    ax.semilogx(x,k1(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k2(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k3(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k4(x)/log_unif(x, 0.005, 2))
"""


fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X1[:,idx]))
    k2 = gaussian_kde(np.log10(X2[:,idx]))  
    k3 = gaussian_kde(np.log10(X3[:,idx]))
    k4 = gaussian_kde(np.log10(X4[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x)) #/log_unif(x, 0.005, 2)**2
    pdf = pdf/simps(pdf,dx=0.001)
    ax.plot(np.log10(x),pdf,label=par_name[idx])
    ax.fill_between(np.log10(x), pdf, interpolate=True,alpha=0.25)
plt.title('Model X: parameter posteriors')
plt.xlim(-2.5,0.5)
#plt.ylim(0,20)
plt.xlabel(r'$log_{10}$ reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()






# %% Plot that Y EPISC
from scipy.integrate import simps
x = np.logspace(-3,0.5,500)

Y1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_Y_par_Tp_nomF_D3.npy')
Y2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_Y_par_Tp_nomS_D3.npy')
Y3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_Y_par_Tm_nomF_D3.npy')
Y4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_Y_par_Tm_nomS_D3.npy')

'''
for idx in range(0,7):
    k1 = gaussian_kde(Y1[:,idx])
    k2 = gaussian_kde(Y2[:,idx])
    k3 = gaussian_kde(Y3[:,idx])
    k4 = gaussian_kde(Y4[:,idx])

    fig = plt.figure(figsize=(12,4))        
    ax = plt.subplot(111)
    ax.semilogx(x,k1(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k2(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k3(x)/log_unif(x, 0.005, 2))
    ax.semilogx(x,k4(x)/log_unif(x, 0.005, 2))
'''


fig = plt.figure(figsize=(10,5))        
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(Y1[:,idx]))
    k2 = gaussian_kde(np.log10(Y2[:,idx]))  
    k3 = gaussian_kde(np.log10(Y3[:,idx]))
    k4 = gaussian_kde(np.log10(Y4[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x)) #/log_unif(x, 0.005, 2)**2
    pdf = pdf/simps(pdf,dx=0.001)
    ax.plot(np.log10(x),pdf,label=par_name[idx])
    ax.fill_between(np.log10(x), pdf, interpolate=True,alpha=0.25)
plt.title('Model Y: parameter posteriors')
plt.xlim(-2.5,0.5)
plt.ylim(0,30)
plt.xlabel(r'$log_{10}$ reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.show()




# %% Compare CHIR and EPISC

x = np.logspace(-3,0.5,500)

X1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomF_D2.npy')
X2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomS_D2.npy')
X3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomF_D2.npy')
X4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomS_D2.npy')
X5 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomF_D3.npy')
X6 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tp_nomS_D3.npy')
X7 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomF_D3.npy')
X8 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/X_par_Tm_nomS_D3.npy')

stat_CHIR = np.zeros((7,2))
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X1[:,idx]))
    k2 = gaussian_kde(np.log10(X2[:,idx])) 
    k3 = gaussian_kde(np.log10(X3[:,idx]))
    k4 = gaussian_kde(np.log10(X4[:,idx]))
    k5 = gaussian_kde(np.log10(X5[:,idx]))
    k6 = gaussian_kde(np.log10(X6[:,idx]))
    k7 = gaussian_kde(np.log10(X7[:,idx]))
    k8 = gaussian_kde(np.log10(X8[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x))*k5(np.log10(x))*k6(np.log10(x))*k7(np.log10(x))*k8(np.log10(x))#/log_unif(x, 0.005, 2)**7
    pdf = pdf/simps(pdf,dx=0.0001)
    stat_CHIR[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
    
    
    
X1 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tp_nomF_D3.npy')
X2 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tp_nomS_D3.npy')
X3 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tm_nomF_D3.npy')
X4 = np.load('/home/ruske/Desktop/Stem-cell-inference-master/EPISC_X_par_Tm_nomS_D3.npy')

stat_EPISC = np.zeros((7,2))
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(X1[:,idx]))
    k2 = gaussian_kde(np.log10(X2[:,idx])) 
    k3 = gaussian_kde(np.log10(X3[:,idx]))
    k4 = gaussian_kde(np.log10(X4[:,idx]))
    pdf = k1(np.log10(x))*k2(np.log10(x))*k3(np.log10(x))*k4(np.log10(x))#/log_unif(x, 0.005, 2)**3
    pdf = pdf/simps(pdf,dx=0.0001)
    stat_EPISC[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
 
    
    
# Abs Bar plot
 
# width of the bars
barWidth = 0.3
 
# Choose the height of bars
bars1 = 10**stat_CHIR[:,0]
bars2 = 10**stat_EPISC[:,0]
 
# Choose the height of the error bars (use std deviation)
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
ax.set_ylim(0.005,2)
par_name = [r'$\theta_{0}$',r'$\theta_{T+}$',r'$\theta_{T-}$',r'$\theta_{S+}$',r'$\theta_{S-}$',r'$\theta_{F+}$',r'$\theta_{F-}$']
ax.set_xticks([r + barWidth for r in range(len(bars1))])
ax.set_xticklabels(par_name)
ax.set_ylabel('Rate [1/day]')
plt.legend()






# Rel Bar plot

# width of the bars
barWidth = 0.3
 
# Choose the height of bars
bars1 = 10**np.array([stat_CHIR[1,0]-stat_CHIR[2,0],stat_CHIR[3,0]-stat_CHIR[4,0],stat_CHIR[5,0]-stat_CHIR[6,0]])
bars2 = 10**np.array([stat_EPISC[1,0]-stat_EPISC[2,0],stat_EPISC[3,0]-stat_EPISC[4,0],stat_EPISC[5,0]-stat_EPISC[6,0]])
 
# Choose the height of the error bars (use std deviation)
#rel_errs = np.sqrt(stat_CHIR[:,1])/stat_CHIR[:,0]
#yer1 = bars1 * np.array([np.sqrt(rel_errs[1]**2+rel_errs[2]**2),np.sqrt(rel_errs[3]**2+rel_errs[4]**2),np.sqrt(rel_errs[5]**2+rel_errs[6]**2)])
#rel_errs = np.sqrt(stat_EPISC[:,1])/stat_EPISC[:,0] 
#yer2 = bars2 * np.array([np.sqrt(rel_errs[1]**2+rel_errs[2]**2),np.sqrt(rel_errs[3]**2+rel_errs[4]**2),np.sqrt(rel_errs[5]**2+rel_errs[6]**2)])
e1 = np.array([10**(stat_CHIR[1,0]-stat_CHIR[2,0]-np.sqrt(stat_CHIR[1,1])-np.sqrt(stat_CHIR[2,1])),10**(stat_CHIR[3,0]-stat_CHIR[4,0]-np.sqrt(stat_CHIR[3,1])-np.sqrt(stat_CHIR[4,1])),10**(stat_CHIR[5,0]-stat_CHIR[6,0]-np.sqrt(stat_CHIR[5,1])-np.sqrt(stat_CHIR[6,1]))])
e2 = np.array([10**(stat_CHIR[1,0]-stat_CHIR[2,0]+np.sqrt(stat_CHIR[1,1])+np.sqrt(stat_CHIR[2,1])),10**(stat_CHIR[3,0]-stat_CHIR[4,0]+np.sqrt(stat_CHIR[3,1])+np.sqrt(stat_CHIR[4,1])),10**(stat_CHIR[5,0]-stat_CHIR[6,0]+np.sqrt(stat_CHIR[5,1])+np.sqrt(stat_CHIR[6,1]))])
yer1 = [e1,e2]
e1 = np.array([10**(stat_EPISC[1,0]-stat_EPISC[2,0]-np.sqrt(stat_EPISC[1,1])-np.sqrt(stat_EPISC[2,1])),10**(stat_EPISC[3,0]-stat_EPISC[4,0]-np.sqrt(stat_EPISC[3,1])-np.sqrt(stat_EPISC[4,1])),10**(stat_EPISC[5,0]-stat_EPISC[6,0]-np.sqrt(stat_EPISC[5,1])-np.sqrt(stat_EPISC[6,1]))])
e2 = np.array([10**(stat_EPISC[1,0]-stat_EPISC[2,0]+np.sqrt(stat_EPISC[1,1])+np.sqrt(stat_EPISC[2,1])),10**(stat_EPISC[3,0]-stat_EPISC[4,0]+np.sqrt(stat_EPISC[3,1])+np.sqrt(stat_EPISC[4,1])),10**(stat_EPISC[5,0]-stat_EPISC[6,0]+np.sqrt(stat_EPISC[5,1])+np.sqrt(stat_EPISC[6,1]))])
yer2 = [e1,e2]

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Create bars
fig = plt.figure(figsize=(10,5))  
ax = plt.subplot(111)
ax.bar(r1, bars1, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='CHIR')
ax.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='EPISC') 
ax.hlines(1,-1,3,linestyles='dashed')
ax.vlines(r1, yer1[0], yer1[1])
ax.vlines(r2, yer2[0], yer2[1])

# general layout
#ax.set_ylim(0,2.8)
ax.set_xlim(-0.3,2.7)
ax.set_yscale('log')
par_name = [r'$\theta_{T+}/\theta_{T-}$',r'$\theta_{S+}/\theta_{S-}$',r'$\theta_{F+}/\theta_{F-}$']
ax.set_xticks([r + barWidth for r in range(len(bars1))])
ax.set_xticklabels(par_name)
ax.set_ylabel(r'Rate imbalance')
plt.legend()
 


    
