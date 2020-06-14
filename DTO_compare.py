# Discrete time observation inference
# packages
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "INSERT PATH TO DTO_X/Y SCRIPT HERE")
import DTO_X
import DTO_Y
import DTO_MS


# Here: log-uniform prior between PAR_MIN and PAR_MAX
def log_unif(x,PAR_MIN,PAR_MAX):
    return 1/(x*(np.log(PAR_MAX)-np.log(PAR_MIN)))

def prior_logU(par):
    if (par > PAR_MIN).all() and (par < PAR_MAX).all():
        return np.prod(log_unif(par,PAR_MIN,PAR_MAX))
    else:
        return 0


#%% create data fom model X

# simulation parameter
pop0 = np.array([3,3])
par = [1,1,1]
T = 1
N = 16

# generate trajectory (transition data)
pop1 = np.zeros((0,2))
for i in range(0,N):
    traj = DTO_X.trajectory_X(par,pop0,T)
    pop = np.c_[ DTO_X.get_population_list(traj,pop0) , np.append(0,traj[:,1]) ]
    pop1 = np.vstack([pop1,pop[-1,0:2]])





#%% create data fom model Y

# simulation parameter
pop0 = np.array([3,3])
par = [1,1,1,1]
T = 1
dT = 1

# generate trajectory (transition data)
traj = DTO_Y.trajectory_Y(par,pop0,T)
pop = np.c_[ DTO_Y.get_population_list(traj,pop0) , np.append(0,traj[:,1]) ]
print(pop[-1,0:2])

# create uniform snapshots
pop_SS = np.zeros((0,2))
for i in range(0,int(T/dT)+1):
    tlim = dT*i
    temp = pop[pop[:,2]<=tlim] # only entries which happen before tlim
    pop_SS = np.vstack([pop_SS,temp[-1,0:2]])

    
    
#%% create custom data

T = 1
pop0 = np.array([3,3])
pop1 = np.array([[8,8]])    




    
    
# %% Harmonic Mean estimation
    
PAR_MIN = 0.1
PAR_MAX = 2

# HM estimation in model X
# get_power_posterior(1,...) is just normal posterior
[par_ls,lhd_ls,mean_acc_prob] = DTO_MS.get_power_posterior_X(1,pop0,pop1,T,prior_logU,500,0,800,PAR_MIN,PAR_MAX)
HM_X = np.zeros(len(lhd_ls)-1)
for i in range(1,len(lhd_ls)):
    HM_X[i-1] = 1/np.mean(1/lhd_ls[0:i])


# HM estimation in model Y
# get_power_posterior(1,...) is just normal posterior
[par_ls,lhd_ls,mean_acc_prob] = DTO_MS.get_power_posterior_Y(1,pop0,pop1,T,prior_logU,500,0,800,PAR_MIN,PAR_MAX)
HM_Y = np.zeros(len(lhd_ls)-1)
for i in range(1,len(lhd_ls)):
    HM_Y[i-1] = 1/np.mean(1/lhd_ls[0:i])




# %% Thermodynamic Integration

PAR_MIN = 0.1
PAR_MAX = 2

# TI in model X
N_temp = 4 # set temperature schedule. Choose one with clustering towards prior (t=0). See paper
temp_arr = np.linspace(0,1,N_temp)**5 # choose power=5. Maybe try more later...
loglhdX_expval = np.zeros(N_temp)
for i in range(0,N_temp):
    MC_temp = temp_arr[i]
    [par_ls,lhd_ls,mean_acc_prob] = DTO_MS.get_power_posterior_X(MC_temp,pop0,pop1,T,prior_logU,1000,500,800,PAR_MIN,PAR_MAX)
    loglhdX_expval[i] = np.mean(np.log(lhd_ls[lhd_ls!=0])) # filter zeros only relevant for MC_temp=0
    
log_mlhd = 0
for j in range(1,N_temp):
    dtemp = temp_arr[j]-temp_arr[j-1]
    log_mlhd += dtemp*(loglhdX_expval[j]+loglhdX_expval[j-1])/2
TI_X = np.exp(log_mlhd)



# TI in model Y
N_temp = 4 # set temperature schedule. Choose one with clustering towards prior (t=0). See paper
temp_arr = np.linspace(0,1,N_temp)**5 # choose power=5. Maybe try more later...
loglhdY_expval = np.zeros(N_temp)
for i in range(0,N_temp):
    MC_temp = temp_arr[i]
    [par_ls,lhd_ls,mean_acc_prob] = DTO_MS.get_power_posterior_Y(MC_temp,pop0,pop1,T,prior_logU,1000,500,800,PAR_MIN,PAR_MAX)
    loglhdY_expval[i] = np.mean(np.log(lhd_ls[lhd_ls!=0])) # filter zeros only relevant for MC_temp=0
    
log_mlhd = 0
for j in range(1,N_temp):
    dtemp = temp_arr[j]-temp_arr[j-1]
    log_mlhd += dtemp*(loglhdY_expval[j]+loglhdY_expval[j-1])/2
TI_Y = np.exp(log_mlhd)




# %% Product space search MC

# define priors and pseudo-priors for models
PAR_MIN = 0.1
PAR_MAX = 2

mdl_idx_ls = DTO_MS.prod_space_XY_fast(pop0,pop1,T,400,4000,prior_logU,PAR_MIN,PAR_MAX)
print(sum(mdl_idx_ls==0)/sum(mdl_idx_ls==1))

BR_PSMC = np.zeros(len(mdl_idx_ls)-1)
for i in range(1,len(mdl_idx_ls)):
    BR_PSMC[i-1] = sum(mdl_idx_ls[0:i]==0)/sum(mdl_idx_ls[0:i]==1)

plt.plot(BR_PSMC)
plt.plot(mdl_idx_ls)
plt.show()

# %% calculate Bayes Factor and plot 1

OUTPUTPATH = 'INSERT OUTPUT PATH HERE'
BR_HM = HM_X/HM_Y

fig = plt.figure(figsize=(20,6))

ax = plt.subplot(131)
#ax.set_title('Harmonic Mean approximation model X')
ax.plot(HM_X[0:])
ax.hlines(TI_X,0,1500, linestyle='--')
ax.set_xlabel('MC Samples')
ax.set_ylabel('Marginal Likelihood')

ax = plt.subplot(132)
#ax.set_title('Harmonic Mean approximation model Y')
ax.plot(HM_Y[0:])
ax.hlines(TI_Y,0,1500, linestyle='--')
ax.set_xlabel('MC Samples')
ax.set_ylabel('Marginal Likelihood')

ax = plt.subplot(133)
#ax.set_title(r'Bayes Factor $p(y|M_{X})/p(y|M_{Y})$')
ax.plot(BR_HM[0:])
ax.hlines(TI_X/TI_Y,0,1500, linestyle='--')
ax.set_xlabel('MC Samples')
ax.set_ylabel('Bayes Factor')

plt.tight_layout()
plt.savefig(OUTPUTPATH+'HM_TI_XdY1_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)
#plt.savefig(OUTPUTPATH+'Harmonic_Mean_Approx_test.png', bbox_inches="tight",dpi=600)




fig = plt.figure(figsize=(6,6))
plt.plot(HM_X[0:])
plt.hlines(TI_X,-50,1500, linestyle='--')
plt.xlabel('MC Samples')
plt.ylabel('Marginal Likelihood')
plt.xlim(-50,1500)
plt.tight_layout()
plt.savefig(OUTPUTPATH+'HM_TI_X_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)

fig = plt.figure(figsize=(6,6))
plt.plot(HM_Y[0:])
plt.hlines(TI_Y,-50,1500, linestyle='--')
plt.xlabel('MC Samples')
plt.ylabel('Marginal Likelihood')
plt.xlim(-50,1500)
plt.tight_layout()
plt.savefig(OUTPUTPATH+'HM_TI_Y_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)

fig = plt.figure(figsize=(6,6))
plt.plot(BR_HM[0:])
plt.hlines(TI_X/TI_Y,-50,1500, linestyle='--')
plt.xlabel('MC Samples')
plt.ylabel('Bayes Factor')
plt.ylim(0.4,1.5)
plt.xlim(-50,1500)
plt.tight_layout()
plt.savefig(OUTPUTPATH+'HM_TI_XdY_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)


# %% # %% calculate Bayes Factor and plot 2

OUTPUTPATH = 'INSERT OUTPUT PATH HERE'
BR_HM = HM_X/HM_Y

fig = plt.figure(figsize=(6,6))

ax = plt.subplot(111)
#ax.set_title(r'Bayes Factor $p(y|M_{X})/p(y|M_{Y})$')
ax.plot(BR_HM[0:])
ax.plot(BR_PSMC[0:])
ax.hlines(TI_X/TI_Y,-50,1500, linestyle='--')
ax.set_xlabel('MC samples')
ax.set_ylabel('Bayes Factor')
plt.ylim(0.3,1.5)
plt.xlim(-50,1500)

plt.tight_layout()
plt.savefig(OUTPUTPATH+'HM_TI_PSMC_XdY_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)
#plt.savefig(OUTPUTPATH+'Harmonic_Mean_Approx_test.png', bbox_inches="tight",dpi=600)


fig = plt.figure(figsize=(6,3))
plt.plot(mdl_idx_ls[1200:1260])
plt.yticks([0,1])
plt.xlabel('Iteration')
plt.ylabel('Model Indicator')
plt.savefig(OUTPUTPATH+'PSMC_XY_T'+str(T)+'-dT'+str(dT)+'_init'+str(pop0)+'_final'+str(pop1)+'.png', bbox_inches="tight",dpi=600)





