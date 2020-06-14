# Discrete time observation inference
# packages
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, "INSERT PATH TO DTO_X SCRIPT HERE")
import DTO_X
trajectory_X = DTO_X.trajectory_X
get_population_list = DTO_X.get_population_list
lhood_CD = DTO_X.lhood_CD
get_posterior = DTO_X.get_posterior
get_posterior_inhom = DTO_X.get_posterior_inhom
plot_all = DTO_X.plot_all
plot_all_inhom = DTO_X.plot_all_inhom

 
# Assumption: Best sampling strategy is to have approx. the same number of 
# unknown events between all snapshots. 
# mean Number of events in [t1,t2]: int[t1,t2] h0 = int[t1,t2] <A> (L_aa+L_ab+L_bb)
def get_optSST(par,T,N):
    ls = []
    dL = par[0]-par[2] # delta Lambda = L_aa-L_bb
    for n in range(1,N+1):
        ls.append(np.log( (N-n+(n-1)*np.exp(dL*T))/(N-1) )/dL)
    return ls



#%% create data fom model

# meta parameter
SMPL_LIM = 3
SMPL_DENS = 0.02
prior = 1 # start with flat prior

# simulation parameter
pop0 = np.array([1,1])
par = [1,1,0]
T = 2

# generate trajectory (transition data)
traj = trajectory_X(par,pop0,T)
pop = np.c_[ get_population_list(traj,pop0) , np.append(0,traj[:,1]) ]
print(pop[-1,0:2])


#%% do MCMC runs with uniform snapshot times

# create uniform snapshots
OUTPUTPATH = 'C:/Users/Liam/Desktop/Master/inference/snapshots/model_X/'#test/multi/2-1-1/uniform_SS/'
dT = 2
pop_SS = np.zeros((0,2))
for i in range(0,int(T/dT)+1):
    tlim = dT*i
    temp = pop[pop[:,2]<=tlim] # only entries which happen before tlim
    pop_SS = np.vstack([pop_SS,temp[-1,0:2]])

# calculate Likelihood given uniform Snapshot data and plot
_ = get_posterior(pop_SS,dT,T,prior,1,0,1,True,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
[output_par,output_traj,mA] = get_posterior(pop_SS,dT,T,prior,15000,5000,20,False,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
plot_all(output_par,mA,OUTPUTPATH,SMPL_LIM,pop,par,traj,T,dT)
    


#%% do MCMC runs with inhom snapshot times

# create inhom snapshots with optimal sampling strategy
OUTPUTPATH = 'C:/Users/Liam/Desktop/Master/inference/snapshots/model_X/test/' # multi/2-1-1/inhom_SS/'
T_arr = get_optSST(par,T,2)
pop_SS = np.zeros((0,2))
for i in range(0,len(T_arr)):
    tlim = T_arr[i]
    temp = pop[pop[:,2]<=tlim] # only entries which happen before tlim
    pop_SS = np.vstack([pop_SS,temp[-1,0:2]])

# calculate Likelihood given inhom Snapshot data
_ = get_posterior_inhom(pop_SS,T_arr,prior,600,0,10,True,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
[output_par,output_traj,mA] = get_posterior_inhom(pop_SS,T_arr,prior,15000,5000,20,False,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
plot_all_inhom(output_par,mA,OUTPUTPATH,SMPL_LIM,pop,par,traj,T_arr)
 


 

