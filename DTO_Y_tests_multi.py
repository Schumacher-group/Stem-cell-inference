# Discrete time observation inference
# packages
import numpy as np
from scipy.stats import gaussian_kde
import sys
sys.path.insert(0, "C:\\Users\\Liam\\Desktop\\Master\\code")
import DTO_Y
trajectory_Y = DTO_Y.trajectory_Y
get_population_list = DTO_Y.get_population_list
lhood_CD = DTO_Y.lhood_CD
get_posterior = DTO_Y.get_posterior
get_posterior_inhom = DTO_Y.get_posterior_inhom
plot_all = DTO_Y.plot_all
#plot_all_inhom = DTO_Y.plot_all_inhom


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
SMPL_DENS = 0.05
prior = 1 # start with flat prior

# simulation parameter
pop0 = np.array([1,1])
par = [1,1,1,1]
T = 2

# generate trajectory (transition data)
traj = trajectory_Y(par,pop0,T)
pop = np.c_[ get_population_list(traj,pop0) , np.append(0,traj[:,1]) ]
print(pop[-1,0:2])



#%% do MCMC runs with uniform snapshot times

# create uniform snapshots
OUTPUTPATH = 'C:/Users/Liam/Desktop/Master/inference/snapshots/model_Y/test/'
dT = 0.5
pop_SS = np.zeros((0,2))
for i in range(0,int(T/dT)+1):
    tlim = dT*i
    temp = pop[pop[:,2]<=tlim] # only entries which happen before tlim
    pop_SS = np.vstack([pop_SS,temp[-1,0:2]])


# calculate Likelihood given uniform Snapshot data and plot
_ = get_posterior(pop_SS,dT,T,prior,400,0,10,True,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
[output_par,output_traj,mA] = get_posterior(pop_SS,dT,T,prior,15000,5000,1,False,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
plot_all(output_par,mA,OUTPUTPATH,SMPL_LIM,pop,par,traj,T,dT)
    



