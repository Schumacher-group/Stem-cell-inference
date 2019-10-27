# Discrete time observation inference
# packages
import numpy as np


#%%  help functions
def trajectory_EPISC_X(par,N_init,T):
    S = np.zeros((32,8),dtype=int)
    S[0,0] = 1
    S[1,1] = 1
    S[2,2] = 1
    S[3,3] = 1
    S[4,4] = 1
    S[5,5] = 1
    S[6,6] = 1
    S[7,7] = 1
    S[8, :] = [-1, 2, 0, 0, 0, 0, 0, 0]
    S[9, :] = [2, -1, 0, 0, 0, 0, 0, 0]
    S[10, :] = [0, -1, 0, 2, 0, 0, 0, 0]
    S[11, :] = [0, 2, 0, -1, 0, 0, 0, 0]
    S[12, :] = [0, 0, 2, -1, 0, 0, 0, 0]
    S[13, :] = [0, 0, -1, 2, 0, 0, 0, 0]
    S[14, :] = [2, 0, -1, 0, 0, 0, 0, 0]
    S[15, :] = [-1, 0, 2, 0, 0, 0, 0, 0]
    S[16, :] = [0, 0, 0, 0, -1, 2, 0, 0]
    S[17, :] = [0, 0, 0, 0, 2, -1, 0, 0]
    S[18, :] = [0, 0, 0, 0, 0, -1, 0, 2]
    S[19, :] = [0, 0, 0, 0, 0, 2, 0, -1]
    S[20, :] = [0, 0, 0, 0, 0, 0, 2, -1]
    S[21, :] = [0, 0, 0, 0, 0, 0, -1, 2]
    S[22, :] = [0, 0, 0, 0, 2, 0, -1, 0]
    S[23, :] = [0, 0, 0, 0, -1, 0, 2, 0]
    S[24, :] = [2, 0, 0, 0, -1, 0, 0, 0]
    S[25, :] = [-1, 0, 0, 0, 2, 0, 0, 0]
    S[26, :] = [0, 2, 0, 0, 0, -1, 0, 0]
    S[27, :] = [0, -1, 0, 0, 0, 2, 0, 0]
    S[28, :] = [0, 0, 0, 2, 0, 0, 0, -1]
    S[29, :] = [0, 0, 0, -1, 0, 0, 0, 2]
    S[30, :] = [0, 0, 2, 0, 0, 0, -1, 0]
    S[31, :] = [0, 0, -1, 0, 0, 0, 2, 0]
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        scale_arr = np.append(N_pop[0:8]*par[0], N_pop[[0,1,1,3,3,2,2,0,4,5,5,7,7,6,6,4,4,0,5,1,7,3,6,2]]*par[1:25])
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32))/scale_arr
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]
        
        if rtime[-1]>T:
            break
    return N_pop



def trajectory_EPISC_Y(par,N_init,T):
    S = np.zeros((32,8),dtype=int)
    S[0,0] = 1
    S[1,1] = 1
    S[2,2] = 1
    S[3,3] = 1
    S[4,4] = 1
    S[5,5] = 1
    S[6,6] = 1
    S[7,7] = 1
    S[8,:] = [-1,1,0,0,0,0,0,0]
    S[9,:] = [1,-1,0,0,0,0,0,0]
    S[10,:] = [0,-1,0,1,0,0,0,0]
    S[11,:] = [0,1,0,-1,0,0,0,0]
    S[12,:] = [0,0,1,-1,0,0,0,0]
    S[13,:] = [0,0,-1,1,0,0,0,0]
    S[14,:] = [1,0,-1,0,0,0,0,0]
    S[15,:] = [-1,0,1,0,0,0,0,0]
    S[16,:] = [0,0,0,0,-1,1,0,0]
    S[17,:] = [0,0,0,0,1,-1,0,0]
    S[18,:] = [0,0,0,0,0,-1,0,1]
    S[19,:] = [0,0,0,0,0,1,0,-1]
    S[20,:] = [0,0,0,0,0,0,1,-1]
    S[21,:] = [0,0,0,0,0,0,-1,1]
    S[22,:] = [0,0,0,0,1,0,-1,0]
    S[23,:] = [0,0,0,0,-1,0,1,0]
    S[24,:] = [1,0,0,0,-1,0,0,0]
    S[25,:] = [-1,0,0,0,1,0,0,0]
    S[26,:] = [0,1,0,0,0,-1,0,0]
    S[27,:] = [0,-1,0,0,0,1,0,0]
    S[28,:] = [0,0,0,1,0,0,0,-1]
    S[29,:] = [0,0,0,-1,0,0,0,1]
    S[30,:] = [0,0,1,0,0,0,-1,0]
    S[31,:] = [0,0,-1,0,0,0,1,0]
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        scale_arr = np.append(N_pop[0:8]*par[0], N_pop[[0,1,1,3,3,2,2,0,4,5,5,7,7,6,6,4,4,0,5,1,7,3,6,2]]*par[1:25])
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32))/scale_arr
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]
        
        if rtime[-1]>T:
            break
    return N_pop



def project_population(pop,marker_unobs):
    if marker_unobs=='F':
        return np.array([np.sum(pop[0:2]),np.sum(pop[2:4]),np.sum(pop[4:6]),np.sum(pop[6:8])])
    elif marker_unobs=='S':
        return np.array([np.sum(pop[[0,2]]),np.sum(pop[[1,3]]),np.sum(pop[[4,6]]),np.sum(pop[[5,7]])])



# model X likelihood if initial population has marker T+
def get_lhd_EPISC_X(N,par,T,pop_fin,T_marker,marker_unobs):
    n = np.zeros(len(pop_fin))  
    # if only T+ known: iterate through states with pop_init T+F+S+, T+F+S-, T+F-S+, T+F-S-
    if T_marker==1:
        pop_init_arr = np.array([[1,0,0,0,0,0,0,0],
                                 [0,1,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0],
                                 [0,0,0,1,0,0,0,0]])
    # if only T- known: iterate through states with pop_init T-F+S+, T-F+S-, T-F-S+, T-F-S-
    elif T_marker==0:
        pop_init_arr = np.array([[0,0,0,0,1,0,0,0],
                                 [0,0,0,0,0,1,0,0],
                                 [0,0,0,0,0,0,1,0],
                                 [0,0,0,0,0,0,0,1]])
    for pop_init in pop_init_arr:
        for i in range(0,N):
            # get population from simulation
            pop =  trajectory_EPISC_X(par,pop_init,T)
            # project population numers according to known markers
            pop_adj = project_population(pop,marker_unobs)
            # add up observation numbers of simulation for all data points
            for j in range(0,len(pop_fin)):
                n[j] += int(np.all(np.abs(pop_adj - pop_fin[j]) <= 1))
                #n[j] += int(np.all(pop_adj==pop_fin[j]))
    return n
    
    
    
# model Y likelihood if initial population has marker T+
def get_lhd_EPISC_Y(N,par,T,pop_fin,T_marker,marker_unobs):
    n = np.zeros(len(pop_fin))  
    # if only T+ known: iterate through states with pop_init T+F+S+, T+F+S-, T+F-S+, T+F-S-
    if T_marker==1:
        pop_init_arr = np.array([[1,0,0,0,0,0,0,0],
                                 [0,1,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0],
                                 [0,0,0,1,0,0,0,0]])
    # if only T- known: iterate through states with pop_init T-F+S+, T-F+S-, T-F-S+, T-F-S-
    elif T_marker==0:
        pop_init_arr = np.array([[0,0,0,0,1,0,0,0],
                                 [0,0,0,0,0,1,0,0],
                                 [0,0,0,0,0,0,1,0],
                                 [0,0,0,0,0,0,0,1]])
    for pop_init in pop_init_arr:
        for i in range(0,N):
            # get population from simulation
            pop =  trajectory_EPISC_Y(par,pop_init,T)
            # project population numers according to known markers
            pop_adj = project_population(pop,marker_unobs)
            # add up observation numbers of simulation for all data points
            for j in range(0,len(pop_fin)):
                #n[j] += int(np.all(pop_adj == pop_fin[j]))
                n[j] += int(np.all(np.abs(pop_adj-pop_fin[j])<=1))
    return n







# %% Posterior calculation

def get_posterior_X(pop1,T_marker,marker_unobs, T, prior, N_MC, burn_in, N_traj, PAR_MIN, PAR_MAX):
    # sample initial parameter from log-unif distribution
    par_old = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),25))
    print('Start MCMC with par_init: ', par_old)
    lhd_SS_old = get_lhd_EPISC_X(N_traj,par_old,T,pop1,T_marker,marker_unobs)

    # loop over MCMC iterations
    output_par = np.zeros((0, 25))
    lhd_ls = []
    acc_prob_ls = []

    for i in range(0, N_MC):
        if i % 5 == 0:
            print(i)
        # propose new parameters from random step
        while True:
            par = par_old + np.random.normal(0, 0.1, 25)
            if np.all(par >= PAR_MIN) and np.all(par <= PAR_MAX):
                break

        # Calculate likelihood of proposed step
        lhd_SS_new = get_lhd_EPISC_X(N_traj,par,T,pop1,T_marker,marker_unobs)

        # accept/reject it with MH-step
        if np.any(lhd_SS_old==0) or prior(par_old) == 0:
            acc_prob = 1
        else:
            acc_prob = np.min([1, (prior(par) / prior(par_old)) * np.prod(lhd_SS_new / lhd_SS_old)])
        if np.random.rand() < acc_prob:
            par_old = par
            lhd_SS_old = lhd_SS_new
        acc_prob_ls.append(acc_prob)

        if i >= burn_in:
            output_par = np.vstack([output_par, par_old])
            lhd_ls.append(np.prod(lhd_SS_old))

    print('Finished. Average acceptance prob: ', np.mean(acc_prob_ls))
    return [output_par, np.array(lhd_ls), np.mean(acc_prob_ls)]


def get_posterior_Y(pop1,T_marker,marker_unobs, T, prior, N_MC, burn_in, N_traj, PAR_MIN, PAR_MAX):
    # sample initial parameter from log-unif distribution
    par_old = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),25))
    print('Start MCMC with par_init: ', par_old)
    lhd_SS_old = get_lhd_EPISC_Y(N_traj,par_old,T,pop1,T_marker,marker_unobs)

    # loop over MCMC iterations
    output_par = np.zeros((0, 25))
    lhd_ls = []
    acc_prob_ls = []

    for i in range(0, N_MC):
        if i % 5 == 0:
            print(i)
        # propose new parameters from random step
        while True:
            par = par_old + np.random.normal(0, 0.1, 25)
            if np.all(par >= PAR_MIN) and np.all(par <= PAR_MAX):
                break
        #par = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),25))

        # Calculate likelihood of proposed step
        lhd_SS_new = get_lhd_EPISC_Y(N_traj,par,T,pop1,T_marker,marker_unobs)

        # accept/reject it with MH-step
        if np.any(lhd_SS_old==0) or prior(par_old) == 0:
            acc_prob = 1
        else:
            acc_prob = np.min([1, (prior(par) / prior(par_old)) * np.prod(lhd_SS_new / lhd_SS_old)])
        if np.random.rand() < acc_prob:
            par_old = par
            lhd_SS_old = lhd_SS_new
        acc_prob_ls.append(acc_prob)

        if i >= burn_in:
            output_par = np.vstack([output_par, par_old])
            lhd_ls.append(np.prod(lhd_SS_old))

    print('Finished. Average acceptance prob: ', np.mean(acc_prob_ls))
    return [output_par, np.array(lhd_ls), np.mean(acc_prob_ls)]
