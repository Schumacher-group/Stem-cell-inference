# Discrete time observation inference
# packages
import numpy as np
from scipy.stats import gaussian_kde

#%%  help functions 

def trajectory_X_v2(par,N_init,N_target,T):
    S = np.array([[1,0],[0,1],[-1,2]])
    [L_aa,L_ab,L_bb] = par
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(3))/np.array([N_pop[0]*L_aa, N_pop[0]*L_ab, N_pop[0]*L_bb])
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]
        
        if rtime[-1]>T:
            break
    
    out = np.zeros(len(N_target))
    for i in range(0,len(N_target)):
        out[i] = int(np.all(N_pop==N_target[i]))
    return out



def trajectory_Y_v2(par,N_init,N_target,T):
    S = np.array([[1,0],[-1,1],[0,1],[1,-1]])
    [L_a,k_ab,L_b,k_ba] = par
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(4))/np.array([N_pop[0]*L_a, N_pop[0]*k_ab, N_pop[1]*L_b, N_pop[1]*k_ba ])
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]

        if rtime[-1]>T:
            break
        
    out = np.zeros(len(N_target))
    for i in range(0,len(N_target)):
        out[i] = int(np.all(N_pop==N_target[i]))
    return out



def get_lhd_SS_X(N,par,T,pop0,pop1):
    n = np.zeros(len(pop1))
    for i in range(0,N):
        n += trajectory_X_v2(par,pop0,pop1,T)
    return n 

    
    
def get_lhd_SS_Y(N,par,T,pop0,pop1):
    n = np.zeros(len(pop1))
    for i in range(0,N):
        n += trajectory_Y_v2(par,pop0,pop1,T)
    return n 



#%% MCMC function with 2 boundary conditions (only 2 snapshots)

def get_power_posterior_X(MC_temp,pop0,pop1,T,prior,N_MC,burn_in,N_traj,PAR_MIN,PAR_MAX):
    
    # guess initial parameter
    par_old = np.array([1,1,1]) 
    print('Start MCMC with par_init: ',par_old)    
    lhd_SS_old = get_lhd_SS_X(N_traj,par_old,T,pop0,pop1)
    
# loop over MCMC iterations        
    output_par = np.zeros((0,3))
    lhd_ls = []
    acc_prob_ls = []
    
    for i in range(0,N_MC):
        if i%100==0:
            print(i)
    # propose new trajectory given current rate constants
        while True:
            par = par_old + np.random.normal(0,0.4,3)
            if np.all(par>=PAR_MIN) and np.all(par<=PAR_MAX):
                break
        
    # Calculate likelihood of proposed step
        lhd_SS_new = get_lhd_SS_X(N_traj,par,T,pop0,pop1)
        
        # accept/reject it with MH-step
        if np.prod(lhd_SS_old)==0 or prior(par_old)==0:
            acc_prob = 1
        else:
            acc_prob = np.min([1,(prior(par)/prior(par_old))*np.prod(lhd_SS_new/lhd_SS_old)**MC_temp])
        if np.random.rand()<acc_prob:
            par_old = par
            lhd_SS_old = lhd_SS_new
            acc_prob_ls.append(acc_prob)
        
        if i>=burn_in:
            output_par = np.vstack([output_par,par_old])
            lhd_ls.append(np.prod(lhd_SS_old))
    
    print('Finished. Average acceptance prob: ',np.mean(acc_prob_ls))   
    return [output_par,np.array(lhd_ls),np.mean(acc_prob_ls)]



def get_power_posterior_Y(MC_temp,pop0,pop1,T,prior,N_MC,burn_in,N_traj,PAR_MIN,PAR_MAX):
    
    # guess initial parameter
    par_old = np.array([1,1,1,1]) 
    print('Start MCMC with par_init: ',par_old)
    lhd_SS_old = get_lhd_SS_Y(N_traj,par_old,T,pop0,pop1)
    
# loop over MCMC iterations        
    output_par = np.zeros((0,4))
    lhd_ls = []
    acc_prob_ls = []
    
    for i in range(0,N_MC):
        if i%100==0:
            print(i)
    # propose new trajectory given current rate constants
        while True:
            par = par_old + np.random.normal(0,0.4,4)
            if np.all(par>=PAR_MIN) and np.all(par<=PAR_MAX):
                break
        
    # Calculate likelihood of proposed step
        lhd_SS_new = get_lhd_SS_Y(N_traj,par,T,pop0,pop1)
        
        # accept/reject it with MH-step
        if np.prod(lhd_SS_old)==0 or prior(par_old)==0:
            acc_prob = 1
        else:
            acc_prob = np.min([1,(prior(par)/prior(par_old))*np.prod(lhd_SS_new/lhd_SS_old)**MC_temp])
        if np.random.rand()<acc_prob:
            par_old = par
            lhd_SS_old = lhd_SS_new
            acc_prob_ls.append(acc_prob)
        
        if i>=burn_in:
            output_par = np.vstack([output_par,par_old])
            lhd_ls.append(np.prod(lhd_SS_old))
    
    print('Finished. Average acceptance prob: ',np.mean(acc_prob_ls))   
    return [output_par,np.array(lhd_ls),np.mean(acc_prob_ls)]




#%% MC prodct space search with posterior determination


def prod_space_XY_fast(pop0,pop1,T,N_MC,N_traj,prior,PAR_MIN,PAR_MAX):
# X: mdl_idx=0, Y: mdl_idx=1      
    # use if prior == pseudo-priors: acc prob independent of prior!
    
    # guess initial parameter and model
    mdl_idx_old = 0 
    par_old = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),3))
    print('Start PS-MC with par_init: ',par_old)    
    lhd_SS_old = get_lhd_SS_X(N_traj,par_old,T,pop0,pop1)
    
# loop over MCMC iterations        
    mdl_idx_ls = []
    acc_prob_ls = []
    
    for i in range(0,N_MC):
        if i%5==0:
            print(i)
        
        # propose new model
        if np.random.rand()<0.5:
            if mdl_idx_old==0: 
                mdl_idx=1
                # sample model parameter from pseudo-prior
                par = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),4))
                # Calculate likelihood of proposed step
                lhd_SS_new = get_lhd_SS_Y(N_traj,par,T,pop0,pop1)
                # accept/reject it with MH-step
                if np.prod(lhd_SS_old)==0: 
                    acc_prob = 1    
                else: 
                    acc_prob = np.min([1,np.prod(lhd_SS_new/lhd_SS_old)])
        
            elif mdl_idx_old==1:
                mdl_idx=0
                # sample model parameter from pseudo-prior
                par = np.exp(np.random.uniform(np.log(PAR_MIN),np.log(PAR_MAX),3))
                # Calculate likelihood of proposed step
                lhd_SS_new = get_lhd_SS_X(N_traj,par,T,pop0,pop1)
                # accept/reject it with MH-step
                if np.prod(lhd_SS_old)==0: 
                    acc_prob = 1    
                else: 
                    acc_prob = np.min([1,np.prod(lhd_SS_new/lhd_SS_old)])
            
            if np.random.rand()<acc_prob:
                mdl_idx_old = mdl_idx
                par_old = par
                lhd_SS_old = lhd_SS_new


        # update parameters in current model
        else:
            if mdl_idx_old==0:
                # propose new trajectory given current rate constants
                while True:
                    par = par_old + np.random.normal(0,0.4,3)
                    if np.all(par>=0) and np.all(par<=PAR_MAX):
                        break
                # Calculate likelihood of proposed step
                lhd_SS_new = get_lhd_SS_X(N_traj,par,T,pop0,pop1)
                # accept/reject it with MH-step
                if np.prod(lhd_SS_old)==0 or prior(par_old)==0:
                    acc_prob = 1
                else:
                    acc_prob = np.min([1,(prior(par)/prior(par_old))*np.prod(lhd_SS_new/lhd_SS_old)])
            
            elif mdl_idx_old==1:
                # propose new trajectory given current rate constants
                while True:
                    par = par_old + np.random.normal(0,0.4,4)
                    if np.all(par>=0) and np.all(par<=PAR_MAX):
                        break
                # Calculate likelihood of proposed step
                lhd_SS_new = get_lhd_SS_Y(N_traj,par,T,pop0,pop1)
                # accept/reject it with MH-step
                if np.prod(lhd_SS_old)==0 or prior(par_old)==0:
                    acc_prob = 1
                else:
                    acc_prob = np.min([1,(prior(par)/prior(par_old))*np.prod(lhd_SS_new/lhd_SS_old)])

            if np.random.rand()<acc_prob:
                par_old = par
                lhd_SS_old = lhd_SS_new
        
        
        acc_prob_ls.append(acc_prob)
        mdl_idx_ls.append(mdl_idx_old)
    
    print('Finished. Average acceptance prob: ',np.mean(acc_prob_ls))   
    return np.array(mdl_idx_ls)

