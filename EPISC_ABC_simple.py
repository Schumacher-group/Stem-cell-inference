# Discrete time observation inference
# packages
import numpy as np
import scipy as sc

#%%  help functions

# loguniform sample
def loguniform(low, high, size):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

# log-uniform PDF between PAR_MIN and PAR_MAX
def log_unif(x, PAR_MIN, PAR_MAX):
    return 1 / (x * (np.log(PAR_MAX) - np.log(PAR_MIN)))

# product of log-uniform PDFs
def prior_logU(par,PAR_MIN, PAR_MAX):
    if (par > PAR_MIN).all() and (par < PAR_MAX).all():
        return np.prod(log_unif(par, PAR_MIN, PAR_MAX))
    else:
        return 0

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
        scale_arr = np.append(N_pop[0:8]*par[0], N_pop[[0,1,1,3,3,2,2,0,4,5,5,7,7,6,6,4,4,0,5,1,7,3,6,2]]*par[[5,6,3,4,6,5,4,3,5,6,3,4,6,5,4,3,2,1,2,1,2,1,2,1]])
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32))/scale_arr
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]
        
        if rtime[-1]>T or np.max(N_pop)>=15:
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
        scale_arr = np.append(N_pop[0:8] * par[0],
                              N_pop[[0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 4, 0, 5, 1, 7, 3, 6, 2]] * par[
                                  [5, 6, 3, 4, 6, 5, 4, 3, 5, 6, 3, 4, 6, 5, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1]])
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32))/scale_arr
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        idx = np.where(times == np.amin(times))[0][0]     
        rtime = np.append(rtime,rtime[-1]+times[idx])
        rtype = np.append(rtype,idx)
        N_pop += S[idx,:]
        
        if rtime[-1]>T or np.max(N_pop)>=15:
            break
    return N_pop


def project_population(pop,marker_unobs):
    if marker_unobs=='F':
        return np.array([np.sum(pop[0:2]),np.sum(pop[2:4]),np.sum(pop[4:6]),np.sum(pop[6:8])])
    elif marker_unobs=='S':
        return np.array([np.sum(pop[[0,2]]),np.sum(pop[[1,3]]),np.sum(pop[[4,6]]),np.sum(pop[[5,7]])])



def stat_decide(stat_sim,stat_true,eps):
    # median is allowed to vary by 1
    if np.any(np.abs(stat_sim[1]-stat_true[1])>1):
        return False
    # if rel deviation mean > eps: reject!
    elif np.any(np.abs(stat_sim[0]-stat_true[0]) > 1):
        return False
    # if rel deviation std > eps: reject!
    elif np.any(np.abs(stat_sim[2] - stat_true[2]) > 1):
        return False
    return True



# model X summary statistics
def statistics_X(par,T,T_marker,marker_unobs,MC_N):
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
    pop_init = pop_init_arr[np.random.randint(4)]
    pop_ls = np.zeros((0,4))
    for i in range(0,MC_N):
        # get population from simulation
        pop =  trajectory_EPISC_X(par,pop_init,T)
        # project population numers according to known markers
        pop_adj = project_population(pop,marker_unobs)
        pop_ls = np.vstack([pop_ls, pop_adj])

    stat = [np.mean(pop_ls,axis=0), np.median(pop_ls,axis=0), np.std(pop_ls,axis=0)]
    return stat
    
    
    
# model Y summary statistics
def statistics_Y(par,T,T_marker,marker_unobs,MC_N):
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
    pop_init = pop_init_arr[np.random.randint(4)]
    pop_ls = np.zeros((0,4))
    for i in range(0,MC_N):
        # get population from simulation
        pop =  trajectory_EPISC_Y(par,pop_init,T)
        # project population numers according to known markers
        pop_adj = project_population(pop,marker_unobs)
        pop_ls = np.vstack([pop_ls, pop_adj])

    stat = [np.mean(pop_ls,axis=0), np.median(pop_ls,axis=0), np.std(pop_ls,axis=0)]
    return stat







# %% Posterior calculation: ABC rejection sampling

def get_ABCrej_X(stat_true,T_marker,marker_unobs, T, N_s, PAR_MIN, PAR_MAX, eps):
    # loop over ABC iterations
    output_par = np.zeros((0, 7))
    for i in range(0, N_s):
        if i % 100 == 0:
            print(i)
        # propose new parameters from random step
        #par = np.exp(np.random.uniform(np.log(PAR_MIN), np.log(PAR_MAX), 7))
        par = np.random.uniform(PAR_MIN, PAR_MAX, 7)

        # Accept/Reject proposed parameters
        stat_sim = statistics_X(par,T,T_marker,marker_unobs,200)
        if stat_decide(stat_sim, stat_true, eps):
            output_par = np.vstack([output_par, par])

    print('Finished. Average acceptance prob: ', len(output_par)/N_s)
    return output_par


def get_ABCrej_Y(stat_true,T_marker,marker_unobs, T, N_s, PAR_MIN, PAR_MAX, eps):
    # loop over ABC iterations
    output_par = np.zeros((0, 7))
    for i in range(0, N_s):
        if i % 100 == 0:
            print(i)
        # propose new parameters from random step
        #par = np.exp(np.random.uniform(np.log(PAR_MIN), np.log(PAR_MAX), 7))
        par = np.random.uniform(PAR_MIN, PAR_MAX, 7)

        # Accept/Reject proposed parameters
        stat_sim = statistics_Y(par,T,T_marker,marker_unobs,200)
        if stat_decide(stat_sim, stat_true, eps):
            output_par = np.vstack([output_par, par])

    print('Finished. Average acceptance prob: ', len(output_par)/N_s)
    return output_par



# %% Posterior calculation: ABC MCMC

def get_ABCMCMC_X(stat_true,T_marker,marker_unobs, T, N_MCMC, PAR_MIN, PAR_MAX, eps):
    # loop over ABC iterations
    output_par = np.zeros((0, 7))
    n = 0
    par_init = loguniform(PAR_MIN, PAR_MAX, 7)
    for i in range(0, N_MCMC):
        if i % 100 == 0:
            print(i)

        while True:
            par = par_init + np.random.normal(0, 0.4*par_init, 7)
            if np.all(par>PAR_MIN) and np.all(par<PAR_MAX): break

        # Accept/Reject proposed parameters
        stat_sim = statistics_X(par,T,T_marker,marker_unobs,400)
        if stat_decide(stat_sim, stat_true, eps):
            ratio = sc.stats.norm.pdf(par_init,par,0.4*par)/sc.stats.norm.pdf(par,par_init,0.4*par_init)
            acc_prob = np.min([1,np.prod(ratio)])
            if np.random.rand() < acc_prob:
                par_init = par
                n += 1
        output_par = np.vstack([output_par, par_init])

    print('Finished. Average acceptance prob: ', int(100*n/N_MCMC))
    return output_par



def get_ABCMCMC_Y(stat_true,T_marker,marker_unobs, T, N_MCMC, PAR_MIN, PAR_MAX, eps):
    # loop over ABC iterations
    output_par = np.zeros((0, 7))
    n = 0
    par_init = loguniform(PAR_MIN, PAR_MAX, 7)
    for i in range(0, N_MCMC):
        if i % 200 == 0:
            print(i)

        while True:
            par = par_init + np.random.normal(0, 0.4*par_init, 7)
            if np.all(par>PAR_MIN) and np.all(par<PAR_MAX): break

        # Accept/Reject proposed parameters
        stat_sim = statistics_Y(par,T,T_marker,marker_unobs,400)
        if stat_decide(stat_sim, stat_true, eps):
            ratio = sc.stats.norm.pdf(par_init,par,0.4*par)/sc.stats.norm.pdf(par,par_init,0.4*par_init)
            acc_prob = np.min([1,prior_logU(par,PAR_MIN, PAR_MAX)/prior_logU(par_init,PAR_MIN, PAR_MAX)*np.prod(ratio)])
            if np.random.rand() < acc_prob:
                par_init = par
                n += 1
        output_par = np.vstack([output_par, par_init])

    print('Finished. Average acceptance prob: ', int(100*n/N_MCMC))
    return output_par



# %% Bayes factor: ABC rejection

def get_ABC_BF(stat_true,T_marker,marker_unobs, T, N_MC, PAR_MIN, PAR_MAX):
    # loop over ABC iterations
    output_parX = np.zeros((0, 7))
    output_parY = np.zeros((0, 7))
    for i in range(0, N_MC):
        print(i)
        # sample parameters from priors (same for X and Y)
        par = loguniform(PAR_MIN,PAR_MAX,7)

        # Accept/Reject proposed X parameters
        stat_simX = statistics_X(par,T,T_marker,marker_unobs,200)
        if stat_decide(stat_simX, stat_true, 0):
            output_parX = np.vstack([output_parX, par])

        # Accept/Reject proposed Y parameters
        stat_simY = statistics_Y(par,T,T_marker,marker_unobs,200)
        if stat_decide(stat_simY, stat_true, 0):
            output_parY = np.vstack([output_parY, par])

    print('Finished. Average acceptance prob: ', len(output_parX)/N_MC)
    return [output_parX,output_parY]


# %% Bayes factor of combined CHIR data 
    
def stat_diff(stat_sim,stat_true):
    # abs median deviation and sum over individual cell states
    med_dev = np.sum(np.abs(stat_sim[1]-stat_true[1]))
    # mean deviation
    mean_dev = np.sum(np.abs(stat_sim[0]-stat_true[0]))
    # std deviation
    std_dev = np.sum(np.abs(stat_sim[2] - stat_true[2]))
    return [med_dev,mean_dev,std_dev]

 


def get_ABC_BF_comb(stat_true_ls, T_ls, T_marker_ls, marker_unobs_ls, N_MC, PAR_MIN, PAR_MAX):
    # loop over ABC iterations
    output_parX = np.zeros((0, 7))
    output_parY = np.zeros((0, 7))

    for i in range(0, N_MC):
        print(i)
        # sample parameters from priors (same for X and Y)
        par = loguniform(PAR_MIN,PAR_MAX,7)

        # iterate over Model X initial coniditions
        d = np.zeros(3)
        for ic in range(0,8):
           T = T_ls[ic]
           T_marker = T_marker_ls[ic]
           marker_unobs = marker_unobs_ls[ic]
           stat_true = stat_true_ls[ic]
            
           # calculate summary statistics
           stat_simX = statistics_X(par,T,T_marker,marker_unobs,200)
           d += stat_diff(stat_simX, stat_true)

        if np.sum(d)<55: # epsilon=55
            output_parX = np.vstack([output_parX, par])


        # iterate over Model Y initial coniditions
        d = np.zeros(3)
        for ic in range(0,8):
            T = T_ls[ic]
            T_marker = T_marker_ls[ic]
            marker_unobs = marker_unobs_ls[ic]
            stat_true = stat_true_ls[ic]

            # calculate summary statistics
            stat_simY = statistics_Y(par,T,T_marker,marker_unobs,200)
            d += stat_diff(stat_simY, stat_true)    

        if np.sum(d)<55: # epsilon=55
            output_parY = np.vstack([output_parY, par])
            
        if i%1000==999:
            np.save('/home/ruske/Desktop/Stem-cell-inference-master/Test_ModelX',output_parX)
            np.save('/home/ruske/Desktop/Stem-cell-inference-master/Test_ModelY',output_parY)
            print('Average acceptance prob for Model X: ', len(output_parX)/N_MC)
            print('\nAverage acceptance prob for Model Y: ', len(output_parY)/N_MC) 
  

    return [output_parX,output_parY]
