# Discrete time observation inference
# packages
import numpy as np
from scipy.stats import skellam
from scipy.stats import poisson
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from random import choices
from scipy.integrate import simps
import seaborn as sns
from matplotlib.gridspec import GridSpec



#%%  MCMC help functions 

def shuffle_reactions(pop0,pop1,S2,r,w):
    r1 = r[0:2]
    r2_o = r[2]
    # pertubate r2 and update r1
    r2 = r2_o + skellam.rvs(w,w)
    #r1 = np.matmul(np.linalg.inv(S1), pop1-pop0-S2*r2) # not needed, S1 is unity matrix
    r1 = pop1-pop0-S2*r2
    if np.any(r1<0) or r2<0:
        while True:
            # pertubate r2 and update r1
            while True:
                r2 = r2_o + skellam.rvs(w,w,size=5)
                if np.any(r2>=0):
                    r2 = r2[r2>=0][0]
                    break
            r1 = pop1-pop0-S2*r2
            if np.all(r1>=0):
                break
    return np.array(np.append(r1,r2),dtype=int)



def get_reactions(pop0,pop1,S2):
    [da,db] = pop1-pop0
    if da>0: #(db<0 not possible)
        temp = np.array([da,db,0])
    else:
        temp = np.array([0,db+2*da,-da])
    return shuffle_reactions(pop0,pop1,S2,temp,5)



def get_population_list(reacts,pop0):
    # create trajectories from reaction events
    pop = np.zeros((len(reacts),2),dtype=int)
    temp = pop0
    for i in range(0,len(reacts)):
        if reacts[i,0]==0: # A->AA
            temp = temp+[1,0]
        elif reacts[i,0]==1: # A->AB
            temp = temp+[0,1]
        elif reacts[i,0]==2: # A->BB
            temp = temp+[-1,2]
        pop[i,:] = temp
    pop = np.vstack([pop0,pop]) # add pop0 as top row
    return pop



def sample_rates(prior,pop0,reacts,dT,SMPL_LIM,SMPL_DENS):
    # We assume a flat prior, thats why Posterior = normalised Lhood
    pop = get_population_list(reacts,pop0) # get population trajectory data
    par_new = []
    
    for i in range(0,3): # loop over parameter
        r_num = np.sum(reacts[:,0]==i) # number of reactions
        
        # if reaction happened in interval
        if len(reacts)>0: 
            integr = pop[0,0]*reacts[0,1] # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
            for j in range(1,len(reacts[:,1])):
                integr += pop[j,0]*(reacts[j,1]-reacts[j-1,1])
            integr += pop[-1,0]*(dT-reacts[-1,1])
        
        # if nothing happened -> Lhood_P = Lhood_CD
        else:
            integr = pop[0,0]*dT
            
        lhd = np.zeros(0)
        for c in np.arange(0,SMPL_LIM,SMPL_DENS):
            lhd = np.append(lhd,(c**r_num)*np.exp(-c*integr))
        post = lhd*prior/simps(lhd*prior, dx=SMPL_DENS) # normalise likelihood 
        rate = choices(np.arange(0,SMPL_LIM,SMPL_DENS), post)[0] # draw rate from posterior distr
        par_new.append(rate) 
    return par_new
    


def get_true_lhd(T,reacts,pop0,par):
    # get population trajectory data
    pop = get_population_list(reacts,pop0)
    total_rate = np.sum(par) # par=[L_aa,L_ab,L_bb]

    # calcultate full-data likelihood
    react_type = int(reacts[0,0])
    temp = pop[0,0]*par[react_type]*np.exp(-pop[0,0]*total_rate*reacts[0,1])
    
    v = reacts[:,0]        
    for i in range(1,len(reacts)):
        t_i = reacts[i,1]
        t_i1 = reacts[i-1,1]
        react_type = int(v[i])
        temp = temp*pop[i,0]*par[react_type]*np.exp(-pop[i,0]*total_rate*(t_i-t_i1))
    
    Lhood_T = temp*np.exp(-pop[-1,0]*total_rate*(T-reacts[-1,1]))
    return Lhood_T



#%% lin Poisson Model

# simulate homogenous Poisson process and then transform time to get lin. inhom. Poisson process
def get_reaction_times_linPois(pop0,pop1,r,dT):
    # calculate reaction hazards for all reactions at T0 and T1
    rtype = np.zeros(0,dtype=int)
    rtime_inhom = np.zeros(0)

    for i in range(0,3):
        rtype = np.append(rtype,i*np.ones(r[i],dtype=int))
        # reaction times are uniform distributed
        rtime_hom = np.random.uniform(0,dT,r[i])
        h0 = pop0[0] # set hazards just to population size, since prefactor (rate) canceles in expression
        h1 = pop1[0]
        # nonlinear trafo of time to get inhom reaction times
        if h0!=h1:
            rtime_inhom = np.append( rtime_inhom, (np.sqrt(h0**2+(h1**2 - h0**2)*rtime_hom/dT)-h0)*dT/(h1-h0) )
        else: # inhom Poisson process == hom Poisson process
            rtime_inhom = np.append( rtime_inhom, rtime_hom)

    reacts = np.array([rtype,rtime_inhom]).transpose()
    reacts = reacts[rtime_inhom.argsort()] # sort in increasing time order
    return reacts



def get_lin_Poisson_lhd(rhaz0,rhaz1,T,reacts):
    # calculate inhom Poisson likelihood
    temp = 1
    for i in range(0,len(reacts)):
        [v,t] = reacts[i,:]
        temp = temp* ( (1-t/T)*rhaz0[int(v)] + t/T*rhaz1[int(v)] )
    Lhood_P = temp*np.exp(-T*(np.sum(rhaz0)+np.sum(rhaz1))/2)   
    return Lhood_P



def get_lin_Poisson_PMF(rhaz0,rhaz1,r,T):
    # calculate PMF of Poisson process given observed reactions
    q = 1
    for i in range(0,3):
        mu = T*(rhaz0[i]+rhaz1[i])/2
        q = q*poisson.pmf(r[i], mu)
    return q



def get_MH_ratio_linPois(pop0,pop1,reacts,par_new,r,T):
    [L_aa,L_ab,L_bb] = par_new
    rhaz0 = [ L_aa*pop0[0], L_ab*pop0[0], L_bb*pop0[0] ]
    rhaz1 = [ L_aa*pop1[0], L_ab*pop1[0], L_bb*pop1[0] ] 
    
    # if reaction happened in interval
    if len(reacts)>0: 
        q = get_lin_Poisson_PMF(rhaz0,rhaz1,r,T)
        Lhood_P = get_lin_Poisson_lhd(rhaz0,rhaz1,T,reacts)
        Lhood_T = get_true_lhd(T,reacts,pop0,par_new)
    # if nothing happened -> Lhood_P = Lhood_CD
    else:
        q = get_lin_Poisson_PMF(rhaz0,rhaz1,r,T)
        Lhood_P = np.exp(-T*(np.sum(rhaz0)+np.sum(rhaz1))/2)   
        Lhood_T = np.exp(-pop1[0]*(L_aa+L_ab+L_bb)*T)
        if np.abs(1-Lhood_P/Lhood_T)>0.01: # just to make sure...
            print('Error!')
          
    # if reaction rate given is 0, but still proces observed: return 0
    if q==0 and Lhood_T==0 and Lhood_P==0:
        return 0
    else:
        return q*Lhood_T/Lhood_P




#%% exp Poisson Model

# simulate homogenous Poisson process and then transform time to get exp inhom. Poisson process
def get_reaction_times_expPois(pop0,pop1,r,T):
    # calculate reaction hazards for all reactions at T0 and T1
    rtype = np.zeros(0,dtype=int)
    rtime_inhom = np.zeros(0)
    N0 = pop0[0] # A-type population at start and end of interval
    N1 = pop1[0]

    for i in range(0,3):
        rtype = np.append(rtype,i*np.ones(r[i],dtype=int))
        unif_sample = np.random.uniform(size=r[i])
        
        # nonlinear trafo of time to get reaction times for inhom process
        if N0!=N1:
            dlogN = (np.log(N1)-np.log(N0))
            inhom_sample = np.log(1+unif_sample*(np.exp(dlogN)-1))*T/dlogN
            rtime_inhom = np.append( rtime_inhom,inhom_sample)
        else: # inhom Poisson process == hom Poisson process
            rtime_inhom = np.append( rtime_inhom, unif_sample)

    reacts = np.array([rtype,rtime_inhom]).transpose()
    reacts = reacts[rtime_inhom.argsort()] # sort in increasing time order
    return reacts



def get_exp_Poisson_lhd(T,reacts,pop0,pop1,par):    
    # calcultate exp Poisson likelihood
    # only A-type population relevant
    N0 = pop0[0]
    N1 = pop1[0] 
    
    # iterate over all reaction events
    react_type = int(reacts[0,0])    
    temp = par[react_type]*N0*(N1/N0)**(reacts[0,1]/T)
    
    v = reacts[:,0]        
    for i in range(1,len(reacts)):
        t_i = reacts[i,1]
        react_type = int(v[i])
        temp = temp*par[react_type]*N0*(N1/N0)**(t_i/T)
    if N0!=N1:
        dN = N1-N0
        dlogN = np.log(N1)-np.log(N0)
        Lhood_P = temp*np.exp(-T*np.sum(par)*dN/dlogN)
    else:
        Lhood_P = temp*np.exp(-T*np.sum(par)*N0)
    return Lhood_P



def get_exp_Poisson_PMF(pop0,pop1,par,r,T):
    # calculate PMF of Poisson process given observed reactions
    N0 = pop0[0]
    N1 = pop1[0]
    if N0!=N1:
        dN = N1-N0
        dlogN = np.log(N1)-np.log(N0)
        q = 1
        for i in range(0,3):
            mu = T*par[i]*dN/dlogN
            q = q*poisson.pmf(r[i], mu)
        return q
    else:
        q = 1
        for i in range(0,3):
            mu = T*par[i]*N0
            q = q*poisson.pmf(r[i], mu)
        return q



def get_MH_ratio_expPois(pop0,pop1,reacts,par_new,r,T):   
    # if reaction happened in interval
    if len(reacts)>0: 
        q = get_exp_Poisson_PMF(pop0,pop1,par_new,r,T)
        Lhood_P = get_exp_Poisson_lhd(T,reacts,pop0,pop1,par_new)
        Lhood_T = get_true_lhd(T,reacts,pop0,par_new)
    # if nothing happened -> Lhood_P = Lhood_CD
    else:
        q = get_exp_Poisson_PMF(pop0,pop1,par_new,r,T)
        Lhood_P = np.exp(-pop1[0]*np.sum(par_new)*T)
        Lhood_T = np.exp(-pop1[0]*np.sum(par_new)*T)
          
    # if reaction rate given is 0, but still proces observed: return 0
    if q==0 and Lhood_T==0 and Lhood_P==0:
        return 0
    else:
        return q*Lhood_T/Lhood_P



#%% MCMC function with homogenous spacing
        
def get_posterior(pop_SS,dT,T,prior,N,burn_in,w,plot,OUTPUTPATH,SMPL_LIM,SMPL_DENS):
    # splitting transition matrix into a quadratic, inveratble part S1 and the rest S2, r the same
    # S1 = S[:,0:2] not necessary, since unity matrix here
    S = np.array([[1,0,-1],[0,1,2]])
    S2 = S[:,2]
    
    # guess initial parameter
    par_init = [1,1,1] #r_old_sum/(T*(pop_SS[0,0]+pop_SS[-1,0])/2) # guess 'good' starting point (~ 1/mean A cells)
    print('Start MCMC with par_init: ',par_init)
    
# generate initial trajectory and calculate MH ratio
    reacts_old = []
    ratio_old = np.zeros(len(pop_SS)-1)
    r_old = np.zeros((len(pop_SS)-1,3))
    
    # loop over intervals (number of snapshots-1)
    for k in range(0,len(pop_SS)-1):
        pop0 = pop_SS[k,:]
        pop1 = pop_SS[k+1,:]
        
    # determine initial combination of allowed transitions
        r = get_reactions(pop0,pop1,S2)

    # generate trajectories for interval and aggregate reactions and r
    # make sure that initial trajectory is valid!!! (ratio > 0)
        while True:
            path_temp = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_temp = get_MH_ratio_expPois(pop0,pop1,path_temp,par_init,r,dT)
            if ratio_temp>0:
                break                
        ratio_old[k] = ratio_temp
        reacts_old.append(path_temp)
        r_old[k] = r

# loop over MCMC iterations        
    output_par = np.zeros((0,3))
    output_traj = []
    acc_prob_ls = []
    reacts = reacts_old
    if plot==True:
        plt.figure(figsize=(12,8))
    for i in range(0,N):
        if i%100==0:
            print(i)
        
# sample rate constants given current sample path (aggregate sample path to one array first)
        reacts_seq = np.zeros((0,2))
        for j in range(0,len(reacts)):
            temp = reacts[j].copy()
            temp[:,1] += j*dT # adjust times since get_path samples times from [0,dT]
            reacts_seq = np.vstack([reacts_seq,temp])
        par = sample_rates(prior,pop_SS[0,:],reacts_seq,T,SMPL_LIM,SMPL_DENS)
        
        # just for plotting the sampled trajectories
        if plot==True:
            traj = get_population_list(reacts_seq,pop_SS[0,:])
            Acell = traj[:,0]
            times = np.hstack([0,reacts_seq[:,1]])
            plt.step(times,Acell,c='b',lw=0.5)
            for l in range(1,int(T/dT)):
                plt.vlines(l*dT,0,np.max(Acell),linestyle='--')

# generate new trajectories for intervals
        reacts = []
        for k in range(0,len(pop_SS)-1):
            pop0 = pop_SS[k,:]
            pop1 = pop_SS[k+1,:]
        
        # propose new trajectory given current rate constants
            r = shuffle_reactions(pop0,pop1,S2,r_old[k],w)

        # generate trajectories for interval
            reacts_new = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_new = get_MH_ratio_expPois(pop0,pop1,reacts_new,par,r,dT)
        
        # accept/reject it with MH-step
            acc_prob = np.min([1,ratio_new/ratio_old[k]])
            if np.random.rand()<acc_prob:
                r_old[k] = r
                ratio_old[k] = ratio_new
                reacts_old[k] = reacts_new   
            reacts.append(reacts_old[k])
            acc_prob_ls.append(acc_prob)
        
        if i>=burn_in:
            output_par = np.vstack([output_par,par])
            output_traj.append(reacts_seq)
    
    print('Finished. Average acceptance prob: ',np.mean(acc_prob_ls))
    if plot==True:
        plt.xlabel('t')
        plt.ylabel(r'$N_{A}$')
        plt.savefig(OUTPUTPATH+'traj_T'+str(T)+'-dT'+str(dT)+'.png', bbox_inches="tight",dpi=300)    
    return [output_par,output_traj,np.mean(acc_prob_ls)]




#%% MCMC function with inhom spacing
        
def get_posterior_inhom(pop_SS,T_arr,prior,N,burn_in,w,plot,OUTPUTPATH,SMPL_LIM,SMPL_DENS):
    # splitting transition matrix into a quadratic, inveratble part S1 and the rest S2, r the same
    # S1 = S[:,0:2] not necessary, since unity matrix here
    S = np.array([[1,0,-1],[0,1,2]])
    S2 = S[:,2]
    
    # guess initial parameter
    par_init = [1,1,1] #r_old_sum/(T*(pop_SS[0,0]+pop_SS[-1,0])/2) # guess 'good' starting point (~ 1/mean A cells)
    print('Start MCMC with par_init: ',par_init)
    
# generate initial trajectory and calculate MH ratio
    reacts_old = []
    ratio_old = np.zeros(len(pop_SS)-1)
    r_old = np.zeros((len(pop_SS)-1,3))
    
    # loop over intervals (number of snapshots-1)
    for k in range(0,len(pop_SS)-1):
        dT = T_arr[k+1]-T_arr[k]
        pop0 = pop_SS[k,:]
        pop1 = pop_SS[k+1,:]
        
    # determine initial combination of allowed transitions
        r = get_reactions(pop0,pop1,S2)

    # generate trajectories for interval and aggregate reactions and r
    # make sure that initial trajectory is valid!!! (ratio > 0)
        while True:
            path_temp = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_temp = get_MH_ratio_expPois(pop0,pop1,path_temp,par_init,r,dT)
            if ratio_temp>0:
                break                
        ratio_old[k] = ratio_temp
        reacts_old.append(path_temp)
        r_old[k] = r

# loop over MCMC iterations        
    output_par = np.zeros((0,3))
    output_traj = []
    acc_prob_ls = []
    reacts = reacts_old
    if plot==True:
        plt.figure(figsize=(12,8))
    for i in range(0,N):
        if i%100==0:
            print(i)
        
# sample rate constants given current sample path (aggregate sample path to one array first)
        reacts_seq = np.zeros((0,2))
        for j in range(0,len(reacts)):
            temp = reacts[j].copy()
            temp[:,1] += T_arr[j] # adjust times since get_path samples times from [0,dT]
            reacts_seq = np.vstack([reacts_seq,temp])
        par = sample_rates(prior,pop_SS[0,:],reacts_seq,T_arr[-1],SMPL_LIM,SMPL_DENS)
        
        # just for plotting the sampled trajectories
        if plot==True:
            traj = get_population_list(reacts_seq,pop_SS[0,:])
            Acell = traj[:,0]
            times = np.hstack([0,reacts_seq[:,1]])
            plt.step(times,Acell,c='b',lw=0.5)
            for l in range(1,len(T_arr)-1):
                plt.vlines(T_arr[l],0,np.max(Acell),linestyle='--')

# generate new trajectories for intervals
        reacts = []
        for k in range(0,len(pop_SS)-1):
            dT = T_arr[k+1]-T_arr[k]
            pop0 = pop_SS[k,:]
            pop1 = pop_SS[k+1,:]
        
        # propose new trajectory given current rate constants
            r = shuffle_reactions(pop0,pop1,S2,r_old[k],w)

        # generate trajectories for interval
            reacts_new = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_new = get_MH_ratio_expPois(pop0,pop1,reacts_new,par,r,dT)
        
        # accept/reject it with MH-step
            acc_prob = np.min([1,ratio_new/ratio_old[k]])
            if np.random.rand()<acc_prob:
                r_old[k] = r
                ratio_old[k] = ratio_new
                reacts_old[k] = reacts_new   
            reacts.append(reacts_old[k])
            acc_prob_ls.append(acc_prob)
        
        if i>=burn_in:
            output_par = np.vstack([output_par,par])
            output_traj.append(reacts_seq)
    
    print('Finished. Average acceptance prob: ',np.mean(acc_prob_ls))
    if plot==True:
        plt.xlabel('t')
        plt.ylabel(r'$N_{A}$')
        plt.savefig(OUTPUTPATH+'traj_T'+str(T_arr)+'.png', bbox_inches="tight",dpi=300)    
    return [output_par,output_traj,np.mean(acc_prob_ls)]



# %% for Model data generation and CD Likelihood

def trajectory_X(par,N_init,T):
    [L_aa,L_ab,L_bb] = par
    [N_a,N_b] = N_init
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        # generate random reaction times
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(3))/np.array([N_a*L_aa, N_a*L_ab, N_a*L_bb])
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        t_min = np.min(times) 
        rtime = np.append(rtime,rtime[-1]+t_min)
        # A -> AA
        if(t_min == times[0]):
           N_a = N_a+1
           N_b = N_b
           rtype = np.append(rtype,0)
        # A -> AB
        elif(t_min == times[1]):
           N_a = N_a
           N_b = N_b+1
           rtype = np.append(rtype,1)
        # A -> BB
        elif(t_min == times[2]):
           N_a = N_a-1
           N_b = N_b+2
           rtype = np.append(rtype,2)   
        if  rtime[-1]>T:
            rtype = rtype[1:-1] # first remove last entries
            rtime = rtime[1:-1]
            break
        
    return np.array([rtype,rtime]).transpose() 


# calculate complete-data likelihood (func a bit different from Bayes_TS.py script)
def lhood_CD(p_data,t_data,T,SMPL_LIM): # c is parameter of interest, transition rate
    lhd = np.zeros((0,SMPL_LIM*100))
    lh_max = np.zeros(3)
    
    integr = 0 # integral: int[0,T] N_A dt, where hazardf = c*N_A
    for j in range(1,len(p_data)):
        integr += p_data[j-1,0]*(p_data[j,2]-p_data[j-1,2]) 
    integr += p_data[-1,0]*(T-p_data[-1,2])
    
    for i in [0,1,2]: # loop over parameter
        r_num = np.sum(t_data[:,0]==i) # number of reactions
        temp = np.zeros(0)
        print('Number of type ',i,' reactions: ',r_num)
        print('Max likelihood for rate ',i,': ',r_num/integr)
        for c in np.arange(0,SMPL_LIM,0.01):
            temp = np.append(temp,(c**r_num)*np.exp(-c*integr))
        lhd = np.append(lhd,[temp],axis=0) 
        lh_max[i] = r_num/integr
        
    return [lhd,lh_max]


# Returns z-scores for MCMC convergence diagnostics. Converged chain: z should oscillate [-1,1]
# Ref: Geweke. Evaluating the accuracy of sampling-based approaches to calculating posterior moments.
def geweke(ls,first,last,nint):
    intervals = np.split(ls,nint)
    z = np.zeros(len(intervals))
    for i in range(0,len(z)):
        seq = intervals[i]
        s_start = seq[:int(len(seq)*first)]
        s_end = seq[int(len(seq)*last):]
        z[i] = (np.mean(s_start)-np.mean(s_end))/np.sqrt(np.var(s_start)+np.var(s_end))
    return z



# %% Testing shit with homogenous spacing

def plot_all(output_par,mA,OUTPUTPATH,SMPL_LIM,pop,par,traj,T,dT):
    # Plot complete data Likelihood
    
    x = np.arange(0,SMPL_LIM,0.01)
    k0 = gaussian_kde(output_par[:,0])
    k1 = gaussian_kde(output_par[:,1])
    k2 = gaussian_kde(output_par[:,2])
    v0_MC = simps((x**2)*k0(x),dx=0.01)-(simps(x*k0(x),dx=0.01))**2
    v1_MC = simps((x**2)*k1(x),dx=0.01)-(simps(x*k1(x),dx=0.01))**2
    v2_MC = simps((x**2)*k2(x),dx=0.01)-(simps(x*k2(x),dx=0.01))**2  
        
    # calculate Likelihood given Complete data
    [lhd_CD,lhd_CD_max] = lhood_CD(pop,traj,T,SMPL_LIM)
        
    lhd_CD[0,:] = lhd_CD[0,:]/simps(lhd_CD[0,:], dx=0.01) # normalise likelihoods 
    lhd_CD[1,:] = lhd_CD[1,:]/simps(lhd_CD[1,:], dx=0.01) # (L_func uses 0.01 as sampling density)
    lhd_CD[2,:] = lhd_CD[2,:]/simps(lhd_CD[2,:], dx=0.01)
    v0_CD = simps((x**2)*lhd_CD[0,:],dx=0.01)-(simps(x*lhd_CD[0,:],dx=0.01))**2
    v1_CD = simps((x**2)*lhd_CD[1,:],dx=0.01)-(simps(x*lhd_CD[1,:],dx=0.01))**2
    v2_CD = simps((x**2)*lhd_CD[2,:],dx=0.01)-(simps(x*lhd_CD[2,:],dx=0.01))**2
    
    fig = plt.figure(figsize=(12,4))
    fig.suptitle(r'Snapshot vs complete data: Parameter estimation, Time: [0,'+str(T)+'], Snapshots: '+str(int(T/dT)), fontsize=12,y=1.05)
        
    ax = plt.subplot(131)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[0]-x[np.argmax(k0(x))])/lhd_CD_max[0],2))+', Var ratio: '+str(round(v0_MC/v0_CD,2)))
    ax.plot(x,lhd_CD[0,:],c='r')
    ax.plot(x,k0(x))
    ax.fill_between(x, k0(x), interpolate=True, color='blue',alpha=0.25)
    plt.vlines(par[0],0,max(lhd_CD[0,:]))
    if np.max(lhd_CD[0,:])>6*np.max(k0(x)):
        ax.set_ylim(0,np.max(k0(x))*3)
    ax.set_xlabel(r'$\lambda_{AA}$')
    
    ax = plt.subplot(132)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[1]-x[np.argmax(k1(x))])/lhd_CD_max[1],2))+', Var ratio: '+str(round(v1_MC/v1_CD,2)))
    ax.plot(x,lhd_CD[1,:],c='r')
    ax.plot(x,k1(x))
    ax.fill_between(x, k1(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[1],0,max(lhd_CD[1,:]))
    if np.max(lhd_CD[1,:])>6*np.max(k1(x)):
        ax.set_ylim(0,np.max(k1(x))*3)
    ax.set_xlabel(r'$\lambda_{AB}$')
    
    ax = plt.subplot(133)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[2]-x[np.argmax(k2(x))])/lhd_CD_max[2],2))+', Var ratio: '+str(round(v2_MC/v2_CD,2)))
    ax.plot(x,lhd_CD[2,:],c='r')
    ax.plot(x,k2(x))
    ax.fill_between(x, k2(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[2],0,max(lhd_CD[2,:]))
    if np.max(lhd_CD[2,:])>6*np.max(k2(x)):
        ax.set_ylim(0,np.max(k2(x))*3)
    ax.set_xlabel(r'$\lambda_{BB}$')
    
    plt.tight_layout()
    plt.savefig(OUTPUTPATH+'CDLH_T'+str(T)+'-dT'+str(dT)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    # Plot MCMC results
    
    # Plot time series
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(r'MCMC Parameter timeseries. Avrg Acceptance prob: '+str(int(1000*mA)/10)+'%', fontsize=12)
    gs = GridSpec(3,4)
    
    ax_joint = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[0,3])
    ax_joint.plot(output_par[:,0],lw=0.3)
    ax_marg_y.hist(output_par[:,0],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AA}$')
    
    ax_joint = fig.add_subplot(gs[1,0:3])
    ax_marg_y = fig.add_subplot(gs[1,3])
    ax_joint.plot(output_par[:,1],lw=0.3)
    ax_marg_y.hist(output_par[:,1],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AB}$')
    
    ax_joint = fig.add_subplot(gs[2,0:3])
    ax_marg_y = fig.add_subplot(gs[2,3])
    ax_joint.plot(output_par[:,2],lw=0.3)
    ax_marg_y.hist(output_par[:,2],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_joint.set_xlabel('Iteration')
    ax_joint.set_ylabel(r'$\lambda_{BB}$')
    plt.savefig(OUTPUTPATH+'TS_T'+str(T)+'-dT'+str(dT)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    
    
    
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
    # Gaussian KDE (kernel density estimate)
    ax1 = sns.jointplot(x=output_par[:,0], y=output_par[:,1], kind='kde')
    ax1.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{AB}$', fontsize=16)
    ax1.ax_joint.plot([par[0]],[par[1]],'ro')
    ax1.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[1]],'rx')
    ax1.ax_marg_x.axvline(par[0], ls='--')
    ax1.ax_marg_y.axhline(par[1], ls='--')
    ax1.savefig(OUTPUTPATH+'MCLH12_T'+str(T)+'-dT'+str(dT)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax2 = sns.jointplot(x=output_par[:,0], y=output_par[:,2], kind='kde')
    ax2.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{BB}$', fontsize=16)
    ax2.ax_joint.plot([par[0]],[par[2]],'ro')
    ax2.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[2]],'rx')
    ax2.ax_marg_x.axvline(par[0], ls='--')
    ax2.ax_marg_y.axhline(par[2], ls='--')
    ax2.savefig(OUTPUTPATH+'MCLH13_T'+str(T)+'-dT'+str(dT)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax3 = sns.jointplot(x=output_par[:,1], y=output_par[:,2], kind='kde')
    ax3.set_axis_labels(r'$\lambda_{AB}$', r'$\lambda_{BB}$', fontsize=16)
    ax3.ax_joint.plot([par[1]],[par[2]],'ro')
    ax3.ax_joint.plot([lhd_CD_max[1]],[lhd_CD_max[2]],'rx')
    ax3.ax_marg_x.axvline(par[1], ls='--')
    ax3.ax_marg_y.axhline(par[2], ls='--')
    ax3.savefig(OUTPUTPATH+'MCLH23_T'+str(T)+'-dT'+str(dT)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    return [round(v0_MC/v0_CD,2),round(v1_MC/v1_CD,2),round(v2_MC/v2_CD,2)]



def plot_all_inhom(output_par,mA,OUTPUTPATH,SMPL_LIM,pop,par,traj,T_arr):
    # Plot complete data Likelihood
    
    x = np.arange(0,SMPL_LIM,0.01)
    k0 = gaussian_kde(output_par[:,0])
    k1 = gaussian_kde(output_par[:,1])
    k2 = gaussian_kde(output_par[:,2])
    v0_MC = simps((x**2)*k0(x),dx=0.01)-(simps(x*k0(x),dx=0.01))**2
    v1_MC = simps((x**2)*k1(x),dx=0.01)-(simps(x*k1(x),dx=0.01))**2
    v2_MC = simps((x**2)*k2(x),dx=0.01)-(simps(x*k2(x),dx=0.01))**2
        
    # calculate Likelihood given Complete data
    [lhd_CD,lhd_CD_max] = lhood_CD(pop,traj,T_arr[-1],SMPL_LIM)
        
    lhd_CD[0,:] = lhd_CD[0,:]/simps(lhd_CD[0,:], dx=0.01) # normalise likelihoods 
    lhd_CD[1,:] = lhd_CD[1,:]/simps(lhd_CD[1,:], dx=0.01) # (L_func uses 0.01 as sampling density)
    lhd_CD[2,:] = lhd_CD[2,:]/simps(lhd_CD[2,:], dx=0.01)
    v0_CD = simps((x**2)*lhd_CD[0,:],dx=0.01)-(simps(x*lhd_CD[0,:],dx=0.01))**2
    v1_CD = simps((x**2)*lhd_CD[1,:],dx=0.01)-(simps(x*lhd_CD[1,:],dx=0.01))**2
    v2_CD = simps((x**2)*lhd_CD[2,:],dx=0.01)-(simps(x*lhd_CD[2,:],dx=0.01))**2
    
    fig = plt.figure(figsize=(12,4))
    fig.suptitle(r'Snapshot vs complete data: Parameter estimation, Snapshots: '+str(T_arr), fontsize=12,y=1.05)
        
    ax = plt.subplot(131)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[0]-x[np.argmax(k0(x))])/lhd_CD_max[0],2))+', Var ratio: '+str(round(v0_MC/v0_CD,2)))
    ax.plot(x,lhd_CD[0,:],c='r')
    ax.plot(x,k0(x))
    ax.fill_between(x, k0(x), interpolate=True, color='blue',alpha=0.25)
    plt.vlines(par[0],0,max(lhd_CD[0,:]))
    if np.max(lhd_CD[0,:])>6*np.max(k0(x)):
        ax.set_ylim(0,np.max(k0(x))*3)
    ax.set_xlabel(r'$\lambda_{AA}$')
    
    ax = plt.subplot(132)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[1]-x[np.argmax(k1(x))])/lhd_CD_max[1],2))+', Var ratio: '+str(round(v1_MC/v1_CD,2)))
    ax.plot(x,lhd_CD[1,:],c='r')
    ax.plot(x,k1(x))
    ax.fill_between(x, k1(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[1],0,max(lhd_CD[1,:]))
    if np.max(lhd_CD[1,:])>6*np.max(k1(x)):
        ax.set_ylim(0,np.max(k1(x))*3)
    ax.set_xlabel(r'$\lambda_{AB}$')
    
    ax = plt.subplot(133)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[2]-x[np.argmax(k2(x))])/lhd_CD_max[2],2))+', Var ratio: '+str(round(v2_MC/v2_CD,2)))
    ax.plot(x,lhd_CD[2,:],c='r')
    ax.plot(x,k2(x))
    ax.fill_between(x, k2(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[2],0,max(lhd_CD[2,:]))
    if np.max(lhd_CD[2,:])>6*np.max(k2(x)):
        ax.set_ylim(0,np.max(k2(x))*3)
    ax.set_xlabel(r'$\lambda_{BB}$')
    
    plt.tight_layout()
    plt.savefig(OUTPUTPATH+'CDLH_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    # Plot MCMC results
    
    # Plot time series
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(r'MCMC Parameter timeseries. Avrg Acceptance prob: '+str(int(1000*mA)/10)+'%', fontsize=12)
    gs = GridSpec(3,4)
    
    ax_joint = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[0,3])
    ax_joint.plot(output_par[:,0],lw=0.3)
    ax_marg_y.hist(output_par[:,0],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AA}$')
    
    ax_joint = fig.add_subplot(gs[1,0:3])
    ax_marg_y = fig.add_subplot(gs[1,3])
    ax_joint.plot(output_par[:,1],lw=0.3)
    ax_marg_y.hist(output_par[:,1],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AB}$')
    
    ax_joint = fig.add_subplot(gs[2,0:3])
    ax_marg_y = fig.add_subplot(gs[2,3])
    ax_joint.plot(output_par[:,2],lw=0.3)
    ax_marg_y.hist(output_par[:,2],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_joint.set_xlabel('Iteration')
    ax_joint.set_ylabel(r'$\lambda_{BB}$')
    plt.savefig(OUTPUTPATH+'TS_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    
    
    
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
    # Gaussian KDE (kernel density estimate)
    ax1 = sns.jointplot(x=output_par[:,0], y=output_par[:,1], kind='kde')
    ax1.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{AB}$', fontsize=16)
    ax1.ax_joint.plot([par[0]],[par[1]],'ro')
    ax1.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[1]],'rx')
    ax1.ax_marg_x.axvline(par[0], ls='--')
    ax1.ax_marg_y.axhline(par[1], ls='--')
    ax1.savefig(OUTPUTPATH+'MCLH12_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax2 = sns.jointplot(x=output_par[:,0], y=output_par[:,2], kind='kde')
    ax2.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{BB}$', fontsize=16)
    ax2.ax_joint.plot([par[0]],[par[2]],'ro')
    ax2.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[2]],'rx')
    ax2.ax_marg_x.axvline(par[0], ls='--')
    ax2.ax_marg_y.axhline(par[2], ls='--')
    ax2.savefig(OUTPUTPATH+'MCLH13_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax3 = sns.jointplot(x=output_par[:,1], y=output_par[:,2], kind='kde')
    ax3.set_axis_labels(r'$\lambda_{AB}$', r'$\lambda_{BB}$', fontsize=16)
    ax3.ax_joint.plot([par[1]],[par[2]],'ro')
    ax3.ax_joint.plot([lhd_CD_max[1]],[lhd_CD_max[2]],'rx')
    ax3.ax_marg_x.axvline(par[1], ls='--')
    ax3.ax_marg_y.axhline(par[2], ls='--')
    ax3.savefig(OUTPUTPATH+'MCLH23_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    return [round(v0_MC/v0_CD,2),round(v1_MC/v1_CD,2),round(v2_MC/v2_CD,2)]

