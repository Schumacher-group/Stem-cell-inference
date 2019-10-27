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

def shuffle_reactions(pop0,pop1,S1,S2,r,w):
    r1 = r[0:2]
    r2_o = r[2:4]
    # pertubate r2 and update r1
    r2 = r2_o + skellam.rvs(w,w,size=2)
    # [[1,1],[0,1]] is inverse of S2=[[0,1],[1,-1]]
    r1 = np.matmul(np.array([[1, 1],[0, 1]]), pop1-pop0-np.matmul(S2,r2)) # not needed, S1 is unity matrix
    r = np.array(np.append(r1,r2),dtype=int)
    if np.any(r<0):
        while True:
            # pertubate r2 and update r1
            r2 = r2_o + skellam.rvs(w,w,size=2)
            r1 = np.matmul(np.array([[1, 1],[0, 1]]), pop1-pop0-np.matmul(S2,r2)) # not needed, S1 is unity matrix
            r = np.array(np.append(r1,r2),dtype=int)
            if np.all(r>=0):
                break
    return r



def get_reactions(pop0,pop1,S1,S2):
    [da,db] = pop1-pop0
    if da>=0 and db>=0:
        temp = np.array([da,0,db,0])
    elif da>0 and db<0: 
        temp = np.array([da+db,0,0,-db])
    elif da<0 and db>0: 
        temp = np.array([0,-da,db+da,0])
    # da<0 and db<0 not possible    
    return shuffle_reactions(pop0,pop1,S1,S2,temp,5)



def get_population_list(reacts,pop0):
    # create trajectories from reaction events
    pop = np.zeros((len(reacts),2),dtype=int)
    temp = pop0
    for i in range(0,len(reacts)):
        if reacts[i,0]==0:
            temp = temp+[1,0]
        elif reacts[i,0]==1:
            temp = temp+[-1,1]
        elif reacts[i,0]==2:
            temp = temp+[0,1]
        elif reacts[i,0]==3:
            temp = temp+[1,-1]
        pop[i,:] = temp
    pop = np.vstack([pop0,pop]) # add pop0 as top row
    return pop



def sample_rates(prior,pop0,reacts,dT,SMPL_LIM,SMPL_DENS): 
    # We assume a flat prior, thats why Posterior = normalised Lhood
    pop = get_population_list(reacts,pop0) # get population trajectory data  
    par_new = []
    
    # loop over L_a, k_ab parameter
    for i in range(0,2): 
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
        post = prior*lhd/simps(prior*lhd, dx=SMPL_DENS) # normalised likelihood 
        temp = choices(np.arange(0,SMPL_LIM,SMPL_DENS), lhd)[0] # draw rate from posterior distr
        par_new.append(temp) 
        
    for i in range(2,4): # loop over L_b, k_ba parameter
        r_num = np.sum(reacts[:,0]==i) # number of reactions
        # if reaction happened in interval
        if len(reacts)>0: 
            integr = pop[0,1]*reacts[0,1] # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
            for j in range(1,len(reacts[:,1])):
                integr += pop[j,1]*(reacts[j,1]-reacts[j-1,1])
            integr += pop[-1,1]*(dT-reacts[-1,1])
        # if nothing happened -> Lhood_P = Lhood_CD
        else:
            integr = pop[0,1]*dT
        
        lhd = np.zeros(0)
        for c in np.arange(0,SMPL_LIM,SMPL_DENS):
            lhd = np.append(lhd,(c**r_num)*np.exp(-c*integr))
        post = prior*lhd/simps(prior*lhd, dx=SMPL_DENS) # normalised likelihood 
        temp = choices(np.arange(0,SMPL_LIM,SMPL_DENS), post)[0] # draw rate from posterior distr 
        par_new.append(temp) 
    
    return par_new



def get_true_lhd(T,reacts,pop0,par_new):
    # get population trajectory data
    pop = get_population_list(reacts,pop0)
    [L_a,k_ab,L_b,k_ba] = par_new

    # calcultate full-data likelihood
    rhaz_T = pop[0,0]*(L_a+k_ab) + pop[0,1]*(L_b+k_ba)
    if reacts[0,0]==0:
        temp = pop[0,0]*L_a*np.exp(-rhaz_T*reacts[0,1])
    elif reacts[0,0]==1:
        temp = pop[0,0]*k_ab*np.exp(-rhaz_T*reacts[0,1])
    elif reacts[0,0]==2:
        temp = pop[0,1]*L_b*np.exp(-rhaz_T*reacts[0,1])
    elif reacts[0,0]==3:
        temp = pop[0,1]*k_ba*np.exp(-rhaz_T*reacts[0,1])
    
    v = reacts[:,0]        
    for i in range(1,len(reacts)):
        t_i = reacts[i,1]
        t_i1 = reacts[i-1,1]
        rhaz_T = pop[i,0]*(L_a+k_ab) + pop[i,1]*(L_b+k_ba)
        if v[i]==0:
            temp = temp*pop[i,0]*L_a*np.exp(-rhaz_T*(t_i-t_i1))
        elif v[i]==1:
            temp = temp*pop[i,0]*k_ab*np.exp(-rhaz_T*(t_i-t_i1))
        elif v[i]==2:
            temp = temp*pop[i,1]*L_b*np.exp(-rhaz_T*(t_i-t_i1))  
        elif v[i]==3:
            temp = temp*pop[i,1]*k_ba*np.exp(-rhaz_T*(t_i-t_i1))    
            
    rhaz_T = pop[-1,0]*(L_a+k_ab) + pop[-1,1]*(L_b+k_ba)                  
    Lhood_T = temp*np.exp(-rhaz_T*(T-reacts[-1,1]))
    return Lhood_T



#%% lin Poisson Model

# simulate homogenous Poisson process and then transform time to get lin. inhom. Poisson process
def get_reaction_times_linPois(pop0,pop1,r,dT):
    # calculate reaction hazards for all reactions at T0 and T1
    # set hazards just to population size, since prefactor (rate) canceles in expression
    rhaz0 = [ pop0[0], pop0[0], pop0[1], pop0[1] ]
    rhaz1 = [ pop1[0], pop1[0], pop1[1], pop1[1] ]
    rtype = np.zeros(0,dtype=int)
    rtime_inhom = np.zeros(0)

    for i in range(0,4):
        rtype = np.append(rtype,i*np.ones(r[i],dtype=int))
        # reaction times are uniform distributed
        rtime_hom = np.random.uniform(0,dT,r[i])
        h0 = rhaz0[i]
        h1 = rhaz1[i]
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
    for i in range(0,4):
        mu = T*(rhaz0[i]+rhaz1[i])/2
        q = q*poisson.pmf(r[i], mu)
    return q



def get_MH_ratio_linPois(pop0,pop1,reacts,par_new,r,T):
    [L_a,k_ab,L_b,k_ba] = par_new
    # calculate reaction hazards for all reactions at T0 and T1
    rhaz0 = [ L_a*pop0[0], k_ab*pop0[0], L_b*pop0[1], k_ba*pop0[1] ]
    rhaz1 = [ L_a*pop1[0], k_ab*pop1[0], L_b*pop1[1], k_ba*pop1[1] ]

    # if reaction happened in interval
    if len(reacts)>0: 
        q = get_lin_Poisson_PMF(rhaz0,rhaz1,r,T)
        Lhood_P = get_lin_Poisson_lhd(rhaz0,rhaz1,T,reacts)
        Lhood_T = get_true_lhd(T,reacts,pop0,par_new)
    # if nothing happened -> Lhood_P = Lhood_CD
    else:
        q = get_lin_Poisson_PMF(rhaz0,rhaz1,r,T)
        Lhood_P = np.exp(-T*(np.sum(rhaz0)+np.sum(rhaz1))/2)   
        Lhood_T = np.exp(-pop1[0]*(L_a+k_ab)*T-pop1[1]*(L_b+k_ba)*T)
        if np.abs(1-Lhood_P/Lhood_T)>0.01: # just to make sure...
            print('Error!')
    
    if q==0 and Lhood_P==0:
        # if reaction hazard is 0 (rate>0, pop=0), but process still observed: return 0
        # specific to model Y, since e.g. [1,0] -> [0,1] -> [1,0] possible -> q=LHP=0, LHCD>0
        return 0    
    else:
        return q*Lhood_T/Lhood_P


#%% exp Poisson Model

# simulate homogenous Poisson process and then transform time to get exp inhom. Poisson process
def get_reaction_times_expPois(pop0,pop1,r,T):
    # calculate reaction hazards for all reactions at T0 and T1
    rtype = np.zeros(0,dtype=int)
    rtime_inhom = np.zeros(0)
    N0_arr = [pop0[0],pop0[0],pop0[1],pop0[1]]
    N1_arr = [pop1[0],pop1[0],pop1[1],pop1[1]]

    for i in range(0,4):
        rtype = np.append(rtype,i*np.ones(r[i],dtype=int))
        unif_sample = np.random.uniform(size=r[i])
        N0 = N0_arr[i]
        N1 = N1_arr[i]
        
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
    N0_arr = [pop0[0],pop0[0],pop0[1],pop0[1]]
    N1_arr = [pop1[0],pop1[0],pop1[1],pop1[1]] 
    
    # iterate over all reaction events
    react_type = int(reacts[0,0])    
    N0 = N0_arr[react_type]
    N1 = N1_arr[react_type]
    temp = par[react_type]*N0*(N1/N0)**(reacts[0,1]/T)
    
    v = reacts[:,0]        
    for i in range(1,len(reacts)):
        t_i = reacts[i,1]
        react_type = int(v[i])
        N0 = N0_arr[react_type]
        N1 = N1_arr[react_type]
        temp = temp*par[react_type]*N0*(N1/N0)**(t_i/T)
    
    dN_a = pop1[0]-pop0[0]
    dN_b = pop1[1]-pop0[1]
    dlogN_a = np.log(pop1[0])-np.log(pop0[0])
    dlogN_b = np.log(pop1[1])-np.log(pop0[1])    
                 
    if pop1[0]!=pop0[0] and pop1[1]!=pop0[1]: # both population changed in interval
        Lhood_P = temp*np.exp(-T*(par[0]+par[1])*dN_a/dlogN_a - T*(par[2]+par[3])*dN_b/dlogN_b)
    elif pop1[0]!=pop0[0] and pop1[1]==pop0[1]: # only A population changed in interval
        Lhood_P = temp*np.exp(-T*(par[0]+par[1])*dN_a/dlogN_a - T*(par[2]+par[3])*pop0[1])
    elif pop1[0]==pop0[0] and pop1[1]!=pop0[1]: # only B population changed in interval
        Lhood_P = temp*np.exp(-T*(par[0]+par[1])*pop0[0] - T*(par[2]+par[3])*dN_b/dlogN_b)
    else: # no changes in interval
        Lhood_P = temp*np.exp(-T*(par[0]+par[1])*pop0[0] - T*(par[2]+par[3])*pop0[1])
    return Lhood_P



def get_exp_Poisson_PMF(pop0,pop1,par,r,T):
    # calculate PMF of Poisson process given observed reactions
    N0_arr = [pop0[0],pop0[0],pop0[1],pop0[1]]
    N1_arr = [pop1[0],pop1[0],pop1[1],pop1[1]] 
    
    q = 1
    for i in range(0,4):
        N0 = N0_arr[i]
        N1 = N1_arr[i]
        if N0!=N1:
            dN = N1-N0
            dlogN = np.log(N1)-np.log(N0)
            mu = T*par[i]*dN/dlogN
        else:
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
    S = np.array([[1,-1,0,1],[0,1,1,-1]])
    S1 = S[:,0:2]
    S2 = S[:,2:4]
    
    # guess initial parameter
    par_init = [1,1,1,1] # guess 'good' starting point
    print('Start MCMC with par_init: ',par_init)
    
# generate initial trajectory and calculate MH ratio
    reacts_old = []
    ratio_old = np.zeros(len(pop_SS)-1)
    r_old = np.zeros((len(pop_SS)-1,4))
    
    # loop over intervals (number of snapshots-1)
    for k in range(0,len(pop_SS)-1):
        pop0 = pop_SS[k,:]
        pop1 = pop_SS[k+1,:]
        
    # determine initial combination of allowed transitions
        r = get_reactions(pop0,pop1,S1,S2)

    # generate trajectories for interval and aggregate reactions and r
        while True:
            # make sure that initial trajectory is valid!!! (ratio > 0)
            r = shuffle_reactions(pop0,pop1,S1,S2,r,w)
            path_temp = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_temp = get_MH_ratio_expPois(pop0,pop1,path_temp,par_init,r,dT)
            if ratio_temp>0:
                break                
        ratio_old[k] = ratio_temp
        reacts_old.append(path_temp)
        r_old[k] = r

# loop over MCMC iterations        
    output_par = np.zeros((0,4))
    output_traj = []
    acc_prob_ls = []
    reacts = reacts_old
    if plot==True:
        plt.figure(figsize=(20,8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
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
            A_cell = traj[:,0]
            B_cell = traj[:,1]
            times = np.hstack([0,reacts_seq[:,1]])
            ax1.step(times,A_cell,c='b',lw=0.5)
            ax2.step(times,B_cell,c='r',lw=0.5)

# generate new trajectories for intervals
        reacts = []
        for k in range(0,len(pop_SS)-1):
            pop0 = pop_SS[k,:]
            pop1 = pop_SS[k+1,:]
        
        # propose new trajectory given current rate constants
            r = shuffle_reactions(pop0,pop1,S1,S2,r_old[k],w)

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
        # draw vertical lines at snapshot positions
        for l in range(1,int(T/dT)):
                ax1.vlines(l*dT,0,np.max(A_cell),linestyle='--')
                ax2.vlines(l*dT,0,np.max(B_cell),linestyle='--')
        ax1.set_xlabel('t')
        ax1.set_ylabel(r'$N_{A}$')
        ax2.set_xlabel('t')
        ax2.set_ylabel(r'$N_{B}$')
        plt.savefig(OUTPUTPATH+'traj_T'+str(T)+'-dT'+str(dT)+'.png', bbox_inches="tight",dpi=300)    
    return [output_par,output_traj,np.mean(acc_prob_ls)]



#%% MCMC function with inhom spacing
        
def get_posterior_inhom(pop_SS,T_arr,prior,N,burn_in,w,plot,OUTPUTPATH,SMPL_LIM,SMPL_DENS):
    # splitting transition matrix into a quadratic, inveratble part S1 and the rest S2, r the same
    S = np.array([[1,-1,0,1],[0,1,1,-1]])
    S1 = S[:,0:2]
    S2 = S[:,2:4]
    
    # guess initial parameter
    par_init = [1,1,1,1] #r_old_sum/(T*(pop_SS[0,0]+pop_SS[-1,0])/2) # guess 'good' starting point (~ 1/mean A cells)
    print('Start MCMC with par_init: ',par_init)
    
# generate initial trajectory and calculate MH ratio
    reacts_old = []
    ratio_old = np.zeros(len(pop_SS)-1)
    r_old = np.zeros((len(pop_SS)-1,4))
    
    # loop over intervals (number of snapshots-1)
    for k in range(0,len(pop_SS)-1):
        dT = T_arr[k+1]-T_arr[k]
        pop0 = pop_SS[k,:]
        pop1 = pop_SS[k+1,:]
        
    # determine initial combination of allowed transitions
        r = get_reactions(pop0,pop1,S1,S2)

    # generate trajectories for interval and aggregate reactions and r
    # make sure that initial trajectory is valid!!! (ratio > 0)
        while True:
            r = shuffle_reactions(pop0,pop1,S1,S2,r,w)
            path_temp = get_reaction_times_expPois(pop0,pop1,r,dT)
            ratio_temp = get_MH_ratio_expPois(pop0,pop1,path_temp,par_init,r,dT)
            if ratio_temp>0:
                break                
        ratio_old[k] = ratio_temp
        reacts_old.append(path_temp)
        r_old[k] = r

# loop over MCMC iterations        
    output_par = np.zeros((0,4))
    output_traj = []
    acc_prob_ls = []
    reacts = reacts_old
    if plot==True:
        plt.figure(figsize=(20,8))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
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
            A_cell = traj[:,0]
            B_cell = traj[:,1]
            times = np.hstack([0,reacts_seq[:,1]])
            ax1.step(times,A_cell,c='b',lw=0.5)
            ax2.step(times,B_cell,c='r',lw=0.5)

# generate new trajectories for intervals
        reacts = []
        for k in range(0,len(pop_SS)-1):
            dT = T_arr[k+1]-T_arr[k]
            pop0 = pop_SS[k,:]
            pop1 = pop_SS[k+1,:]
        
        # propose new trajectory given current rate constants
            r = shuffle_reactions(pop0,pop1,S1,S2,r_old[k],w)

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
        # draw vertical lines at snapshot positions
        for l in range(1,len(T_arr)-1):
                ax1.vlines(T_arr[l],0,np.max(A_cell),linestyle='--')
                ax2.vlines(T_arr[l],0,np.max(B_cell),linestyle='--')
        ax1.set_xlabel('t')
        ax1.set_ylabel(r'$N_{A}$')
        ax2.set_xlabel('t')
        ax2.set_ylabel(r'$N_{B}$')
        plt.savefig(OUTPUTPATH+'traj_T'+str(T_arr)+'.png', bbox_inches="tight",dpi=300)    
    return [output_par,output_traj,np.mean(acc_prob_ls)]



# %% for Model data generation and CD Likelihood

def trajectory_Y(par,N_init,T):
    [L_a,k_ab,L_b,k_ba] = par
    [N_a,N_b] = N_init
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True: 
        # generate random reaction times
        with np.errstate(divide='ignore', invalid='ignore'): # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(4))/np.array([N_a*L_a, N_b*L_b, N_a*k_ab, N_b*k_ba ])
            times[np.logical_or(np.logical_or(times==np.inf, times==0), np.logical_or(times==-np.inf, np.isnan(times)))] = T+1
        t_min = np.min(times) 
        rtime = np.append(rtime,rtime[-1]+t_min)
        
        # A -> AA
        if(t_min == times[0]):
            N_a = N_a+1
            N_b = N_b 
            rtype = np.append(rtype,0)
        # A -> B
        elif(t_min == times[2]):
            N_a = N_a-1
            N_b = N_b+1 
            rtype = np.append(rtype,1)
        # B -> BB
        elif(t_min == times[1]):
            N_a = N_a
            N_b = N_b+1 
            rtype = np.append(rtype,2)
        # B -> A
        elif(t_min == times[3]):
            N_a = N_a+1
            N_b = N_b-1
            rtype = np.append(rtype,3)

        if  rtime[-1]>T:
            rtype = rtype[1:-2] # remove last entries
            rtime = rtime[1:-2]
            break
    return np.array([rtype,rtime]).transpose() 


# calculate complete-data likelihood (func a bit different from Bayes_TS.py script)
def lhood_CD(p_data,t_data,T,SMPL_LIM): # c is parameter of interest, transition rate
    lhd = np.zeros((0,SMPL_LIM*100))
    lh_max = np.zeros(4)
    
    integrX = 0 # integral: int[0,T] N_A dt, where hazardf = c*N_A for par[0,1]
    for j in range(1,len(p_data)):
        integrX += p_data[j-1,0]*(p_data[j,2]-p_data[j-1,2]) 
    integrX += p_data[-1,0]*(T-p_data[-1,2])
    
    integrY = 0 # integral: int[0,T] N_B dt, where hazardf = c*N_B for par[2,3]
    for j in range(1,len(p_data)):
        integrY += p_data[j-1,1]*(p_data[j,2]-p_data[j-1,2]) 
    integrY += p_data[-1,1]*(T-p_data[-1,2])
        
    for i in [0,1]: # loop over pop_A dependent parameter
        r_num = np.sum(t_data[:,0]==i) # number of reactions
        temp = np.zeros(0)
        print('Number of type ',i,' reactions: ',r_num)
        print('Max likelihood for rate ',i,': ',r_num/integrX)
        for c in np.arange(0,SMPL_LIM,0.01):
            temp = np.append(temp,(c**r_num)*np.exp(-c*integrX))
        lhd = np.append(lhd,[temp],axis=0) 
        lh_max[i] = r_num/integrX
        
    for i in [2,3]: # loop over pop_B dependent parameter
        r_num = np.sum(t_data[:,0]==i) # number of reactions
        temp = np.zeros(0)
        print('Number of type ',i,' reactions: ',r_num)
        print('Max likelihood for rate ',i,': ',r_num/integrY)
        for c in np.arange(0,SMPL_LIM,0.01):
            temp = np.append(temp,(c**r_num)*np.exp(-c*integrY))
        lhd = np.append(lhd,[temp],axis=0)
        lh_max[i] = r_num/integrY
        
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
    k3 = gaussian_kde(output_par[:,3])
    v0_MC = simps((x**2)*k0(x),dx=0.01)-(simps(x*k0(x),dx=0.01))**2
    v1_MC = simps((x**2)*k1(x),dx=0.01)-(simps(x*k1(x),dx=0.01))**2
    v2_MC = simps((x**2)*k2(x),dx=0.01)-(simps(x*k2(x),dx=0.01))**2
    v3_MC = simps((x**2)*k3(x),dx=0.01)-(simps(x*k3(x),dx=0.01))**2
        
    # calculate Likelihood given Complete data
    [lhd_CD,lhd_CD_max] = lhood_CD(pop,traj,T,SMPL_LIM)
        
    lhd_CD[0,:] = lhd_CD[0,:]/simps(lhd_CD[0,:], dx=0.01) # normalise likelihoods 
    lhd_CD[1,:] = lhd_CD[1,:]/simps(lhd_CD[1,:], dx=0.01) # (L_func uses 0.01 as sampling density)
    lhd_CD[2,:] = lhd_CD[2,:]/simps(lhd_CD[2,:], dx=0.01)
    lhd_CD[3,:] = lhd_CD[3,:]/simps(lhd_CD[3,:], dx=0.01)
    v0_CD = simps((x**2)*lhd_CD[0,:],dx=0.01)-(simps(x*lhd_CD[0,:],dx=0.01))**2
    v1_CD = simps((x**2)*lhd_CD[1,:],dx=0.01)-(simps(x*lhd_CD[1,:],dx=0.01))**2
    v2_CD = simps((x**2)*lhd_CD[2,:],dx=0.01)-(simps(x*lhd_CD[2,:],dx=0.01))**2
    v3_CD = simps((x**2)*lhd_CD[3,:],dx=0.01)-(simps(x*lhd_CD[3,:],dx=0.01))**2
        
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(r'Snapshot vs complete data: Parameter estimation, Time: [0,'+str(T)+'], Snapshots: '+str(int(T/dT)), fontsize=12,y=1.05)
        
    ax = plt.subplot(221)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[0]-x[np.argmax(k0(x))])/lhd_CD_max[0],2))+', Var ratio: '+str(round(v0_MC/v0_CD,2)))
    ax.plot(x,lhd_CD[0,:],c='r')
    ax.plot(x,k0(x))
    ax.fill_between(x, k0(x), interpolate=True, color='blue',alpha=0.25)
    plt.vlines(par[0],0,max(lhd_CD[0,:]))
    if np.max(lhd_CD[0,:])>6*np.max(k0(x)):
        ax.set_ylim(0,np.max(k0(x))*3)
    ax.set_xlabel(r'$\lambda_{A}$')
    
    ax = plt.subplot(222)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[1]-x[np.argmax(k1(x))])/lhd_CD_max[1],2))+', Var ratio: '+str(round(v1_MC/v1_CD,2)))
    ax.plot(x,lhd_CD[1,:],c='r')
    ax.plot(x,k1(x))
    ax.fill_between(x, k1(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[1],0,max(lhd_CD[1,:]))
    if np.max(lhd_CD[1,:])>6*np.max(k1(x)):
        ax.set_ylim(0,np.max(k1(x))*3)
    ax.set_xlabel(r'$k_{AB}$')
    
    ax = plt.subplot(223)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[2]-x[np.argmax(k2(x))])/lhd_CD_max[2],2))+', Var ratio: '+str(round(v2_MC/v2_CD,2)))
    ax.plot(x,lhd_CD[2,:],c='r')
    ax.plot(x,k2(x))
    ax.fill_between(x, k2(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[2],0,max(lhd_CD[2,:]))
    if np.max(lhd_CD[2,:])>6*np.max(k2(x)):
        ax.set_ylim(0,np.max(k2(x))*3)
    ax.set_xlabel(r'$\lambda_{B}$')
    
    ax = plt.subplot(224)
    ax.set_title('Rel Bias: '+str(round(np.abs(lhd_CD_max[3]-x[np.argmax(k3(x))])/lhd_CD_max[3],2))+', Var ratio: '+str(round(v3_MC/v3_CD,2)))
    ax.plot(x,lhd_CD[3,:],c='r')
    ax.plot(x,k3(x))
    ax.fill_between(x, k3(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[3],0,max(lhd_CD[2,:]))
    if np.max(lhd_CD[3,:])>6*np.max(k3(x)):
        ax.set_ylim(0,np.max(k3(x))*3)
    ax.set_xlabel(r'$k_{BA}$')
    
    plt.tight_layout()
    plt.savefig(OUTPUTPATH+'CDLH_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)
    
    # Plot MCMC results
    
    # Plot time series
    fig = plt.figure(figsize=(10,14))
    fig.suptitle(r'MCMC Parameter timeseries. Avrg Acceptance prob: '+str(int(1000*mA)/10)+'%', fontsize=12)
    gs = GridSpec(4,4)
    
    ax_joint = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[0,3])
    ax_joint.plot(output_par[:,0],lw=0.3)
    ax_marg_y.hist(output_par[:,0],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{A}$')
    
    ax_joint = fig.add_subplot(gs[1,0:3])
    ax_marg_y = fig.add_subplot(gs[1,3])
    ax_joint.plot(output_par[:,1],lw=0.3)
    ax_marg_y.hist(output_par[:,1],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$k_{AB}$')
    
    ax_joint = fig.add_subplot(gs[2,0:3])
    ax_marg_y = fig.add_subplot(gs[2,3])
    ax_joint.plot(output_par[:,2],lw=0.3)
    ax_marg_y.hist(output_par[:,2],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{B}$')
    
    ax_joint = fig.add_subplot(gs[3,0:3])
    ax_marg_y = fig.add_subplot(gs[3,3])
    ax_joint.plot(output_par[:,3],lw=0.3)
    ax_marg_y.hist(output_par[:,3],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_joint.set_xlabel('Iteration')
    ax_joint.set_ylabel(r'$k_{BA}$')
    plt.savefig(OUTPUTPATH+'TS_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)
    
    
    
    
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
    # Gaussian KDE (kernel density estimate)
    ax1 = sns.jointplot(x=output_par[:,0], y=output_par[:,1], kind='kde')
    ax1.set_axis_labels(r'$\lambda_{A}$', r'$\lambda_{B}$', fontsize=16)
    ax1.ax_joint.plot([par[0]],[par[2]],'ro')
    ax1.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[2]],'rx')
    ax1.ax_marg_x.axvline(par[0], ls='--')
    ax1.ax_marg_y.axhline(par[2], ls='--')
    ax1.savefig(OUTPUTPATH+'MCLH13_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)
    
    ax2 = sns.jointplot(x=output_par[:,0], y=output_par[:,2], kind='kde')
    ax2.set_axis_labels(r'$k_{AB}$', r'$k_{BA}$', fontsize=16)
    ax2.ax_joint.plot([par[1]],[par[3]],'ro')
    ax2.ax_joint.plot([lhd_CD_max[1]],[lhd_CD_max[3]],'rx')
    ax2.ax_marg_x.axvline(par[1], ls='--')
    ax2.ax_marg_y.axhline(par[3], ls='--')
    ax2.savefig(OUTPUTPATH+'MCLH24_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)      
    
    ax3 = sns.jointplot(x=output_par[:,0], y=output_par[:,3], kind='kde')
    ax3.set_axis_labels(r'$\lambda_{A}$', r'$k_{BA}$', fontsize=16)
    ax3.ax_joint.plot([par[0]],[par[3]],'ro')
    ax3.ax_joint.plot([lhd_CD_max[0]],[lhd_CD_max[3]],'rx')
    ax3.ax_marg_x.axvline(par[0], ls='--')
    ax3.ax_marg_y.axhline(par[3], ls='--')
    ax3.savefig(OUTPUTPATH+'MCLH14_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)      

    ax4 = sns.jointplot(x=output_par[:,1], y=output_par[:,2], kind='kde')
    ax4.set_axis_labels(r'$\lambda_{B}$', r'$k_{AB}$', fontsize=16)
    ax4.ax_joint.plot([par[2]],[par[1]],'ro')
    ax4.ax_joint.plot([lhd_CD_max[2]],[lhd_CD_max[1]],'rx')
    ax4.ax_marg_x.axvline(par[2], ls='--')
    ax4.ax_marg_y.axhline(par[1], ls='--')
    ax4.savefig(OUTPUTPATH+'MCLH23_T'+str(T)+'-dT'+str(dT)+'_a'+str(par[0])+'_ab'+str(par[1])+'_b'+str(par[2])+'_ba'+str(par[3])+'.png', bbox_inches="tight",dpi=300)      

    return [round(v0_MC/v0_CD,2),round(v1_MC/v1_CD,2),round(v2_MC/v2_CD,2)]


# not done yet
'''              
    # calculate Likelihood given Snapshot data
    [MCMC_output,mA] = get_posterior_inhom(pop_SS,T_arr,S,steps,burn_in,w,False,OUTPUTPATH,SMPL_LIM,SMPL_DENS)
        
    x = np.arange(0,SMPL_LIM,0.01)
    k0 = gaussian_kde(MCMC_output[:,0])
    k1 = gaussian_kde(MCMC_output[:,1])
    k2 = gaussian_kde(MCMC_output[:,2])
    k3 = gaussian_kde(MCMC_output[:,3])
    var0_MC = simps((x**2)*k0(x),dx=0.01)-(simps(x*k0(x),dx=0.01))**2
    var1_MC = simps((x**2)*k1(x),dx=0.01)-(simps(x*k1(x),dx=0.01))**2
    var2_MC = simps((x**2)*k2(x),dx=0.01)-(simps(x*k2(x),dx=0.01))**2
    var3_MC = simps((x**2)*k3(x),dx=0.01)-(simps(x*k3(x),dx=0.01))**2
        
        
    # calculate Likelihood given Complete data
    [lhd_CD,lhd_CD_max] = lhood_CD(pop,traj,T_arr[-1],SMPL_LIM)
        
    lhd_CD[0,:] = lhd_CD[0,:]/simps(lhd_CD[0,:], dx=0.01) # normalise likelihoods 
    lhd_CD[1,:] = lhd_CD[1,:]/simps(lhd_CD[1,:], dx=0.01) # (L_func uses 0.01 as sampling density)
    lhd_CD[2,:] = lhd_CD[2,:]/simps(lhd_CD[2,:], dx=0.01)
    lhd_CD[3,:] = lhd_CD[3,:]/simps(lhd_CD[3,:], dx=0.01)
    var0_CD = simps((x**2)*lhd_CD[0,:],dx=0.01)-(simps(x*lhd_CD[0,:],dx=0.01))**2
    var1_CD = simps((x**2)*lhd_CD[1,:],dx=0.01)-(simps(x*lhd_CD[1,:],dx=0.01))**2
    var2_CD = simps((x**2)*lhd_CD[2,:],dx=0.01)-(simps(x*lhd_CD[2,:],dx=0.01))**2
    var3_CD = simps((x**2)*lhd_CD[3,:],dx=0.01)-(simps(x*lhd_CD[3,:],dx=0.01))**2



def plot_all_inhom(ls,OUTPUTPATH,par,T_arr):
    # Plot complete data Likelihood
        
    [par_ls,x,k0,k1,k2,v0_MC,v1_MC,v2_MC,lhd,lh_max,v0_CD,v1_CD,v2_CD,mA] = ls
    
    fig = plt.figure(figsize=(12,4))
    fig.suptitle(r'Snapshot vs complete data: Parameter estimation, Snapshots: '+str(T_arr), fontsize=12,y=1.05)
        
    ax = plt.subplot(131)
    ax.set_title('Rel Bias: '+str(round(np.abs(lh_max[0]-x[np.argmax(k0(x))])/lh_max[0],2))+', Var ratio: '+str(round(v0_MC/v0_CD,2)))
    ax.plot(x,lhd[0,:],c='r')
    ax.plot(x,k0(x))
    ax.fill_between(x, k0(x), interpolate=True, color='blue',alpha=0.25)
    plt.vlines(par[0],0,max(lhd[0,:]))
    if np.max(lhd[0,:])>6*np.max(k0(x)):
        ax.set_ylim(0,np.max(k0(x))*3)
    ax.set_xlabel(r'$\lambda_{AA}$')
    
    ax = plt.subplot(132)
    ax.set_title('Rel Bias: '+str(round(np.abs(lh_max[1]-x[np.argmax(k1(x))])/lh_max[1],2))+', Var ratio: '+str(round(v1_MC/v1_CD,2)))
    ax.plot(x,lhd[1,:],c='r')
    ax.plot(x,k1(x))
    ax.fill_between(x, k1(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[1],0,max(lhd[1,:]))
    if np.max(lhd[1,:])>6*np.max(k1(x)):
        ax.set_ylim(0,np.max(k1(x))*3)
    ax.set_xlabel(r'$\lambda_{AB}$')
    
    ax = plt.subplot(133)
    ax.set_title('Rel Bias: '+str(round(np.abs(lh_max[2]-x[np.argmax(k2(x))])/lh_max[2],2))+', Var ratio: '+str(round(v2_MC/v2_CD,2)))
    ax.plot(x,lhd[2,:],c='r')
    ax.plot(x,k2(x))
    ax.fill_between(x, k2(x), interpolate=True, color='blue',alpha=0.25)
    ax.vlines(par[2],0,max(lhd[2,:]))
    if np.max(lhd[2,:])>6*np.max(k2(x)):
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
    ax_joint.plot(par_ls[:,0],lw=0.3)
    ax_marg_y.hist(par_ls[:,0],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AA}$')
    
    ax_joint = fig.add_subplot(gs[1,0:3])
    ax_marg_y = fig.add_subplot(gs[1,3])
    ax_joint.plot(par_ls[:,1],lw=0.3)
    ax_marg_y.hist(par_ls[:,1],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    ax_joint.set_ylabel(r'$\lambda_{AB}$')
    
    ax_joint = fig.add_subplot(gs[2,0:3])
    ax_marg_y = fig.add_subplot(gs[2,3])
    ax_joint.plot(par_ls[:,2],lw=0.3)
    ax_marg_y.hist(par_ls[:,2],orientation="horizontal",bins=30)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_joint.set_xlabel('Iteration')
    ax_joint.set_ylabel(r'$\lambda_{BB}$')
    plt.savefig(OUTPUTPATH+'TS_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    
    
    
    # Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
    # Gaussian KDE (kernel density estimate)
    ax1 = sns.jointplot(x=par_ls[:,0], y=par_ls[:,1], kind='kde')
    ax1.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{AB}$', fontsize=16)
    ax1.ax_joint.plot([par[0]],[par[1]],'ro')
    ax1.ax_joint.plot([lh_max[0]],[lh_max[1]],'rx')
    ax1.ax_marg_x.axvline(par[0], ls='--')
    ax1.ax_marg_y.axhline(par[1], ls='--')
    ax1.savefig(OUTPUTPATH+'MCLH12_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax2 = sns.jointplot(x=par_ls[:,0], y=par_ls[:,2], kind='kde')
    ax2.set_axis_labels(r'$\lambda_{AA}$', r'$\lambda_{BB}$', fontsize=16)
    ax2.ax_joint.plot([par[0]],[par[2]],'ro')
    ax2.ax_joint.plot([lh_max[0]],[lh_max[2]],'rx')
    ax2.ax_marg_x.axvline(par[0], ls='--')
    ax2.ax_marg_y.axhline(par[2], ls='--')
    ax2.savefig(OUTPUTPATH+'MCLH13_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    ax3 = sns.jointplot(x=par_ls[:,1], y=par_ls[:,2], kind='kde')
    ax3.set_axis_labels(r'$\lambda_{AB}$', r'$\lambda_{BB}$', fontsize=16)
    ax3.ax_joint.plot([par[1]],[par[2]],'ro')
    ax3.ax_joint.plot([lh_max[1]],[lh_max[2]],'rx')
    ax3.ax_marg_x.axvline(par[1], ls='--')
    ax3.ax_marg_y.axhline(par[2], ls='--')
    ax3.savefig(OUTPUTPATH+'MCLH23_T'+str(T_arr)+'_aa'+str(par[0])+'_ab'+str(par[1])+'_bb'+str(par[2])+'.png', bbox_inches="tight",dpi=300)
    
    return [round(v0_MC/v0_CD,2),round(v1_MC/v1_CD,2),round(v2_MC/v2_CD,2)]
'''
