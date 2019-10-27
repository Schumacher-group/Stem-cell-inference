import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import simps

################## inference functions ##################

# create array with reaction times and types
def trans_dat(sample_A,sample_B,model):
    pos_react = np.where( np.any([np.diff(sample_A)!=0, np.diff(sample_B)!=0],axis=0) )[0]
    r_time = pos_react+1
    r_type = np.zeros(len(r_time), dtype = np.uint64)

    for i in range(0,len(r_time)):
        k = pos_react[i]
        d_AB = [np.diff(sample_A)[k], np.diff(sample_B)[k]] # population changes at reaction
    
        if(model=='X'):
            if d_AB==[1,0]: # A -> AA
                r_type[i]=0
            elif d_AB==[0,1]: # A -> AB
                r_type[i]=1
            elif d_AB==[-1,2]: # A -> BB
                r_type[i]=2
            else: # unknown transition
                r_type[i]=-1
                print('unknown reaction at pos ',k)
            
        if(model=='Y'):
            if d_AB==[1,0]: # A -> AA
                r_type[i]=0
            elif d_AB==[-1,1]: # A -> B
                r_type[i]=1
            elif d_AB==[0,1]: # B -> BB
                r_type[i]=2
            elif d_AB==[1,-1]: # B -> A
                r_type[i]=3
            else: # unknown transition
                r_type[i]=-1
                print('unknown reaction at pos ',k)

    t_data = np.array([r_time,r_type]).transpose()
    p_data = np.array([sample_A,sample_B]).transpose()
   
    return (t_data,p_data)


# calculate complete-data likelihood (for mass-action kinetic rate laws, otherwise does not factorise)
def L_func(p_data,t_data,model,dt): # c is parameter of interest, transition rate
    
    if(model=='X'):
        lhd = np.zeros((0,1000))
        
        for i in [0,1,2]: # loop over parameter
            r_num = np.sum(t_data[:,1]==i) # number of reactions
            integr = np.sum(dt*p_data[:,0]) # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
                
            temp = np.zeros(0)
            print('Number of type ',i,' reactions: ',r_num)
            print('Max likelihood for rate ',i,': ',r_num/integr)
            for c in np.arange(0,10,0.01):
                temp = np.append(temp,(c**r_num)*np.exp(-c*integr))
            lhd = np.append(lhd,[temp],axis=0) # normalised lhd
            
    if(model=='Y'):
        lhd = np.zeros((0,1000))
        
        for i in [0,1]: # loop over pop_A dependent parameter
            r_num = np.sum(t_data[:,1]==i) # number of reactions
            integr = np.sum(dt*p_data[:,0]) # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
            
            temp = np.zeros(0)
            print('Number of type ',i,' reactions: ',r_num)
            print('Max likelihood for rate ',i,': ',r_num/integr)
            for c in np.arange(0,10,0.01):
                temp = np.append(temp,(c**r_num)*np.exp(-c*integr))
            lhd = np.append(lhd,[temp],axis=0) # normalised lhd
        
        for i in [2,3]: # loop over pop_B dependent parameter
            r_num = np.sum(t_data[:,1]==i) # number of reactions
            integr = np.sum(dt*p_data[:,1])
            
            temp = np.zeros(0)
            print('Number of type ',i,' reactions: ',r_num)
            print('Max likelihood for rate ',i,': ',r_num/integr)
            for c in np.arange(0,10,0.01):
                temp = np.append(temp,(c**r_num)*np.exp(-c*integr))
            lhd = np.append(lhd,[temp],axis=0) # normalised lhd
    
    return [lhd]


# calculate complete-data likelihood (for mass-action kinetic rate laws, otherwise does not factorise)
def BR(p_data,t_data,dt,model): # c is parameter of interest, transition rate
    
    if model=='X':
        r_times = np.array(t_data[t_data[:,1]==1,0],dtype=int) # get reaction times A->AB
        n = np.sum(t_data[:,1]==1) # get number of reactions A->AB
    
        intA = np.sum(dt*p_data[:,0]) # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
        intB = np.sum(dt*p_data[:,1])        
    
        prod = 1     
        for k in r_times:
            prod = prod*p_data[k,0]/p_data[k,1]
    
        return prod*(intB/intA)**(n+1)
    
    if model=='Y':
        r_times = np.array(t_data[t_data[:,1]==2,0],dtype=int) # get reaction times A->AB
        n = np.sum(t_data[:,1]==2) # get number of reactions A->AB
    
        intA = np.sum(dt*p_data[:,0]) # integral: int[0,T] g(N) dt, where hazardf = c*g(N)
        intB = np.sum(dt*p_data[:,1])        
    
        prod = 1     
        for k in r_times:
            prod = prod*p_data[k,0]/p_data[k,1]
    
        return prod*(intB/intA)**(n+1)




# %% Model Parameter inference given Model data

# time-series input path and result output path
model = 'X'
INPUTPATH = "C:/Users/Liam/Desktop/Master/simulation/model_"+model+"/data/"
OUTPUTPATH = "C:/Users/Liam/Desktop/Master/inference/time_series/model_"+model+"/"
#p = PdfPages(OUTPUTPATH+"sample1_tvar.pdf")

L_aa = 0
L_ab = 1
L_bb = 0

L_a = 1
L_b = 1
k_ab = 0
k_ba = 0


################## data input ##################

if(model=='X'):
    pop_A = np.load(INPUTPATH+"popA"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".dat")
    pop_B = np.load(INPUTPATH+"popB"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".dat")
    npzfile = np.load(INPUTPATH+"summary"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".npz")
    
if(model=='Y'):
    pop_A = np.load(INPUTPATH+"popA"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".dat")
    pop_B = np.load(INPUTPATH+"popB"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".dat")
    npzfile = np.load(INPUTPATH+"summary"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".npz")
   
dt = npzfile["dt"]
samples = npzfile["samples"]




# %% Plotting likelihoods
################## plotting results ##################

# scan over different time intervals
for i in range(0,4):
    T = int((i+1)*np.shape(pop_A)[1]/4)
    sample_A = pop_A[5,0:T]
    sample_B = pop_B[5,0:T]

    (t_data,p_data) = trans_dat(sample_A,sample_B,model)
    lhd = L_func(p_data,t_data,model,dt)[0] # get likelihoods
    
    if model=='X':
        # normalise likelihoods
        lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
        lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
        lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)


        fig = plt.figure(figsize=(6, 4))
        a = fig.add_subplot(1, 1, 1)
        plt.plot(np.arange(0, 10, 0.01), lhd[0, :])
        plt.vlines(L_aa, 0, 5)
        plt.xlim(0, 2)
        plt.yticks([1,2,3,4,5])
        plt.xlabel(r'$\lambda_{AA}$',fontsize=15)
        plt.ylabel(r'$L(\xi|\lambda_{AA})$ [a.u.]',fontsize=15)
        plt.savefig("C:/Users/Liam/Desktop/Master/"+'Tinescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'_t0-'+str(int(T*dt))+'_samples'+str(i)+'.png', bbox_inches="tight", dpi=600)


        # plot likelihoods
        fig = plt.figure(figsize=(16,6))
        fig.suptitle(r'Model X: Time-series estimation [0,'+str(T*dt)+'], 1 sample', fontsize=12,y=1.05)
    
        ax = plt.subplot(131)
        ax.set_title(r'Likelihood of $\lambda_{AA}$')
        ax.plot(np.arange(0,10,0.01),lhd[0,:])
        ax.vlines(L_aa,0,max(lhd[0,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{AA}$')

        ax = plt.subplot(132)
        ax.set_title(r'Likelihood of $\lambda_{AB}$')
        ax.plot(np.arange(0,10,0.01),lhd[1,:])
        ax.vlines(L_ab,0,max(lhd[1,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{AB}$')

        ax = plt.subplot(133)
        ax.set_title(r'Likelihood of $\lambda_{BB}$')
        ax.plot(np.arange(0,10,0.01),lhd[2,:])
        ax.vlines(L_bb,0,max(lhd[2,:]))
        ax.set_xlim(0,1)
        ax.set_xlabel(r'$\lambda_{BB}$')
        
        plt.tight_layout()
        plt.savefig(OUTPUTPATH+'Timescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'_t0-'+str(int(T*dt))+'_samples1.png', bbox_inches="tight")
    
    if model=='Y':
        # normalise likelihoods
        lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
        lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
        lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
        lhd[3,:] = lhd[3,:]/simps(lhd[3,:], dx=0.01)
        fig = plt.figure(figsize=(20,6))
        fig.suptitle(r'Model Y: Time-series estimation [0,'+str(T*dt)+'], 1 sample', fontsize=12,y=1.05)
        
        ax = plt.subplot(141)
        ax.set_title(r'Likelihood of $\lambda_{A}$')
        ax.plot(np.arange(0,10,0.01),lhd[0,:])
        ax.vlines(L_a,0,max(lhd[0,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{A}$')

        ax = plt.subplot(142)
        ax.set_title(r'Likelihood of $k_{AB}$')
        ax.plot(np.arange(0,10,0.01),lhd[1,:])
        ax.vlines(k_ab,0,max(lhd[1,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$k_{AB}$')

        ax = plt.subplot(143)
        ax.set_title(r'Likelihood of $\lambda_{B}$')
        ax.plot(np.arange(0,10,0.01),lhd[2,:])
        ax.vlines(L_b,0,max(lhd[2,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{B}$')
        
        ax = plt.subplot(144)
        ax.set_title(r'Likelihood of $k_{BA}$')
        ax.plot(np.arange(0,10,0.01),lhd[3,:])
        ax.vlines(k_ba,0,max(lhd[3,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$k_{BA}$')
            
        plt.tight_layout()
        plt.savefig(OUTPUTPATH+'Timescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'_t0-'+str(int(T*dt))+'_samples1.png', bbox_inches="tight")

    
# scan over different sample sizes
for i in range(1,9): 
    T = int(1/dt)
    
    # create lkh for first sample
    sample_A = pop_A[0,0:T]
    sample_B = pop_B[0,0:T]
    (t_data,p_data) = trans_dat(sample_A,sample_B,model)
    lhd = L_func(p_data,t_data,model,dt)[0] # get likelihoods

    # add desired amount of additional samples
    for j in range(1,i+1):
        sample_A = pop_A[j+20,0:T]
        sample_B = pop_B[j+20,0:T]
        (t_data,p_data) = trans_dat(sample_A,sample_B,model)
        lhd = lhd*L_func(p_data,t_data,model,dt)[0] # get likelihoods 
    
    if model=='X':
        # normalise likelihoods
        lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
        lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
        lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)


        fig = plt.figure(figsize=(6, 4))
        a = fig.add_subplot(1, 1, 1)
        plt.plot(np.arange(0, 10, 0.01), lhd[0, :])
        plt.vlines(L_aa, 0, 1.5)
        plt.xlim(0, 2)
        plt.yticks([0.5,1,1.5])
        plt.xlabel(r'$\lambda_{AA}$',fontsize=15)
        plt.ylabel(r'$L(\xi|\lambda_{AA})$ [a.u.]',fontsize=15)
        plt.savefig("C:/Users/Liam/Desktop/Master/"+'Samplescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'_t0-'+str(int(T*dt))+'_samples'+str(i)+'.png', bbox_inches="tight", dpi=600)


        fig = plt.figure(figsize=(16,6))
        fig.suptitle(r'Model X: Time-series estimation [0,'+str(T*dt)+'], '+str(i)+' sample', fontsize=12,y=1.05)
    
        ax = plt.subplot(131)
        ax.set_title(r'Likelihood of $\lambda_{AA}$')
        ax.plot(np.arange(0,10,0.01),lhd[0,:])
        ax.vlines(L_aa,0,max(lhd[0,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{AA}$')

        ax = plt.subplot(132)
        ax.set_title(r'Likelihood of $\lambda_{AB}$')
        ax.plot(np.arange(0,10,0.01),lhd[1,:])
        ax.vlines(L_ab,0,max(lhd[1,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{AB}$')

        ax = plt.subplot(133)
        ax.set_title(r'Likelihood of $\lambda_{BB}$')
        ax.plot(np.arange(0,10,0.01),lhd[2,:])
        ax.vlines(L_bb,0,max(lhd[2,:]))
        ax.set_xlim(0,1)
        ax.set_xlabel(r'$\lambda_{BB}$')
        
        plt.tight_layout()
        plt.savefig(OUTPUTPATH+'Samplescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'_t0-'+str(int(T*dt))+'_samples'+str(i)+'.png', bbox_inches="tight")
    
    if model=='Y':
        # normalise likelihoods
        lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
        lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
        lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
        lhd[3,:] = lhd[3,:]/simps(lhd[3,:], dx=0.01)
        fig = plt.figure(figsize=(20,6))
        fig.suptitle(r'Model Y: Time-series estimation [0,'+str(T*dt)+'], '+str(i)+' sample', fontsize=12,y=1.05)
        
        ax = plt.subplot(141)
        ax.set_title(r'Likelihood of $\lambda_{A}$')
        ax.plot(np.arange(0,10,0.01),lhd[0,:])
        ax.vlines(L_a,0,max(lhd[0,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{A}$')

        ax = plt.subplot(142)
        ax.set_title(r'Likelihood of $k_{AB}$')
        ax.plot(np.arange(0,10,0.01),lhd[1,:])
        ax.vlines(k_ab,0,max(lhd[1,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$k_{AB}$')

        ax = plt.subplot(143)
        ax.set_title(r'Likelihood of $\lambda_{B}$')
        ax.plot(np.arange(0,10,0.01),lhd[2,:])
        ax.vlines(L_b,0,max(lhd[2,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$\lambda_{B}$')
        
        ax = plt.subplot(144)
        ax.set_title(r'Likelihood of $k_{BA}$')
        ax.plot(np.arange(0,10,0.01),lhd[3,:])
        ax.vlines(k_ba,0,max(lhd[3,:]))
        ax.set_xlim(0,2)
        ax.set_xlabel(r'$k_{BA}$')

        plt.tight_layout()
        plt.savefig(OUTPUTPATH+'Samplescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'_t0-'+str(int(T*dt))+'_samples'+str(i)+'.png', bbox_inches="tight")













# %% Calculating coef of var

################## plotting results ##################
if model=='X': 
    cov_t = np.zeros((25,8,3))
    cov_s = np.zeros((25,10,3))
else: 
    cov_t = np.zeros((25,8,4))
    cov_s = np.zeros((25,10,4))

# scan over different time intervals
for k in range(0,25): # loop over different trajectories
    for i in range(0,8): # time loop
        T = int((i+1)*np.shape(pop_A)[1]/8)
        sample_A = pop_A[k,0:T]
        sample_B = pop_B[k,0:T]
        (t_data,p_data) = trans_dat(sample_A,sample_B,model)
        lhd = L_func(p_data,t_data,model,dt)[0] # get likelihoods
    
        if model=='X':
            # normalise likelihoods
            lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
            lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
            lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
            # calculate mean and var
            m1 = simps(np.arange(0,10,0.01)*lhd[0,:],dx=0.01)
            v1 = simps((np.arange(0,10,0.01)**2)*lhd[0,:],dx=0.01)-m1**2
            c1 = np.sqrt(v1)/m1
            m2 = simps(np.arange(0,10,0.01)*lhd[1,:],dx=0.01)
            v2 = simps((np.arange(0,10,0.01)**2)*lhd[1,:],dx=0.01)-m2**2
            c2 = np.sqrt(v2)/m2
            m3 = simps(np.arange(0,10,0.01)*lhd[2,:],dx=0.01)
            v3 = simps((np.arange(0,10,0.01)**2)*lhd[2,:],dx=0.01)-m3**2
            c3 = np.sqrt(v3)/m3
            cov_t[k,i,:] = [c1,c2,c3]
    
        if model=='Y':
            # normalise likelihoods
            lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
            lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
            lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
            lhd[3,:] = lhd[3,:]/simps(lhd[3,:], dx=0.01)
            # calculate mean and var
            m1 = simps(np.arange(0,10,0.01)*lhd[0,:],dx=0.01)
            v1 = simps((np.arange(0,10,0.01)**2)*lhd[0,:],dx=0.01)-m1**2
            c1 = np.sqrt(v1)/m1
            m2 = simps(np.arange(0,10,0.01)*lhd[1,:],dx=0.01)
            v2 = simps((np.arange(0,10,0.01)**2)*lhd[1,:],dx=0.01)-m2**2
            c2 = np.sqrt(v2)/m2
            m3 = simps(np.arange(0,10,0.01)*lhd[2,:],dx=0.01)
            v3 = simps((np.arange(0,10,0.01)**2)*lhd[2,:],dx=0.01)-m3**2
            c3 = np.sqrt(v3)/m3
            m4 = simps(np.arange(0,10,0.01)*lhd[3,:],dx=0.01)
            v4 = simps((np.arange(0,10,0.01)**2)*lhd[3,:],dx=0.01)-m4**2
            c4 = np.sqrt(v4)/m4
            cov_t[k,i,:] = [c1,c2,c3,c4]
        

# scan over different sample sizes
for k in range(0,25): # loop over different trajectory sets
    s_start = 10*k+1
    s_stop = 10*k+11
    for i in range(s_start,s_stop): # sample loop 
        T = int(np.shape(pop_A)[1]/4)
    
        # create lkh for first sample
        sample_A = pop_A[s_start-1,0:T]
        sample_B = pop_B[s_start-1,0:T]
        (t_data,p_data) = trans_dat(sample_A,sample_B,model)
        lhd = L_func(p_data,t_data,model,dt)[0] # get likelihoods
        
        # add desired amount of additional samples
        for j in range(s_start,i):
            sample_A = pop_A[j,0:T]
            sample_B = pop_B[j,0:T]
            (t_data,p_data) = trans_dat(sample_A,sample_B,model)
            lhd = lhd*L_func(p_data,t_data,model,dt)[0] # get likelihoods 
    
        if model=='X':
            # normalise likelihoods
            lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
            lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
            lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
            # calculate mean and var
            m1 = simps(np.arange(0,10,0.01)*lhd[0,:],dx=0.01)
            v1 = simps((np.arange(0,10,0.01)**2)*lhd[0,:],dx=0.01)-m1**2
            c1 = np.sqrt(v1)/m1
            m2 = simps(np.arange(0,10,0.01)*lhd[1,:],dx=0.01)
            v2 = simps((np.arange(0,10,0.01)**2)*lhd[1,:],dx=0.01)-m2**2
            c2 = np.sqrt(v2)/m2
            m3 = simps(np.arange(0,10,0.01)*lhd[2,:],dx=0.01)
            v3 = simps((np.arange(0,10,0.01)**2)*lhd[2,:],dx=0.01)-m3**2
            c3 = np.sqrt(v3)/m3
            cov_s[k,int(i%10)-1,:] = [c1,c2,c3]
    
        if model=='Y':
            # normalise likelihoods
            lhd[0,:] = lhd[0,:]/simps(lhd[0,:], dx=0.01) 
            lhd[1,:] = lhd[1,:]/simps(lhd[1,:], dx=0.01) 
            lhd[2,:] = lhd[2,:]/simps(lhd[2,:], dx=0.01)
            lhd[3,:] = lhd[3,:]/simps(lhd[3,:], dx=0.01)
            # calculate mean and var
            m1 = simps(np.arange(0,10,0.01)*lhd[0,:],dx=0.01)
            v1 = simps((np.arange(0,10,0.01)**2)*lhd[0,:],dx=0.01)-m1**2
            c1 = np.sqrt(v1)/m1
            m2 = simps(np.arange(0,10,0.01)*lhd[1,:],dx=0.01)
            v2 = simps((np.arange(0,10,0.01)**2)*lhd[1,:],dx=0.01)-m2**2
            c2 = np.sqrt(v2)/m2
            m3 = simps(np.arange(0,10,0.01)*lhd[2,:],dx=0.01)
            v3 = simps((np.arange(0,10,0.01)**2)*lhd[2,:],dx=0.01)-m3**2
            c3 = np.sqrt(v3)/m3
            m4 = simps(np.arange(0,10,0.01)*lhd[3,:],dx=0.01)
            v4 = simps((np.arange(0,10,0.01)**2)*lhd[3,:],dx=0.01)-m4**2
            c4 = np.sqrt(v4)/m4
            cov_s[k,int(i%10)-1,:] = [c1,c2,c3,c4]



# %% MODEL X: plot cov as a function of samplesize or time

# factors of proportionality
#fops = 0.65
#fopt = 1.65

fops = 1.00
fopt = 0.72

ls_T = np.linspace(np.shape(pop_A)[1]*dt/8,np.shape(pop_A)[1]*dt,8)
mean_cov_t = np.mean(cov_t[:,:,0],axis=0)
dL = L_aa-L_bb

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model X: Coefficient of variation of $\lambda_{AA}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_T,cov_t[k,:,0],c='lightblue')
plt.plot(ls_T,mean_cov_t,lw=2)
plt.plot(ls_T, fopt*np.exp(-dL*np.array(ls_T)/2), linestyle='--', c='k')
plt.xlabel(r'T')
plt.ylabel(r'$C_{v}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COV_Timescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'.png', bbox_inches="tight",dpi=300)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model X: Coefficient of variation of $\lambda_{AA}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.semilogy(ls_T,cov_t[k,:,0],c='lightblue')
plt.semilogy(ls_T,mean_cov_t,lw=2)
plt.semilogy(ls_T, fopt*np.exp(-dL*np.array(ls_T)/2), linestyle='--', c='k')
plt.xlabel(r'T')
plt.ylabel(r'$log(C_{v})$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COVlog_Timescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'.png', bbox_inches="tight",dpi=300)


ls_S = np.arange(1,11,1)
mean_cov_s = np.mean(cov_s[:,:,0],axis=0)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model X: Coefficient of variation of $\lambda_{AA}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_S,cov_s[k,:,0],c='lightblue')
plt.plot(ls_S,mean_cov_s,lw=2)
plt.plot(ls_S,fops/np.sqrt(ls_S), linestyle='--', c='k')
plt.xlabel(r'N')
plt.ylabel(r'$C_{v}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COV_Samplescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'.png', bbox_inches="tight",dpi=300)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model X: Coefficient of variation of $\lambda_{AA}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_S,1/cov_s[k,:,0]**2,c='lightblue')
plt.plot(ls_S,1/mean_cov_s**2,lw=2)
plt.plot(ls_S,ls_S/(fops**2), linestyle='--', c='k')
plt.xlabel(r'N')
plt.ylabel(r'$C_{v}^{-2}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COVlog_Samplescan_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'.png', bbox_inches="tight",dpi=300)

# %% MODEL Y: plot cov as a function of samplesize or time

# factors of proportionality
fops = 0.68
fopt = 1.75

#fops = 0.65
#fopt = 1.25

ls_T = np.linspace(np.shape(pop_A)[1]*dt/8,np.shape(pop_A)[1]*dt,8)
mean_cov_t = np.mean(cov_t[:,:,0],axis=0)
dL = (L_a+L_b-k_ab-k_ba+np.sqrt((L_a-k_ab-L_b+k_ba)**2 + 4*k_ab*k_ba))/2

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model Y: Coefficient of variation of $\lambda_{A}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_T,cov_t[k,:,0],c='lightblue')
plt.plot(ls_T,mean_cov_t,lw=2)
plt.plot(ls_T, fopt*np.exp(-dL*np.array(ls_T)/2), linestyle='--', c='k')
plt.xlabel(r'T')
plt.ylabel(r'$C_{v}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COV_Timescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'.png', bbox_inches="tight",dpi=300)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model Y: Coefficient of variation of $\lambda_{A}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.semilogy(ls_T,cov_t[k,:,0],c='lightblue')
plt.semilogy(ls_T,mean_cov_t,lw=2)
plt.semilogy(ls_T, fopt*np.exp(-dL*np.array(ls_T)/2), linestyle='--', c='k')
plt.xlabel(r'T')
plt.ylabel(r'$log(C_{v})$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COVlog_Timescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'.png', bbox_inches="tight",dpi=300)


ls_S = np.arange(1,11,1)
mean_cov_s = np.mean(cov_s[:,:,0],axis=0)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model Y: Coefficient of variation of $\lambda_{A}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_S,cov_s[k,:,0],c='lightblue')
plt.plot(ls_S,mean_cov_s,lw=2)
plt.plot(ls_S,fops/np.sqrt(ls_S), linestyle='--', c='k')
plt.xlabel(r'N')
plt.ylabel(r'$C_{v}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COV_Samplescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'.png', bbox_inches="tight",dpi=300)

fig = plt.figure(figsize=(5,4))
fig.suptitle(r'Model Y: Coefficient of variation of $\lambda_{A}$', fontsize=12,y=1.05)        
for k in range(0,25):
    plt.plot(ls_S,1/cov_s[k,:,0]**2,c='lightblue')
plt.plot(ls_S,1/mean_cov_s**2,lw=2)
plt.plot(ls_S,ls_S/(fops**2), linestyle='--', c='k')
plt.xlabel(r'N')
plt.ylabel(r'$C_{v}^{-2}$')
plt.tight_layout()
plt.savefig(OUTPUTPATH+'COVlog_Samplescan_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'.png', bbox_inches="tight",dpi=300)













# %% Model inference given unknown data

# time-series input path and result output path
dat_model = 'Y'

INPUTPATH = "C:/Users/Liam/Desktop/Master/simulation/model_"+dat_model+"/data/"
OUTPUTPATH = 'C:/Users/Liam/Desktop/Master/inference/time_series/comparison/'+dat_model+'_data/'
x_lim = 5.0
dx = 0.1
#p = PdfPages(OUTPUTPATH+"sample1_tvar.pdf")

L_aa = 0
L_ab = 1
L_bb = 0

# for model comparison: k_ab=k_ba=0 (Y) and L_bb=0 (X) otherwise foribidden transition
L_a = 0
L_b = 1
k_ab = 0
k_ba = 0


################## data input ##################

if(dat_model=='X'):
    pop_A = np.load(INPUTPATH+"popA"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".dat")
    pop_B = np.load(INPUTPATH+"popB"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".dat")
    npzfile = np.load(INPUTPATH+"summary"+"_aa"+str(L_aa)+"_ab"+str(L_ab)+"_bb"+str(L_bb)+".npz")
    
if(dat_model=='Y'):
    pop_A = np.load(INPUTPATH+"popA"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".dat")
    pop_B = np.load(INPUTPATH+"popB"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".dat")
    npzfile = np.load(INPUTPATH+"summary"+"_a"+str(L_a)+"_b"+str(L_b)+"_ab"+str(k_ab)+"_ba"+str(k_ba)+".npz")
   
dt = npzfile["dt"]
samples = npzfile["samples"]



# %% calculate mean of base Ratio for different times
bf = np.zeros((350,11))

for k in range(0,350): # loop over samples
    print('Calculating sample ',k)
    for i in range(0,11): # loop over time
        T = int(i*np.shape(pop_A)[1]/10)
        if T!=0:
            sample_A = pop_A[k,0:T]
            sample_B = pop_B[k,0:T]
        else: 
            sample_A = pop_A[k,0:1]
            sample_B = pop_B[k,0:1]

        (t_data,p_data) = trans_dat(sample_A,sample_B,dat_model)
        bf[k,i] = BR(p_data,t_data,dt,dat_model)
    

m_bf = np.zeros(11)
md_bf = np.zeros(11)
std_bf = np.zeros(11)
lw_bf = np.zeros(11)
hg_bf = np.zeros(11)
for k in range(0,11):
    temp = bf[:,k]
    temp = temp[~np.isnan(temp)]
    temp = temp[~np.isinf(temp)]
    m_bf[k] = np.mean(temp)
    md_bf[k] = np.median(temp)
    lw_bf[k] = np.quantile(temp, 0.05)
    hg_bf[k] = np.quantile(temp, 0.95)
    std_bf[k] = np.std(temp)


from scipy.special import gamma
def fb_y(x,a,b):
    return gamma(b)/(a*gamma(b*np.exp(x))) * (b*(np.exp(x)-1)/x)**(1+b*(np.exp(x)-1))

def fb_x(x,a,b):
    return gamma(b)/(a*gamma(b+a*x)) * (b+a*x/2)**(1+a*x)

def fa(x,a,b):
    return b*x/(a*(np.exp(x)-1))



fig = plt.figure(figsize=(5,4))
x = np.linspace(0,4,100)
#plt.plot(x,fa(x,1,1))
#plt.plot(x,fb_x(x,3,1))
plt.plot(x,fb_y(x,1,3))

plt.scatter(np.linspace(0,4,len(m_bf)),m_bf)
#for k in range(0,11):
#    plt.vlines(k*0.5,lw_bf[k],hg_bf[k],colors='b')
#plt.yscale('log')
plt.xticks([0,1,2,3,4])
plt.hlines(1,0,4,linestyle='--')
plt.xlabel(r'T')
plt.ylabel(r'BF')
plt.tight_layout()
plt.show()
plt.savefig("C:/Users/Liam/Desktop/Master/BF_a3b1_0100", bbox_inches="tight", dpi=300)


if dat_model=='X':
    fig.suptitle(r'Bayes factor using Model '+dat_model+' data: $\lambda_{AA}$='+str(L_aa)+", $\lambda_{AB}=$"+str(L_ab), fontsize=12,y=1.05)        
    plt.savefig(OUTPUTPATH+'BFXt'+'_aa'+str(L_aa)+'_ab'+str(L_ab)+'_bb'+str(L_bb)+'_samples'+str(len(bf))+'.png', bbox_inches="tight",dpi=300)
if dat_model=='Y':
    fig.suptitle(r'Bayes factor using Model '+dat_model+' data: $\lambda_{A}$='+str(L_a)+", $\lambda_{B}=$"+str(L_b), fontsize=12,y=1.05)        
    plt.savefig(OUTPUTPATH+'BFYt'+'_a'+str(L_a)+'_b'+str(L_b)+'_kab'+str(k_ab)+'_kba'+str(k_ba)+'_samples'+str(len(bf))+'.png', bbox_inches="tight",dpi=300)













