
import pyabc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Local path of data folder and output folder
# PATH_DAT = '/home/linus/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/inference'
# PATH_OUT = '/home/linus/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/pyabc'
PATH_DAT = '~/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/inference'
PATH_OUT = '~/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/pyabc'
# bounds of log-uniform priors
PAR_MIN = 0.01
PAR_MAX = 1.0


# Metadata of experimental data (here: EPISC D3 data)
T_ls = [3,3,3,3]
marker_unobs_ls = ['S','F','S','F']
T_marker_ls = [1,1,0,0]


# Load experimental data (here: EPISC D3 data)
xls = pd.ExcelFile(PATH_DAT+'/raw_clonal_data_adj.xlsx')
Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TF-D3')
pop1 = Tp_TF.values
stat_true1 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'EPISC-Tp-TS-D3')
pop1 = Tp_TF.values
stat_true2 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TF-D3')
pop1 = Tp_TF.values
stat_true3 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'EPISC-Tm-TS-D3')
pop1 = Tp_TF.values
stat_true4 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
observations = [stat_true1,stat_true2,stat_true3,stat_true4]


#################################  Model definitions  #################################

# model C definition
def model_C(parameter):
    par = np.array([parameter['p1'],parameter['p2'],parameter['p3'],parameter['p4'],parameter['p5'],parameter['p6'],parameter['p7']])
    MC_N = 100
    output = dict()

    # iterate over Model X initial coniditions
    for ic in range(0, 4):
        T = T_ls[ic]
        T_marker = T_marker_ls[ic]
        marker_unobs = marker_unobs_ls[ic]

        # if only T+ known: iterate through states with pop_init T+F+S+, T+F+S-, T+F-S+, T+F-S-
        if T_marker == 1:
            pop_init_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0]])
            pop_init = pop_init_arr[np.random.choice(4, 1, p=[0.49 * 0.75, 0.49 * 0.25, 0.51 * 0.75, 0.51 * 0.25])[0]]

        # if only T- known: iterate through states with pop_init T-F+S+, T-F+S-, T-F-S+, T-F-S-
        elif T_marker == 0:
            pop_init_arr = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1]])
            pop_init = pop_init_arr[np.random.choice(4, 1, p=[0.26 * 0.46, 0.26 * 0.54, 0.74 * 0.46, 0.74 * 0.54])[0]]

        pop_ls = np.zeros((0, 4))
        for i in range(0, MC_N):
            # get population from simulation
            pop = trajectory_EPISC_X(par, pop_init, T)
            # project population numers according to known markers
            pop_adj = project_population(pop, marker_unobs)
            pop_ls = np.vstack([pop_ls, pop_adj])
        output['data'+str(ic+1)] = [np.mean(pop_ls, axis=0), np.median(pop_ls, axis=0), np.std(pop_ls, axis=0)]
    return output


# model U definition
def model_U(parameter):
    par = np.array([parameter['p1'],parameter['p2'],parameter['p3'],parameter['p4'],parameter['p5'],parameter['p6'],parameter['p7']])
    MC_N = 100
    output = dict()

    # iterate over Model X initial coniditions
    for ic in range(0, 4):
        T = T_ls[ic]
        T_marker = T_marker_ls[ic]
        marker_unobs = marker_unobs_ls[ic]

        # if only T+ known: iterate through states with pop_init T+F+S+, T+F+S-, T+F-S+, T+F-S-
        if T_marker == 1:
            pop_init_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0]])
            pop_init = pop_init_arr[np.random.choice(4, 1, p=[0.49 * 0.75, 0.49 * 0.25, 0.51 * 0.75, 0.51 * 0.25])[0]]

        # if only T- known: iterate through states with pop_init T-F+S+, T-F+S-, T-F-S+, T-F-S-
        elif T_marker == 0:
            pop_init_arr = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1]])
            pop_init = pop_init_arr[np.random.choice(4, 1, p=[0.26 * 0.46, 0.26 * 0.54, 0.74 * 0.46, 0.74 * 0.54])[0]]

        pop_ls = np.zeros((0, 4))
        for i in range(0, MC_N):
            # get population from simulation
            pop = trajectory_EPISC_Y(par, pop_init, T)
            # project population numers according to known markers
            pop_adj = project_population(pop, marker_unobs)
            pop_ls = np.vstack([pop_ls, pop_adj])
        output['data'+str(ic+1)] = [np.mean(pop_ls, axis=0), np.median(pop_ls, axis=0), np.std(pop_ls, axis=0)]
    return output


# Needed for Model C trajectory generation
def trajectory_EPISC_X(par, N_init, T):
    S = np.zeros((32, 8), dtype=int)
    S[0, 0] = 1
    S[1, 1] = 1
    S[2, 2] = 1
    S[3, 3] = 1
    S[4, 4] = 1
    S[5, 5] = 1
    S[6, 6] = 1
    S[7, 7] = 1
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
    # N_pop = TSF [+++,++-,+-+,+--,-++,...]
    # par = [T-,T+,S-,S+,F-,F+]
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True:
        scale_arr = np.append(N_pop[0:8] * par[0],
                              N_pop[[0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 4, 0, 5, 1, 7, 3, 6, 2]] * par[
                                  [5, 6, 3, 4, 6, 5, 4, 3, 5, 6, 3, 4, 6, 5, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1]])
        with np.errstate(divide='ignore', invalid='ignore'):  # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32)) / scale_arr
            times[np.logical_or(np.logical_or(times == np.inf, times == 0),
                                np.logical_or(times == -np.inf, np.isnan(times)))] = T + 1
        idx = np.where(times == np.amin(times))[0][0]
        rtime = np.append(rtime, rtime[-1] + times[idx])
        rtype = np.append(rtype, idx)
        N_pop += S[idx, :]

        if rtime[-1] > T or np.max(N_pop) >= 15:
            break
    return N_pop


# Needed for Model U trajectory generation
def trajectory_EPISC_Y(par, N_init, T):
    S = np.zeros((32, 8), dtype=int)
    S[0, 0] = 1
    S[1, 1] = 1
    S[2, 2] = 1
    S[3, 3] = 1
    S[4, 4] = 1
    S[5, 5] = 1
    S[6, 6] = 1
    S[7, 7] = 1
    S[8, :] = [-1, 1, 0, 0, 0, 0, 0, 0]
    S[9, :] = [1, -1, 0, 0, 0, 0, 0, 0]
    S[10, :] = [0, -1, 0, 1, 0, 0, 0, 0]
    S[11, :] = [0, 1, 0, -1, 0, 0, 0, 0]
    S[12, :] = [0, 0, 1, -1, 0, 0, 0, 0]
    S[13, :] = [0, 0, -1, 1, 0, 0, 0, 0]
    S[14, :] = [1, 0, -1, 0, 0, 0, 0, 0]
    S[15, :] = [-1, 0, 1, 0, 0, 0, 0, 0]
    S[16, :] = [0, 0, 0, 0, -1, 1, 0, 0]
    S[17, :] = [0, 0, 0, 0, 1, -1, 0, 0]
    S[18, :] = [0, 0, 0, 0, 0, -1, 0, 1]
    S[19, :] = [0, 0, 0, 0, 0, 1, 0, -1]
    S[20, :] = [0, 0, 0, 0, 0, 0, 1, -1]
    S[21, :] = [0, 0, 0, 0, 0, 0, -1, 1]
    S[22, :] = [0, 0, 0, 0, 1, 0, -1, 0]
    S[23, :] = [0, 0, 0, 0, -1, 0, 1, 0]
    S[24, :] = [1, 0, 0, 0, -1, 0, 0, 0]
    S[25, :] = [-1, 0, 0, 0, 1, 0, 0, 0]
    S[26, :] = [0, 1, 0, 0, 0, -1, 0, 0]
    S[27, :] = [0, -1, 0, 0, 0, 1, 0, 0]
    S[28, :] = [0, 0, 0, 1, 0, 0, 0, -1]
    S[29, :] = [0, 0, 0, -1, 0, 0, 0, 1]
    S[30, :] = [0, 0, 1, 0, 0, 0, -1, 0]
    S[31, :] = [0, 0, -1, 0, 0, 0, 1, 0]
    N_pop = N_init.copy()
    rtype = np.zeros(1)
    rtime = np.zeros(1)
    while True:
        scale_arr = np.append(N_pop[0:8] * par[0],
                              N_pop[[0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 4, 0, 5, 1, 7, 3, 6, 2]] * par[
                                  [5, 6, 3, 4, 6, 5, 4, 3, 5, 6, 3, 4, 6, 5, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1]])
        with np.errstate(divide='ignore', invalid='ignore'):  # Zum filtern der unendlichen Werte
            times = -np.log(np.random.rand(32)) / scale_arr
            times[np.logical_or(np.logical_or(times == np.inf, times == 0),
                                np.logical_or(times == -np.inf, np.isnan(times)))] = T + 1
        idx = np.where(times == np.amin(times))[0][0]
        rtime = np.append(rtime, rtime[-1] + times[idx])
        rtype = np.append(rtype, idx)
        N_pop += S[idx, :]

        if rtime[-1] > T or np.max(N_pop) >= 15:
            break
    return N_pop


# project population numbers on observed populations
def project_population(pop,marker_unobs):
    if marker_unobs=='F':
        return np.array([np.sum(pop[0:2]),np.sum(pop[2:4]),np.sum(pop[4:6]),np.sum(pop[6:8])])
    elif marker_unobs=='S':
        return np.array([np.sum(pop[[0,2]]),np.sum(pop[[1,3]]),np.sum(pop[[4,6]]),np.sum(pop[[5,7]])])

#################################  Model definitions end  #################################


# Set up parameter priors for models (identical for Model C and U)
models = [model_C, model_U]
par_prior = pyabc.Distribution(p1=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p2=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p3=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p4=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p5=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p6=pyabc.RV("loguniform", PAR_MIN, PAR_MAX),
                               p7=pyabc.RV("loguniform", PAR_MIN, PAR_MAX))
priors = [par_prior,par_prior]


# distance function, x: simulated data, y: experimental data
def distance(x, y):
    d = np.zeros(3)
    for ic in range(0,4):
        # abs median deviation and sum over individual cell states
        med_dev = np.sum(np.abs(x['data'+str(ic+1)][1] - y['data'+str(ic+1)][1]))
        # mean deviation
        mean_dev = np.sum(np.abs(x['data'+str(ic+1)][0] - y['data'+str(ic+1)][0]))
        # std deviation
        std_dev = np.sum(np.abs(x['data'+str(ic+1)][2] - y['data'+str(ic+1)][2]))
        d += np.array([med_dev, mean_dev, std_dev])
    eps = np.sum(d)
    return eps


# # Option 1 Set up ABC-SMC parameter inference for single model
# abc = pyabc.ABCSMC(model_C, par_prior, distance)
# db_path = ("sqlite:///"+PATH_OUT+"/test.db")
# abc.new(db_path, {"data1": observations[0],"data2": observations[1],"data3": observations[2],"data4": observations[3]})
# history = abc.run(minimum_epsilon=20, max_nr_populations=20)

# # Option 2 Set up ABC-SMC inference for model comparison
# abc = pyabc.ABCSMC(models, priors, distance,population_size=10000,
#                     transitions=[pyabc.transition.LocalTransition(k_fraction=0.25),
#                                 pyabc.transition.LocalTransition(k_fraction=0.25)])
# db_path = ("sqlite:///"+PATH_OUT+"/test.db")
# abc.new(db_path, {"data1": observations[0],"data2": observations[1],"data3": observations[2],"data4": observations[3]})
# history = abc.run(minimum_epsilon=0.1, max_nr_populations=25)

# Option 3 Load or resuming stored ABC run
abc_continued = pyabc.ABCSMC(model_C, par_prior, distance)
db_path = ("sqlite:///"+PATH_OUT+"/test.db")
abc_continued.load(db_path, 1) # second argument is ID which is assigned to SMC run (ID is generated at first execution)
history = abc_continued.history
# history = abc_continued.run(minimum_epsilon=10, max_nr_populations=10)


# Visualise model comparison
pyabc.visualization.plot_model_probabilities(history)
plt.savefig("model_comparison.pdf")


# Visualise posteriors via KDE of sampled parameters
df, w = abc.history.get_distribution(m=0) # m is Model index (0=C, 1=U), optional: t= SMC population index
x = np.logspace(-2,0,500)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
par_col = ['slategray','orangered','tab:red','limegreen','darkgreen','deepskyblue','navy']
for idx in range(0,7):
    k1 = gaussian_kde(np.log10(df.values[:,idx]),weights=w)
    ax.plot(np.log10(x),k1(np.log10(x)),label=par_name[idx])
    ax.fill_between(np.log10(x), k1(np.log10(x)), interpolate=True,alpha=0.25)
#plt.title('Model X: parameter posteriors')
plt.xlim(-2,0.0)
#plt.ylim(0,28)
plt.xlabel(r'$log_{10}$ reaction rate [1/d]')
plt.ylabel(r'$p(\theta)$')
ax.legend()
plt.savefig("posteriors.pdf")
