
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
PATH_DAT = '/Users/linus/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/inference'
PATH_OUT = '/Users/linus/Dropbox/projects/cellStates/MScPhysProject/Stem-cell-inference/pyabc'
# PATH_DAT = 'C:/Users/Liam/Desktop/Stem-cell-inference-master/inference'
# PATH_OUT = 'C:/Users/Liam/Desktop/Stem-cell-inference-master/pyabc'

# bounds of log-uniform priors
PAR_MIN = 0.01
PAR_MAX = 1.0

# SMC-ABC population size
POP_SIZE = 10000
transition_kernels = [pyabc.transition.LocalTransition(k_fraction=0.25),
            pyabc.transition.LocalTransition(k_fraction=0.25)]
#################################  Model definitions  #################################

# model C definition
def model_C(parameter):
    par = np.array([parameter['p1'],parameter['p2'],parameter['p3'],parameter['p4'],parameter['p5'],parameter['p6'],parameter['p7']])
    MC_N = 100
    output = dict()

    # iterate over Model X initial coniditions
    for ic in range(0, N_set):
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

        # special case: sample from D2 population distribution given T+ at D0
        elif T_marker == 21:
            pop_init_arr = np.load(PATH_DAT+'/D2_simpop_CHIR_Tp_X.npy')
            pop_init = pop_init_arr[np.random.choice(len(pop_init_arr))]

        # special case: sample from D2 population distribution given T- at D0
        elif T_marker == 20:
            pop_init_arr = np.load(PATH_DAT+'/D2_simpop_CHIR_Tm_X.npy')
            pop_init = pop_init_arr[np.random.choice(len(pop_init_arr))]

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
    for ic in range(0, N_set):
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

        # special case: sample from D2 population distribution given T+ at D0
        elif T_marker == 21:
            pop_init_arr = np.load(PATH_DAT + '/D2_simpop_CHIR_Tp_X.npy')
            pop_init = pop_init_arr[np.random.choice(len(pop_init_arr))]

        # special case: sample from D2 population distribution given T- at D0
        elif T_marker == 20:
            pop_init_arr = np.load(PATH_DAT + '/D2_simpop_CHIR_Tm_X.npy')
            pop_init = pop_init_arr[np.random.choice(len(pop_init_arr))]

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
    for ic in range(0,N_set):
        # abs median deviation and sum over individual cell states
        med_dev = np.sum(np.abs(x['data'+str(ic+1)][1] - y['data'+str(ic+1)][1]))
        # mean deviation
        mean_dev = np.sum(np.abs(x['data'+str(ic+1)][0] - y['data'+str(ic+1)][0]))
        # std deviation
        std_dev = np.sum(np.abs(x['data'+str(ic+1)][2] - y['data'+str(ic+1)][2]))
        d += np.array([med_dev, mean_dev, std_dev])
    eps = np.sum(d)
    return eps



# Load experimental data (here: CHIR data)
xls = pd.ExcelFile(PATH_DAT+'/raw_clonal_data_adj.xlsx')
Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D2')
pop1 = Tp_TF.values
stat_true1 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D2')
pop1 = Tp_TF.values
stat_true2 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D2')
pop1 = Tp_TF.values
stat_true3 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D2')
pop1 = Tp_TF.values
stat_true4 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TF-D3')
pop1 = Tp_TF.values
stat_true5 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tp-TS-D3')
pop1 = Tp_TF.values
stat_true6 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TF-D3')
pop1 = Tp_TF.values
stat_true7 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]
Tp_TF = pd.read_excel(xls, 'CHIR-Tm-TS-D3')
pop1 = Tp_TF.values
stat_true8 = [np.mean(pop1,axis=0), np.median(pop1,axis=0), np.std(pop1,axis=0)]



# Metadata of experimental data (here: CHIR D2+D3 data)
N_set = 8
T_ls = [2,2,2,2,3,3,3,3]
marker_unobs_ls = ['S','F','S','F','S','F','S','F']
T_marker_ls = [1,1,0,0,1,1,0,0]
observations = [stat_true1,stat_true2,stat_true3,stat_true4,stat_true5,stat_true6,stat_true7,stat_true8]

# # Set up ABC-SMC inference for model comparison
# abc = pyabc.ABCSMC(models, priors, distance, population_size=POP_SIZE,
#                     transitions=transition_kernels)
# db_path = ("sqlite:///"+PATH_OUT+"/CHIR_combined.db")
# abc.new(db_path, {"data1": observations[0],"data2": observations[1],"data3": observations[2],"data4": observations[3],"data5": observations[4],"data6": observations[5],"data7": observations[6],"data8": observations[7]})
# history = abc.run(minimum_epsilon=10, max_nr_populations=20)



# Metadata of experimental data (here: CHIR D2 data)
N_set = 4
T_ls = [2,2,2,2] # because runs from D0 to D2
marker_unobs_ls = ['S','F','S','F']
T_marker_ls = [1,1,0,0] # flag that initial conditions should be sampled from existing D2 populations
observations = [stat_true1,stat_true2,stat_true3,stat_true4]

# # Set up ABC-SMC inference for model comparison
# abc = pyabc.ABCSMC(models, priors, distance, population_size=POP_SIZE,
#                     transitions=transition_kernels)
# db_path = ("sqlite:///"+PATH_OUT+"/CHIR_D2.db")
# abc.new(db_path, {"data1": observations[0],"data2": observations[1],"data3": observations[2],"data4": observations[3]})
# history = abc.run(minimum_epsilon=5, max_nr_populations=20)



# Metadata of experimental data (here: CHIR D3 data)
N_set = 4
T_ls = [1,1,1,1] # because runs from D2 to D3
marker_unobs_ls = ['S','F','S','F']
T_marker_ls = [21,21,20,20] # flag that initial conditions should be sampled from existing D2 populations
observations = [stat_true5,stat_true6,stat_true7,stat_true8]

# # Set up ABC-SMC inference for model comparison
# abc = pyabc.ABCSMC(models, priors, distance, population_size=POP_SIZE,
#                     transitions=transition_kernels)
# db_path = ("sqlite:///"+PATH_OUT+"/CHIR_D3.db")
# abc.new(db_path, {"data1": observations[0],"data2": observations[1],"data3": observations[2],"data4": observations[3]})
# history = abc.run(minimum_epsilon=5, max_nr_populations=20)






# VISUALIZATION (assuming the EPISC run is stored in the same directory with name 'EPISC_D3.db')

# load existing CHIR run
abc_continued = pyabc.ABCSMC(models, priors, distance)
db_path = ("sqlite:///"+PATH_OUT+"/CHIR_combined.db")
abc_continued.load(db_path, 2) # second argument is abc_id
df_CH, w_CH = abc_continued.history.get_distribution(m=0)
# Visualise model comparison
pyabc.visualization.plot_model_probabilities(abc_continued.history)
plt.savefig("model_comparison_CHIR.pdf")

# load existing EPISC run
abc_continued = pyabc.ABCSMC(models, priors, distance)
db_path = ("sqlite:///"+PATH_OUT+"/EPISC_D3.db")
abc_continued.load(db_path, 1)
df_EP, w_EP = abc_continued.history.get_distribution(m=0)
# Visualise model comparison
pyabc.visualization.plot_model_probabilities(abc_continued.history)
plt.savefig("model_comparison_EPISC.pdf")

# # load existing CHIR D2 run
# abc_continued = pyabc.ABCSMC(models, priors, distance)
# db_path = ("sqlite:///"+PATH_OUT+"/CHIR_D2.db")
# abc_continued.load(db_path, 1)
# df_CH2, w_CH2 = abc_continued.history.get_distribution(m=0)
#
# # load existing CHIR D3 run
# abc_continued = pyabc.ABCSMC(models, priors, distance)
# db_path = ("sqlite:///"+PATH_OUT+"/CHIR_D3.db")
# abc_continued.load(db_path, 1)
# df_CH3, w_CH3 = abc_continued.history.get_distribution(m=0)




# # Compare posterior means of CHIR/EPISC
# import matplotlib
# matplotlib.rcParams.update({'font.size': 16})
# from scipy.integrate import simps
# x = np.logspace(-2,0,1000)
# stat_CHIR = np.zeros((7,2))
# for idx in range(0,7):
#     k1 = gaussian_kde(np.log10(df_CH.values[:,idx]),weights=w_CH)
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_CHIR[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                          simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]
# stat_EPISC = np.zeros((7,2))
# for idx in range(0,7):
#     k1 = gaussian_kde(np.log10(df_EP.values[:,idx]),weights=w_EP)
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_EPISC[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
#
# # width of the bars
# barWidth = 0.3
# # Choose the height of bars
# bars1 = 10**stat_CHIR[:,0]
# bars2 = 10**stat_EPISC[:,0]
# # Choose the height of the error bars (use std deviation)
# yer1 = [10**(stat_CHIR[:,0]-np.sqrt(stat_CHIR[:,1])), 10**(stat_CHIR[:,0]+np.sqrt(stat_CHIR[:,1]))]
# yer2 = [10**(stat_EPISC[:,0]-np.sqrt(stat_EPISC[:,1])), 10**(stat_EPISC[:,0]+np.sqrt(stat_EPISC[:,1]))]
# # The x position of bars
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# # Create bars
# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(111)
# ax.bar(r1, bars1, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='CHIR')
# ax.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='EPISC')
# ax.vlines(r1, yer1[0], yer1[1])
# ax.vlines(r2, yer2[0], yer2[1])
# # general layout
# ax.set_yscale('log')
# ax.set_ylim(0.01,1.1)
# par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
# ax.set_xticks([r + barWidth for r in range(len(bars1))])
# ax.set_xticklabels(par_name)
# ax.set_ylabel('Rate [1/day]')
# plt.legend()
# # plt.show()
# plt.savefig('posterior_means.pdf')
#
#
#
# # Compare posterior rate imbalance of CHIR/EPISC
# x = np.logspace(-3,3,1000)
# stat_CHIR = np.zeros((3, 2))
# for idx in range(0, 3):
#     k1 = gaussian_kde(np.log10(df_CH.values[:, 2 * idx + 2] / df_CH.values[:, 2 * idx + 1]),weights=w_CH)
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_CHIR[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                          simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]
# stat_EPISC = np.zeros((3, 2))
# for idx in range(0, 3):
#     k1 = gaussian_kde(np.log10(df_EP.values[:, 2 * idx + 2] / df_EP.values[:, 2 * idx + 1]),weights=w_EP)
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_EPISC[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                           simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]
#
# # width of the bars
# barWidth = 0.2
# # Choose the height of bars
# bars1 = 10 ** stat_CHIR[:, 0]
# bars2 = 10 ** stat_EPISC[:, 0]
# # Choose the height of the error bars (use std deviation)
# yer1 = [10 ** (stat_CHIR[:, 0] - np.sqrt(stat_CHIR[:, 1])), 10 ** (stat_CHIR[:, 0] + np.sqrt(stat_CHIR[:, 1]))]
# yer2 = [10 ** (stat_EPISC[:, 0] - np.sqrt(stat_EPISC[:, 1])), 10 ** (stat_EPISC[:, 0] + np.sqrt(stat_EPISC[:, 1]))]
# # The x position of bars
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# # Create bars
# fig = plt.figure(figsize=(10, 5))
# ax = plt.subplot(111)
# ax.bar(r1, bars1, width=barWidth, color='lightblue', edgecolor='black', capsize=7, label='CHIR')
# ax.bar(r2, bars2, width=barWidth, color='orange', edgecolor='black', capsize=7, label='EPISC')
# ax.hlines(1, -1, 3, linestyles='dashed')
# ax.vlines(r1, yer1[0], yer1[1])
# ax.vlines(r2, yer2[0], yer2[1])
# # general layout
# # ax.set_ylim(0,2.8)
# ax.set_xlim(-0.3, 2.7)
# ax.set_yscale('log')
# par_name = [r'$\theta_{T+}/\theta_{T-}$', r'$\theta_{S+}/\theta_{S-}$', r'$\theta_{F+}/\theta_{F-}$']
# ax.set_xticks([r + barWidth for r in range(len(bars1))])
# ax.set_xticklabels(par_name)
# ax.set_ylabel(r'Rate imbalance')
# plt.legend()
# # plt.show()
# plt.savefig('rate_imbalance.pdf')




# # Compare posterior means of CHIR D2 and CHIR D3
# x = np.logspace(-2,0,500)
# stat_CHIR2 = np.zeros((7,2))
# for idx in range(0,7):
#     k1 = gaussian_kde(np.log10(df_CH2.values[:,idx]),weights=w_CH2)
#     pdf = k1(np.log10(x))
#     pdf = pdf/simps(pdf,dx=0.0001)
#     stat_CHIR2[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
# stat_CHIR3 = np.zeros((7,2))
# for idx in range(0,7):
#     k1 = gaussian_kde(np.log10(df_CH3.values[:,idx]),weights=w_CH3)
#     pdf = k1(np.log10(x))
#     pdf = pdf/simps(pdf,dx=0.0001)
#     stat_CHIR3[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
# stat_CHIR23 = np.zeros((7,2))
# for idx in range(0,7):
#     k1 = gaussian_kde(np.log10(df_CH.values[:,idx]),weights=w_CH)
#     pdf = k1(np.log10(x))
#     pdf = pdf/simps(pdf,dx=0.0001)
#     stat_CHIR23[idx,:] = [simps(np.log10(x)*pdf,dx=0.0001), simps(np.log10(x)**2 *pdf,dx=0.0001)-simps(np.log10(x)*pdf,dx=0.0001)**2]
#
# # width of the bars
# barWidth = 0.2
# # Choose the height of bars
# bars1 = 10**stat_CHIR2[:,0]
# bars2 = 10**stat_CHIR3[:,0]
# bars3 = 10**stat_CHIR23[:,0]
# # Choose the height of the error bars (use std deviation)
# yer1 = [10**(stat_CHIR2[:,0]-np.sqrt(stat_CHIR2[:,1])), 10**(stat_CHIR2[:,0]+np.sqrt(stat_CHIR2[:,1]))]
# yer2 = [10**(stat_CHIR3[:,0]-np.sqrt(stat_CHIR3[:,1])), 10**(stat_CHIR3[:,0]+np.sqrt(stat_CHIR3[:,1]))]
# yer3 = [10**(stat_CHIR23[:,0]-np.sqrt(stat_CHIR23[:,1])), 10**(stat_CHIR23[:,0]+np.sqrt(stat_CHIR23[:,1]))]
# # The x position of bars
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# # Create bars
# fig = plt.figure(figsize=(10,5),dpi=200)
# matplotlib.rcParams.update({'font.size': 16})
# ax = plt.subplot(111)
# ax.bar(r1, bars1, width = barWidth, color = 'navy', edgecolor = 'black', capsize=7, label='CHIR D2')
# ax.bar(r2, bars3, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='CHIR D2+D3')
# ax.bar(r3, bars2, width = barWidth, color = 'azure', edgecolor = 'black', capsize=7, label='CHIR D3')
# ax.vlines(r1, yer1[0], yer1[1])
# ax.vlines(r3, yer2[0], yer2[1])
# ax.vlines(r2, yer3[0], yer3[1])
# # general layout
# ax.set_yscale('log')
# ax.set_ylim(0.01,1)
# par_name = [r'$\theta_{0}$',r'$\theta_{T-}$',r'$\theta_{T+}$',r'$\theta_{S-}$',r'$\theta_{S+}$',r'$\theta_{F-}$',r'$\theta_{F+}$']
# ax.set_xticks([r + barWidth for r in range(len(bars1))])
# ax.set_xticklabels(par_name)
# ax.set_ylabel('Rate [1/day]')
# plt.legend()
# plt.show()




# # Compare posterior rate imbalance of CHIR D2 and CHIR D3
# x = np.logspace(-3,3,1000)
# stat_CHIR2 = np.zeros((3, 2))
# for idx in range(0, 3):
#     k1 = gaussian_kde(np.log10(df_CH2.values[:, 2*idx+2]/df_CH2.values[:, 2*idx+1]))
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_CHIR2[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                           simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]
# stat_CHIR3 = np.zeros((3, 2))
# for idx in range(0, 3):
#     k1 = gaussian_kde(np.log10(df_CH3.values[:, 2*idx+2]/df_CH3.values[:, 2*idx+1]))
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_CHIR3[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                           simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]
# stat_CHIR23 = np.zeros((3, 2))
# for idx in range(0, 3):
#     k1 = gaussian_kde(np.log10(df_CH.values[:, 2*idx+2]/df_CH.values[:, 2*idx+1]))
#     pdf = k1(np.log10(x))
#     pdf = pdf / simps(pdf, dx=0.0001)
#     stat_CHIR23[idx, :] = [simps(np.log10(x) * pdf, dx=0.0001),
#                           simps(np.log10(x) ** 2 * pdf, dx=0.0001) - simps(np.log10(x) * pdf, dx=0.0001) ** 2]

# # width of the bars
# barWidth = 0.2
# # Choose the height of bars
# bars1 = 10**stat_CHIR2[:,0]
# bars2 = 10**stat_CHIR3[:,0]
# bars3 = 10**stat_CHIR23[:,0]
# # Choose the height of the error bars (use std deviation)
# yer1 = [10**(stat_CHIR2[:,0]-np.sqrt(stat_CHIR2[:,1])), 10**(stat_CHIR2[:,0]+np.sqrt(stat_CHIR2[:,1]))]
# yer2 = [10**(stat_CHIR3[:,0]-np.sqrt(stat_CHIR3[:,1])), 10**(stat_CHIR3[:,0]+np.sqrt(stat_CHIR3[:,1]))]
# yer3 = [10**(stat_CHIR23[:,0]-np.sqrt(stat_CHIR23[:,1])), 10**(stat_CHIR23[:,0]+np.sqrt(stat_CHIR23[:,1]))]
# # The x position of bars
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# # Create bars
# fig = plt.figure(figsize=(10,5),dpi=200)
# matplotlib.rcParams.update({'font.size': 16})
# ax = plt.subplot(111)
# ax.bar(r1, bars1, width = barWidth, color = 'navy', edgecolor = 'black', capsize=7, label='CHIR D2')
# ax.bar(r2, bars3, width = barWidth, color = 'lightblue', edgecolor = 'black', capsize=7, label='CHIR D2+D3')
# ax.bar(r3, bars2, width = barWidth, color = 'azure', edgecolor = 'black', capsize=7, label='CHIR D3')
# ax.vlines(r1, yer1[0], yer1[1])
# ax.vlines(r3, yer2[0], yer2[1])
# ax.vlines(r2, yer3[0], yer3[1])
# # general layout
# ax.hlines(1,-1,3,linestyles='dashed')
# ax.set_xlim(-0.3,2.7)
# ax.set_yscale('log')
# par_name = [r'$\theta_{T+}/\theta_{T-}$',r'$\theta_{S+}/\theta_{S-}$',r'$\theta_{F+}/\theta_{F-}$']
# ax.set_xticks([r + barWidth for r in range(len(bars1))])
# ax.set_xticklabels(par_name)
# ax.set_ylabel(r'Rate imbalance')
# plt.legend()
# plt.show()
