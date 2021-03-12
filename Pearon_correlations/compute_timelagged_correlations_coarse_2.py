import sys
base_directory = '/Users/giuliobondanelli/OneDrive - Fondazione Istituto Italiano Tecnologia/Code/code_Valente21_github'
sys.path.insert(0,base_directory + '/modules_')
from numpy import *
from encdec import *
from stattools import *
from information import *
from plottingtools import *
from managingtools import *
from lif import *
from matplotlib.pyplot import *
import rc_parameters
import pandas as pd
from scipy.stats import pearsonr

import warnings
warnings.simplefilter("ignore")

[tau, tau_sampling, tau_lowpass, Rin, noisecorr, outputrate_stim1, varout_stim1, CVrateout_stim1, outsvmacc, insvmacc] \
= open_pickle('./datasets/spatial/dataset_noisecorr.pkl')

STI1 = load(base_directory + '/data/data_pearson_correlations/sti1.npy',allow_pickle = True)
STI2 = load(base_directory + '/data/data_pearson_correlations/sti2.npy',allow_pickle = True)
STIM = load(base_directory + '/data/data_pearson_correlations/Stim.npy',allow_pickle = True)
PRED_STIM = load(base_directory + '/data/data_pearson_correlations/Pred_Stim.npy',allow_pickle = True)

n_trials, n_in, n_tau, n_corr, n_tausampling, n_taulp = insvmacc.shape

t = linspace(0,100000,100000000+1)

i_in = 1
i_tau = 0
i_c = 0
i_s = 0
i_lp = 1
dt = 0.001

sti1 = STI1[i_in, i_tau, i_c, i_s, i_lp]
sti2 = STI2[i_in, i_tau, i_c, i_s, i_lp]
sti1_jittered = jitter_spike_train(sti1,1,dt)
sti2_jittered = jitter_spike_train(sti2,1,dt)

stimuli = STIM[i_in, i_tau, i_c, i_s, i_lp]
predicted_stimuli = PRED_STIM[i_in, i_tau, i_c, i_s, i_lp]

n_time_divisions = 2
lags = arange(2)
n_lags = len(lags)
nsubsamplings = 200
frac = .9

correlations_corr_s1 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correlations_corr_s2 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correlations_err_s1  = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correlations_err_s2  = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correction_corr_s1 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correction_corr_s2 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correction_err_s1 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))
correction_err_s2 = nan*ones((nsubsamplings, n_time_divisions-len(lags)+1, n_lags))

i_trial = 0

rates_i_s1, rates_i_s2 = [compute_rate_from_spike_trains2(st, tau_sampling[i_s]/n_time_divisions, t) for st in [sti1,sti2]]
rates_i = hstack((rates_i_s1[:,:,i_trial],rates_i_s2[:,:,i_trial]))

rates_i_s1_jittered, rates_i_s2_jittered = [compute_rate_from_spike_trains2(st, tau_sampling[i_s]/n_time_divisions, t) for st in [sti1_jittered,sti2_jittered]]
rates_i_jittered = hstack((rates_i_s1_jittered[:,:,i_trial],rates_i_s2_jittered[:,:,i_trial]))

x = predicted_stimuli[:, i_trial] == stimuli
xs1 = x[:int(len(stimuli)/2)]
xs2 = x[int(len(stimuli)/2):]
eq_trials_s1 = equalize_trials(xs1, frac, nsubsamplings)
eq_trials_s2 = int(len(stimuli)/2) + equalize_trials(xs2, frac, nsubsamplings)
n_eq_trials_s1 = eq_trials_s1.shape[1]
n_eq_trials_s2 = eq_trials_s2.shape[1]

for i_sub in range(nsubsamplings):
    print(i_sub)
    idx_corr_s1 = eq_trials_s1[0,:,i_sub]
    idx_err_s1  = eq_trials_s1[1,:,i_sub]
    idx_corr_s2 = eq_trials_s2[0,:,i_sub]
    idx_err_s2  = eq_trials_s2[1,:,i_sub]

    # correct s1
    for i_div in range(n_time_divisions-len(lags)+1):
        for i_lag in range(n_lags):
            T1 = idx_corr_s1*n_time_divisions + i_div
            T2 = idx_corr_s1*n_time_divisions + i_lag
            s1 = rates_i[0,T1]
            s2 = rates_i[1,T2]
            correlations_corr_s1[i_sub, i_div, i_lag] = pearsonr(s1,s2)[0]

    #correct s2
    for i_div in range(n_time_divisions-len(lags)+1):
        for i_lag in range(n_lags):
            T1 = idx_corr_s2*n_time_divisions + i_div
            T2 = idx_corr_s2*n_time_divisions + i_lag
            s1 = rates_i[0,T1]
            s2 = rates_i[1,T2]
            correlations_corr_s2[i_sub, i_div, i_lag] = pearsonr(s1,s2)[0]

    #error s1
    for i_div in range(n_time_divisions-len(lags)+1):
        for i_lag in range(n_lags):
            T1 = idx_err_s1*n_time_divisions + i_div
            T2 = idx_err_s1*n_time_divisions + i_lag
            s1 = rates_i[0,T1]
            s2 = rates_i[1,T2]
            correlations_err_s1[i_sub, i_div, i_lag] = pearsonr(s1,s2)[0]
    #error s2
    for i_div in range(n_time_divisions-len(lags)+1):
        for i_lag in range(n_lags):
            T1 = idx_err_s2*n_time_divisions + i_div
            T2 = idx_err_s2*n_time_divisions + i_lag
            s1 = rates_i[0,T1]
            s2 = rates_i[1,T2]
            correlations_err_s2[i_sub, i_div, i_lag] = pearsonr(s1,s2)[0]

save(base_directory + '/data/data_pearon_correlations/correlations_corr_s1.npy', correlations_corr_s1)
save(base_directory + '/data/data_pearon_correlations/correlations_corr_s2.npy', correlations_corr_s2)
save(base_directory + '/data/data_pearon_correlations/correlations_err_s1.npy', correlations_err_s1)
save(base_directory + '/data/data_pearon_correlations/correlations_err_s2.npy', correlations_err_s2)