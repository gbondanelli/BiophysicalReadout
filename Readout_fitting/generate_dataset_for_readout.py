import sys
base_directory = '/Users/giuliobondanelli/OneDrive - Fondazione Istituto Italiano Tecnologia/Code/code_Valente21_to_share'
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
= open_pickle(base_directory + '/data/data_pearson_correlations/dataset_alpha.pkl')

STI1 = load(base_directory + '/data/data_pearson_correlations/sti1.npy',allow_pickle = True)
STI2 = load(base_directory + '/data/data_pearson_correlations/sti2.npy',allow_pickle = True)
STIM = load(base_directory + '/data/data_pearson_correlations/Stim.npy',allow_pickle = True)
PRED_STIM = load(base_directory + '/data/data_pearson_correlations/Pred_Stim.npy',allow_pickle = True)

##
n_trials, n_in, n_tau, n_corr, n_tausampling, n_taulp = insvmacc.shape
t = linspace(0,100000,100000000+1)

i_in    = 0
i_tau   = 0
i_s     = 0
dt      = t[1] - t[0]
n_time_divisions = 2
lags    = arange(2)
n_lags  = len(lags)

# correlated data
i_c = 0
i_lp = 0
predicted_stim = PRED_STIM[i_in, i_tau, i_c, i_s, i_lp]
stimuli = STIM[i_in, i_tau, i_c, i_s, i_lp]
frac=0.5
n_subsamplings = 20
eq_trials = equalize_trials(predicted_stim == 1, frac, n_subsamplings)

ntrials_sub = 2*eq_trials.shape[1]

S = nan*ones((n_subsamplings, n_time_divisions, n_lags, ntrials_sub))
Sdec = nan*ones((n_subsamplings, n_time_divisions, n_lags, ntrials_sub))
cons = nan*ones((n_subsamplings, n_time_divisions, n_lags, ntrials_sub))
choices = nan*ones((n_subsamplings, n_time_divisions, n_lags, ntrials_sub))

i_trial = 0

sti1 = STI1[i_in, i_tau, i_c, i_s, i_lp]
sti2 = STI2[i_in, i_tau, i_c, i_s, i_lp]

rates_i_s1, rates_i_s2 = [compute_rate_from_spike_trains2(st, tau_sampling[i_s]/n_time_divisions, t) for st in [sti1,sti2]]
rates_i = hstack((rates_i_s1[:,:,i_trial],rates_i_s2[:,:,i_trial]))
stimuli_fine = hstack((-ones(int(rates_i.shape[1]/2)),+ones(int(rates_i.shape[1]/2))))

for i_sub in range(n_subsamplings):
    for t1 in range(n_time_divisions):
        print(i_sub,t1)
        for t2 in range(n_lags):
            if t1+t2>=n_time_divisions:
                S[i_sub,t1,t2,:] = nan*ones(ntrials_sub)
                Sdec[i_sub,t1,t2,:] = nan*ones(ntrials_sub)
                cons[i_sub,t1,t2,:] = nan*ones(ntrials_sub)
                choices[i_sub,t1,t2,:] = nan*ones(ntrials_sub)
            else:
                idx = hstack((eq_trials[1,:,i_sub],eq_trials[0,:,i_sub]))
                T1 = n_time_divisions*idx + t1
                T2 = n_time_divisions*idx + t2
                R = concatenate((rates_i[:,T1],rates_i[:,T2]),axis=0)
                acc1,_,Sp = decode_from_rates(R[:,:,None], stimuli[idx], 'linsvm')
                acc2,_,Sp1 = decode_from_rates(rates_i[:,T1][:,:,None], stimuli[idx], 'linsvm')
                acc3,_,Sp2 = decode_from_rates(rates_i[:,T2][:,:,None], stimuli[idx], 'linsvm')
                print(acc1,acc2,acc3)
                S[i_sub,t1,t2,:] = stimuli[idx]
                Sdec[i_sub,t1,t2,:] = Sp[:,i_trial]
                cons[i_sub,t1,t2,:] = (Sp1[:,i_trial] == Sp2[:,i_trial]).astype(float)
                choices[i_sub,t1,t2,:] = predicted_stim[idx,i_trial]
##
save(base_directory + '/data/data_readouts/S_eq_choices.npy', S)
save(base_directory + '/data/data_readouts/Sdec_eq_choices.npy', Sdec)
save(base_directory + '/data/data_readouts/cons_eq_choices.npy', cons)
save(base_directory + '/data/data_readouts/choices_eq_choices.npy', choices)

##

