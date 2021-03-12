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

##
[tau, tau_sampling, Rin, alpha, tau_lowpass, outputrate_stim1, stdout_stim1, CVrateout_stim1, outsvmacc, insvmacc] \
= open_pickle('../model_info_correlations/results/data_temporal_corr.pkl')

n_trials, n_in, n_tau, n_corr, n_tausampling, n_taulp = insvmacc.shape

##
InputInfo = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))
OutputInfo = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))
OutputRateGain = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))
OutputVarGain = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))
NormalizedCV = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))
TransmissionGain = empty((n_trials, n_in,n_tau,n_corr,n_tausampling, n_taulp))

for i_lp in range(n_taulp):
    for i_s in range(n_tausampling):
        for i_in in range(n_in):
            for i_tau in range(n_tau):
                for i_c in range(n_corr):

                    InputInfo[:, i_in, i_tau, i_c, i_s, i_lp] = 100*(insvmacc[:, i_in, i_tau, i_c, i_s,i_lp])

                    OutputInfo[:, i_in, i_tau, i_c, i_s, i_lp] = 100 * (outsvmacc[:, i_in, i_tau, i_c, i_s, i_lp])

                    OutputRateGain[:, i_in, i_tau, i_c, i_s, i_lp] = 100*(outputrate_stim1[:, i_in, i_tau, i_c, i_s, i_lp] / mean(outputrate_stim1[:, i_in, i_tau, i_c, i_s, 0])-1)

                    OutputVarGain[:, i_in, i_tau, i_c, i_s, i_lp] = 100*(stdout_stim1[:, i_in, i_tau, i_c, i_s, i_lp] / mean(
                        stdout_stim1[:, i_in, i_tau, i_c, i_s, 0])-1)

                    NormalizedCV[:,i_in, i_tau, i_c, i_s, i_lp] = CVrateout_stim1[:, i_in, i_tau, i_c, i_s, i_lp] / (CVrateout_stim1[:, i_in, i_tau, i_c, i_s, 0])

                    # TransmissionGain[:, i_in, i_tau, i_c, i_s, i_lp] = 100*(1-abs(outsvmacc[:, i_in, i_tau, i_c, i_s, i_lp] - insvmacc[:, i_in, i_tau, i_c, i_s, i_lp])\
                    #                                   / mean(abs(outsvmacc[:, i_in, i_tau, i_c, i_s, 0] - insvmacc[:, i_in, i_tau, i_c, i_s, 0])))

                    TransmissionGain[:, i_in, i_tau, i_c, i_s, i_lp] = 100 * (abs(
                        (outsvmacc[:, i_in, i_tau, i_c, i_s, i_lp]-.5)/(insvmacc[:, i_in, i_tau, i_c, i_s, i_lp]-.5)) \
                                                                              / mean(
                                abs((outsvmacc[:, i_in, i_tau, i_c, i_s, 0]-.5)/(insvmacc[:, i_in, i_tau, i_c, i_s, 0]-.5)))-1)

##
i_in = 0
i_tau = 0
i_s = 0
i_c = 2
# Input info

figure(figsize=(1.6,1.6))
col = ['#696f7a']; colfill = ['#c9ccd1']
labels = ['']
pars = {'ls':'-', 'lw':1.5, 'markersize':0, 'err':'se'}
F = [(InputInfo[:,i_in,i_tau,i_c,i_s,:])]
x = tau_lowpass
my_fill_between(x, F, col, colfill, labels,  **pars)
ylabel('Input decoding acc. (\%)', fontsize=7)
xlabel(r'Temp. corr. $\tau_C$', fontsize=7)
xlim([0.005,0.1])
xticks([0.005,0.05, 0.1],[r'$5$',r'$50$', r'$100$'])
# ylim([.559,.57])
# yticks([.56,.565,.57],[r'$56.0$',r'$56.5$',r'$57.0$'])
tight_layout()


#Output info
figure(figsize=(1.6,1.6))
col = ['#696f7a']; colfill = ['#c9ccd1']
labels = ['']
pars = {'ls':'-', 'lw':1.5, 'markersize':0, 'err':'se'}
F = [(OutputInfo[:,i_in,i_tau,i_c,i_s,:])]
x = tau_lowpass
my_fill_between(x, F, col, colfill, labels,  **pars)
ylabel('Output decoding acc. (\%)', fontsize=7)
xlabel(r'Temp. corr. $\tau_C$', fontsize=7)
xlim([0.005,0.1])
xticks([0.005,0.05, 0.1],[r'$5$',r'$50$', r'$100$'])
# ylim([.559,.57])
# yticks([.56,.565,.57],[r'$56.0$',r'$56.5$',r'$57.0$'])
tight_layout()

#Output Rate Gain

# Output Var Gain
figure(figsize=(1.2,1.6))
col = ['#4f9bd1','#fa952a']; colfill = ['#c8d3e6','#fcce9d']
labels = ['','']
pars = {'ls':'-', 'lw':1.5, 'markersize':0, 'err':'se'}
F = [(OutputRateGain[:,i_in,i_tau,i_c,i_s,:]),(OutputVarGain[:,i_in,i_tau,i_c,i_s,:])]
x = tau_lowpass
my_fill_between(x, F, col, colfill, labels,  **pars)
ylabel('Output gain (\%)', fontsize=7)
xlabel(r'Temp. corr. $\tau_C$', fontsize=7)
xlim([0.005,0.1])
xticks([0.005,0.05, 0.1],[r'$5$',r'$50$', r'$100$'])
ylim(0,30)
tight_layout()


# Normalized CV
figure(figsize=(1.2,1.6))
col = ['#80c2d1']; colfill = ['#d1eaf0']
labels = ['']
pars = {'ls':'-', 'lw':1.5, 'markersize':0, 'err':'se'}
F = [(NormalizedCV[:,i_in,i_tau,i_c,i_s,:])]
x = tau_lowpass
my_fill_between(x, F, col, colfill, labels,  **pars)
ylabel('Coeff. of variation (norm.)', fontsize=7)
xlabel(r'Temp. corr. $\tau_C$', fontsize=7)
xlim([0.005,0.1])
xticks([0.005,0.05, 0.1],[r'$5$',r'$50$', r'$100$'])
tight_layout()


# Transmission Gain
figure(figsize=(1.6,1.6))
col = ['#ad2f51']; colfill = ['#ffb3c7']
labels = ['']
pars = {'ls':'-', 'lw':1.5, 'markersize':0, 'err':'se'}
F = [(TransmissionGain[:,i_in,i_tau,i_c,i_s,:])]
x = tau_lowpass
my_fill_between(x, F, col, colfill, labels,  **pars)
ylabel('Transmission gain (\%)', fontsize=7)
xlabel(r'Temp. corr. $\tau_C$(ms)', fontsize=7)
xlim([0.005,0.1])
xticks([0.005,0.05, 0.1],[r'$5$',r'$50$', r'$100$'])
tight_layout()

