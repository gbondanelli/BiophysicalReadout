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
N       = 2
T       = 50000
nsteps  = 50000000
t       = linspace(0, T, nsteps)
dt = t[1] - t[0]
signal_axis = array([1., 1.])/sqrt(2)
ntrials = 20
nsamples = 10000
nsamples_dec = 1000
SNR = 0.2
dm = 1.

Vr = -70.0  
Vth = -50.0
w = array([15.,15.])
w_mf = array([0.,0.])
mu0 = 0.0
sigma_v = 0.

tau_lowpass = linspace(0.005, 0.1, 10)
tau = [0.005]
tau_sampling = [1]
Rin = [2,4,6,8,10]
alpha = [0,.5,.9]

##
inputrate_n1_stim1  = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
varin_n1_stim1      = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
CVratein_n1_stim1   = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
outputrate_stim1    = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
outputrate_stim2    = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
stdout_stim1        = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
stdout_stim2        = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
CVrateout_stim1     = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
CVrateout_stim2     = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
outputMI            = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
inputMI             = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
outsvmacc           = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))
insvmacc            = empty((ntrials,len(Rin),len(tau),len(alpha),len(tau_sampling),len(tau_lowpass)))

for i_lp in range(len(tau_lowpass)):
    for i_s in range(len(tau_sampling)):
        for i_c in range(len(alpha)):
            print('Finished {:.1f}% of the analyses'.format(100.*(i_s * len(alpha) + i_c) / len(alpha) / len(tau_sampling)))
            for i_tau in range(len(tau)):
                for i_in in range(len(Rin)):

                    muS = array([Rin[i_in]-dm/2., Rin[i_in]+dm/2.])
                    sigma = dm / SNR

                    r1, r2 = generate_correlated_rates(signal_axis, muS, sigma, alpha[i_c], t)
                    r1, r2 = [low_pass_filter(r, tau_lowpass[i_lp], t) for r in [r1, r2]]
                    sti1, sti2 = [generate_poisson(r, dt, ntrials) for r in [r1, r2]]
                    input = [sti1, sti2]
                    input_mf = r1
                    [sto1, _], [sto2, _] = [simulate_lif(Vr, Vth, tau[i_tau], t, w, w_mf, inp, input_mf, mu0, sigma_v) for inp in input]

                    rates_i_stim1 = compute_rate_from_spike_trains2(sti1, tau_sampling[i_s], t)    # rates has shape N x nsamples x ntrials
                    rates_i_stim2 = compute_rate_from_spike_trains2(sti2, tau_sampling[i_s], t)
                    rates_o_stim1 = compute_rate_from_spike_trains2(sto1[None, :, :], tau_sampling[i_s], t)
                    rates_o_stim2 = compute_rate_from_spike_trains2(sto2[None, :, :], tau_sampling[i_s], t)

                    m1 = mean(rates_o_stim1[0, :, :], 0)
                    s1 = std(rates_o_stim1[0, :, :], 0)
                    
                    outputrate_stim1[:, i_in, i_tau, i_c, i_s, i_lp]  = m1
                    stdout_stim1[:, i_in, i_tau, i_c, i_s, i_lp]      = s1
                    CVrateout_stim1[:, i_in, i_tau, i_c, i_s, i_lp]   = s1/m1

                    nsamples = rates_i_stim1.shape[1]

                    insvmacc[:, i_in, i_tau, i_c, i_s, i_lp],_,_ = \
                        decode_from_spikes(sti1, sti2, tau_sampling[i_s], t, nsamples, 'linsvm', 0.05)

                    outsvmacc[:, i_in, i_tau, i_c, i_s, i_lp],_,_ = \
                        decode_from_spikes(sto1[None,:,:], sto2[None,:,:], tau_sampling[i_s], t, nsamples, 'linsvm')


objlist = [tau, tau_sampling, Rin, alpha, tau_lowpass, outputrate_stim1, stdout_stim1, CVrateout_stim1, outsvmacc, insvmacc] 
save_pickle('./results/data_temporal_corr.pkl',objlist)
