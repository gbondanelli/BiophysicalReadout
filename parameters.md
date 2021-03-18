#### Parameters used in simulations:

N       = 2 # **number of input neurons (pools)**\
T       = 100000 # **total simulation time (seconds)**\
nsteps  = T * 1000 # **number of time steps (1ms time step)**\
signal_axis = array([1., 1.])/sqrt(2) # **axis defined by the difference in mean activity across stimuli**\
ntrials = 1 # **number of independent simulations**\
SNR = 0.2 # **signal to noise ratio** \
dm = 1. # **difference in mean activity**

Vr = -70.0 # **resting potetntial**\
Vth = -50.0 # **voltage threshold for spiking**\
w = array([15.,15.]) # **EPSP strength**\
w_mf = array([0.,0.]) \
mu0 = 0.0 # **constant external input**\
sigma_v = 0.

tau_lowpass = [0.1,0.5] # **strength of temporal correlations**\
tau = [0.005, 0.007] # **membrane time constant**\
tau_sampling = [1.] # **length of single trial; for IO_info analyses, quantities are computed using spike counts computed in a time window of tau_smapling**\
Rin = [2,6] # **mean input firing rate** \
alpha = [0.9] # **strength of spatial correlations**
