# BiophysicalReadout
Supplementary code for the paper "Correlations enhance the behavioral readout of neural population activity in association cortex" by M. Valente, G. Pica, G. Bondanelli, M. Moroni, C. A. Runyan, A. S. Morcos, C. D. Harvey, S. Panzeri, (2021).


The repository contains the code to reproduce Figure 6 d-g, i-l, m-o. It contains:

- a folder modules_/
- a folder data_IO_info/across-pool for generating panels d-g
- a folder data_IO_info/across-time for generating panels i-l
- a folder Pearon_correlations/ for generating panels m,n
- a folder Readout_fitting/ for generating panel o
- a folder data/ that contains the data 

#### Panels d-g:
1. Generate the dataset by running data_IO_info/across-pool/compute_IOinfo_across_pool.py
2. Plot results using data_IO_info/across-pool/plot_IOinfo_across_pool.py

#### Panels i-l:
Same as above but running the scripts contained in data_IO_info/across-time/

#### Panels m-n:
1. Generate the dataset by running Pearson_correlations/generate_dataset.py
2. Compute Pearson correlations in correct vs. error trial using Pearson_correlations/compute_pearsoncorr.py
3. Plot results using Pearson_correlations/plot_pearsoncorr.py

#### Panel o:

1. Generate the dataset by running Readout_fitting/generate_dataset_for readouts.py
2. Compute Pearson correlations in correct vs. error trial using Readout_fitting/fit_readouts.py
3. Plot results using Readout_fitting/plot_data_readouts.py

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


