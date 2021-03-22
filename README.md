# BiophysicalReadout
Supplementary code for the paper "Correlations enhance the behavioral readout of neural population activity in association cortex" by M. Valente, G. Pica, G. Bondanelli, M. Moroni, C. A. Runyan, A. S. Morcos, C. D. Harvey, S. Panzeri, Nature Neuroscience (2021).


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


