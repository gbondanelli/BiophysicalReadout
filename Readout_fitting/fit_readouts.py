import sys
sys.path.insert(0,'../encoding_decoding')
from numpy import *
from encdec import *
from stattools import *
from information import *
from plottingtools import *
from managingtools import *
from lif import *
from matplotlib.pyplot import *
import rc_parameters
from sklearn.metrics import log_loss
from scipy.special import *

import warnings
warnings.simplefilter("ignore")

S = load('./datasets/spatial/dataset_for_readout/S_eq_choices_v2.npy')
Sdec = load('./datasets/spatial/dataset_for_readout/Sdec_eq_choices_v2.npy')
cons = load('./datasets/spatial/dataset_for_readout/cons_eq_choices_v2.npy')
choices = load('./datasets/spatial/dataset_for_readout/choices_eq_choices_v2.npy')

nsubsamplings = S.shape[0]
A = S[0,:,:,:].ravel()
where_not_nan = ~isnan(A)
nsamples = len(A[where_not_nan])

frac_dev_expl_full = empty(nsubsamplings)
frac_dev_expl_nocons = empty(nsubsamplings)
frac_dev_expl_noneural = empty(nsubsamplings)

coef_full = empty((nsubsamplings,5))
coef_nocons = empty((nsubsamplings,5))
coef_noneural = empty((nsubsamplings,5))

for i_sub in range(nsubsamplings):
    print(i_sub)
    # full
    C = choices[i_sub,:,:,:].ravel()[where_not_nan]
    regressors = empty((4,nsamples))
    regressors[0,:] = zeros(S[i_sub,:,:,:].ravel()[where_not_nan].shape)
    # regressors[0,:] = S[i_sub,:,:,:].ravel()[where_not_nan]
    regressors[1,:] = Sdec[i_sub,:,:,:].ravel()[where_not_nan]
    regressors[2,:] = (Sdec[i_sub,:,:,:].ravel()[where_not_nan]+1)*cons[i_sub,:,:,:].ravel()[where_not_nan]/2
    regressors[3,:] = (Sdec[i_sub,:,:,:].ravel()[where_not_nan]-1)*cons[i_sub,:,:,:].ravel()[where_not_nan]/2
    model, betas, bias, _, _ = logistic_regression_cv(C, regressors, 3)
    R2 = frac_dev_expl(C, regressors.T, model)
    frac_dev_expl_full[i_sub] = R2
    coef_full[i_sub, 0] = bias[0]
    coef_full[i_sub, 1:] = betas

    #no cons
    regressors[2,:] = regressors[2,random.permutation(arange(len(regressors[2,:])))]
    regressors[3,:] = regressors[3,random.permutation(arange(len(regressors[3,:])))]
    model, betas, bias, _, _ = logistic_regression_cv(C, regressors, 3)
    R2 = frac_dev_expl(C, regressors.T, model)
    frac_dev_expl_nocons[i_sub] = R2
    coef_nocons[i_sub, 0] = bias[0]
    coef_nocons[i_sub, 1:] = betas

    #no neural
    regressors[1,:] = regressors[1,random.permutation(arange(len(regressors[1,:])))]
    model, betas, bias, _, _ = logistic_regression_cv(C, regressors, 3)
    R2 = frac_dev_expl(C, regressors.T, model)
    frac_dev_expl_noneural[i_sub] = R2
    coef_noneural[i_sub, 0] = bias[0]
    coef_noneural[i_sub, 1:] = betas


save('./results/logistic_readout/frac_dev_expl_full_eq_choices.npy',frac_dev_expl_full)
save('./results/logistic_readout/frac_dev_expl_nocons_eq_choices.npy',frac_dev_expl_nocons)
save('./results/logistic_readout/frac_dev_expl_noneural_eq_choices.npy',frac_dev_expl_noneural)
save('./results/logistic_readout/coef_full_eq_choices.npy',coef_full)
save('./results/logistic_readout/coef_nocons_eq_choices.npy',coef_nocons)
save('./results/logistic_readout/coef_noneural_eq_choices.npy',coef_noneural)










