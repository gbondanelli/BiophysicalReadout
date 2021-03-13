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
from scipy.stats import ttest_ind    

##

correlations_corr_s1 =load(base_directory + '/data/data_pearson_correlations/correlations_corr_s1.npy')
correlations_corr_s2 =load(base_directory + '/data/data_pearson_correlations/correlations_corr_s2.npy')
correlations_err_s1 =load(base_directory + '/data/data_pearson_correlations/correlations_err_s1.npy')
correlations_err_s2 =load(base_directory + '/data/data_pearson_correlations/correlations_err_s2.npy')

n_div = correlations_corr_s1.shape[2]

# average over time lags

correlations_corr_s1_ = nanmean(correlations_corr_s1,1)
correlations_corr_s2_ = nanmean(correlations_corr_s2,1)
correlations_err_s1_ = nanmean(correlations_err_s1,1)
correlations_err_s2_ = nanmean(correlations_err_s2,1)

##
n =2
# average over stimuli
vcorr = (correlations_corr_s1_[:,:n]+correlations_corr_s2_[:,:n])/2
verr = (correlations_err_s1_[:,:n]+correlations_err_s2_[:,:n])/2

# average over subsamplings
mcorr = nanmean(vcorr,0)
merr = nanmean(verr,0)
scorr = nanstd(vcorr,0)
serr = nanstd(verr,0)

pvalues = ttest_ind(vcorr,verr)[1]

tau_sampling = 1
bin = int(1000*tau_sampling/n_div)
fig,ax = subplots(figsize=(1.7,1.9))

pars={'ls':'.-','lw':1.5,'markersize':5, 'err':'sd'}
my_fill_between(bin*arange(n), [vcorr,verr], ['#4181a6','#d64f7e'], ['#c7dcff','#ffabb3'], ['Correct','Error'],  **pars)

ylimmax = ax.get_ylim()[1]
for i in range(len(pvalues)):
    if mcorr[i]>merr[i] and pvalues[i]<0.001:
        ax.plot([bin*arange(n)[i]],[ylimmax],'x',color='#b5b0b1',markersize=3)
# ax.spines['left'].set_bounds(0, .1)
ax.spines['bottom'].set_bounds(0, 400)
xticks([0,200,400])
ax.set_xlabel('Lag (ms)')
ax.set_ylabel('Pearson corr.')
# legend(frameon=0,loc=3)
tight_layout()
# savefig('../../Valente_NN/figs/correlations_correct_error.pdf',transparent=True)

##
i_lag = 1
data = [vcorr[:,i_lag],verr[:,i_lag]]
labels = ['Correct','Error']
facecolor = ['#4181a6','#d64f7e']
colorwhisk = ['#4181a6','#4181a6','#d64f7e','#d64f7e']
colorcaps = colorwhisk
colorfliers = 'grey'
width = .8
ax = my_boxplot((1.2,1.9), data, labels, 40, facecolor, colorwhisk, colorcaps, colorfliers, width)
# y1 = 0.06
# y2=0.08
# ax.spines['left'].set_bounds(y1, y2)
# ax.set_yticks([y1, y2])
ax.spines['bottom'].set_bounds(1, 2)         
tight_layout()
print('p = ', ttest_ind(vcorr[:,i_lag],verr[:,i_lag])[1], 't = ', ttest_ind(vcorr[:,i_lag],verr[:,i_lag])[0])


