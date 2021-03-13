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
from scipy.stats import ttest_ind
from brokenaxes import brokenaxes

##
# frac_dev_expl_full = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/frac_dev_expl_full_eq_choices.npy')
# frac_dev_expl_nocons = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/frac_dev_expl_nocons_eq_choices.npy')
# frac_dev_expl_noneural = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/frac_dev_expl_noneural_eq_choices.npy')
# coef_full = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/coef_full_eq_choices.npy')
# coef_nocons = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/coef_nocons_eq_choices.npy')
# coef_noneural = load('../model_info_correlations/results/logistic_readout/frac05_Rin6_5lags_onlyneural/coef_noneural_eq_choices.npy')

# frac_dev_expl_full = load('../model_info_correlations/results/logistic_readout/new/frac_dev_expl_full_eq_choices.npy')
# frac_dev_expl_nocons = load('../model_info_correlations/results/logistic_readout/new/frac_dev_expl_nocons_eq_choices.npy')
# frac_dev_expl_noneural = load('../model_info_correlations/results/logistic_readout/new/frac_dev_expl_noneural_eq_choices.npy')
# coef_full = load('../model_info_correlations/results/logistic_readout/new/coef_full_eq_choices.npy')
# coef_nocons = load('../model_info_correlations/results/logistic_readout/new/coef_nocons_eq_choices.npy')
# coef_noneural = load('../model_info_correlations/results/logistic_readout/new/coef_noneural_eq_choices.npy')

frac_dev_expl_full = load('../model_info_correlations/results/logistic_readout/submission_march/frac_dev_expl_full_eq_choices.npy')
frac_dev_expl_nocons = load('../model_info_correlations/results/logistic_readout/submission_march/frac_dev_expl_nocons_eq_choices.npy')
frac_dev_expl_noneural = load('../model_info_correlations/results/logistic_readout/submission_march/frac_dev_expl_noneural_eq_choices.npy')
coef_full = load('../model_info_correlations/results/logistic_readout/submission_march/coef_full_eq_choices.npy')
coef_nocons = load('../model_info_correlations/results/logistic_readout/submission_march/coef_nocons_eq_choices.npy')
coef_noneural = load('../model_info_correlations/results/logistic_readout/submission_march/coef_noneural_eq_choices.npy')
##
p_full_nocons = ttest_ind(frac_dev_expl_nocons,frac_dev_expl_full)[1]
data = [frac_dev_expl_full,frac_dev_expl_nocons]
labels = ['Full','No cons']
facecolor = ['#0a611e', '#76d68c']
colorwhisk = ['#0a611e','#0a611e', '#76d68c', '#76d68c']
colorcaps = colorwhisk
colorfliers = 'grey'
ax=my_boxplot((1.5,2), data, labels, 60, facecolor, colorwhisk, colorcaps, colorfliers, .8)
ylabel('Frac. Dev. Expl.',fontsize=7)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_bounds(0.15,0.17)
yticks([0.15,.17])
tight_layout()
print('p = ', ttest_ind(frac_dev_expl_full,frac_dev_expl_nocons)[1], 't = ', ttest_ind(frac_dev_expl_full,frac_dev_expl_nocons)[0])
print(stats.t(df=20+20-2).ppf((0.05, 0.95)))

##
data = [coef_full[:,2],coef_full[:,3],coef_full[:,4]]
labels = [r'$\beta_{\hat{s}}$',r'$\beta_{i1}$',r'$\beta_{i2}$']
facecolor = ['#c7cdd1', '#c7cdd1','#c7cdd1','#c7cdd1','#c7cdd1']
colorwhisk = 10*['#929fb3']
colorcaps = colorwhisk
colorfliers = 'grey'
ax=my_boxplot((1.7,1.8), data, labels, 0, facecolor,colorwhisk,colorcaps,colorfliers,.7)
plot([0.5,3.5],[0,0],'--',color='#97a0a1',lw=1.)
ylabel('Coeff. value',fontsize=7)
ax.spines['bottom'].set_bounds(1,3)
ax.spines['left'].set_bounds(0,.8)
xlim(0.5,3.5)
tight_layout()
##
data = [coef_nocons[:,0],coef_nocons[:,1],coef_nocons[:,2],coef_nocons[:,3],coef_nocons[:,4]]
labels = [r'$\beta_0$',r'$\beta_s$',r'$\beta_{\hat{s}}$',r'$\beta_{i1}$',r'$\beta_{i2}$']
facecolor = ['#c7cdd1', '#c7cdd1','#c7cdd1','#c7cdd1','#c7cdd1']
my_boxplot((2.1,2), data, labels, 0, facecolor)
plot([1,5],[0,0],'-',color='#97a0a1',lw=1.5)
ylabel('Coeff. value')
tight_layout()

##
data = [coef_noneural[:,0],coef_noneural[:,1],coef_noneural[:,2],coef_noneural[:,3],coef_noneural[:,4]]
labels = [r'$\beta_0$',r'$\beta_s$',r'$\beta_{\hat{s}}$',r'$\beta_{i1}$',r'$\beta_{i2}$']
facecolor = ['#c7cdd1', '#c7cdd1','#c7cdd1','#c7cdd1','#c7cdd1']
my_boxplot((2.1,2), data, labels, 0, facecolor)
plot([1,5],[0,0],'-',color='#97a0a1',lw=1.5)
ylabel('Coeff. value')
tight_layout()

##

