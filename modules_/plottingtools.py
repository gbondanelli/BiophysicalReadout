from matplotlib.pyplot import *
from matplotlib.colors import LinearSegmentedColormap
from numpy import *
import rc_parameters

def my_fill_between(x, F, col, colfill, labels,  **pars):
    ls = pars['ls']
    lw = pars['lw']
    ms = pars['markersize']
    m  = [mean(f,0) for f in F]
    s  = [std(f,0) for f in F]
    ntrials = F[0].shape[0]
    a = sqrt(ntrials) if pars['err'] == 'se' else 1
    for i in range(len(m)):
        plot(x, m[i], ls, lw = lw, color = col[i], label = labels[i],markersize=ms)
        fill_between(x, m[i]-s[i]/a, m[i]+s[i]/a,color = colfill[i])

def my_boxplot(figsize, data, labels, rotation):
    fig = figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    col = 'k'
    flierprops      = dict(marker = 'o', markerfacecolor = col, markersize = 4, markeredgewidth = .3, linestyle = 'none')
    whiskerprops    = dict(linewidth=1., color=col)
    boxprops        = dict(linewidth = 1., color = col)
    capprops        = dict(linewidth = 1., color = col)
    bp = ax.boxplot(data, flierprops = flierprops, widths = .35, patch_artist = True,
                    whiskerprops = whiskerprops, boxprops = boxprops, capprops = capprops)
    for box in bp['boxes']:
        box.set(facecolor='w', ) 
    for median in bp['medians']:
        median.set(color=col, linewidth=.8)
    xticks(arange(1,len(labels)+1,1), labels, rotation=rotation)
    tight_layout()


def define_colormap(colors, N):
    cm = LinearSegmentedColormap.from_list('new_cm', colors, N)
    return cm

##

