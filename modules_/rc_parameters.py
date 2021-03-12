from matplotlib.pyplot import*

rc('text', usetex=True)
rcParams['figure.figsize'] = [3, 2]
rcParams['lines.linewidth'] = 2.
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['axes.spines.top']= False
rcParams['axes.spines.right']= False
rcParams['xtick.major.size']=2
rcParams['xtick.major.width']=.7
rcParams['ytick.major.size']=2
rcParams['ytick.major.width']=.7
rcParams['axes.linewidth']=.7
rcParams['xtick.labelsize']=8
rcParams['ytick.labelsize']=8
rcParams['axes.labelsize']=9
rcParams['axes.prop_cycle']=cycler(
'color',
['#f16a6a','#6ab0f1','#79c94a','#be80e0','#5fdbdd','#eedd6d','#f5c5f7','#ef2576','#b3c7df','#8e582e']
                                    )
