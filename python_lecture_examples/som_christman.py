#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#.............................................
# INTITIAL SETUP
#.............................................

#.............................................
# IMPORT STATEMENTS
#.............................................
#.............................................
# IMPORT STATEMENTS
#.............................................
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import Counter
#import matplotlib.mlab as mlab
#import scipy.signal as sig
#from scipy import stats
#from scipy import interpolate
#import numpy.ma as ma
#import csv
#from numpy import genfromtxt
#from mpl_toolkits.basemap import Basemap

from sompy.sompy import SOMFactory

#from sompy.visualization.mapview import View2D
#from sompy.visualization.bmuhits import BmuHitsView
#from sompy.visualization.hitmap import HitMapView


import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

import textbook_params as tbk
reload(tbk)

#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt

savefigpath = '../figures/'


#%% START CODE

warnings.warn('Cannot use day of year or hour of day as SOM takes the average of the values and these are cyclic')

filename = '/Users/eabarnes/GoogleDrive/WORK/AUTHORED_MANUSCRIPTS/INPREP/Hartmann_Barnes_2018/python/chapter5/christman_2016.csv'
data_input = np.genfromtxt(filename, delimiter = ',')
#grab_indices = [0,1,2,3,4,5,9,10,11]
grab_indices = [2,3,5,9,10,11]
data = data_input[:,grab_indices]

names_input = ['date','time','temp (F)', 'RH (%)', 'DewPt (F)','Wind (mph)', 'Dir (deg.)', 'Gust (mph)', 'Gust Dir (deg.)','Pres (mb)', 'Solar (W/m^2)','Precip (in)']
names = [names_input[i] for i in grab_indices]

# convert precip inches to mm
data[:,[i for i, s in enumerate(names) if 'Precip' in s]] = data[:,[i for i, s in enumerate(names) if 'Precip' in s]]*25.4
names[names.index('Precip (in)')] = 'Precip (mm)'
         
# convert time into hour of day
#data[:,[i for i, s in enumerate(names) if 'time' in s]] = np.round(data[:,[i for i, s in enumerate(names) if 'time' in s]]*24.)     
#names[names.index('time')] = 'hour of day'  
      
# convert to day of year
#data[:,[i for i, s in enumerate(names) if 'date' in s]] = data[:,[i for i, s in enumerate(names) if 'date' in s]] - data[0,[i for i, s in enumerate(names) if 'date' in s]] + 1.     
#names[names.index('date')] = 'day of year'  
      
#%%
sm = SOMFactory().build(data, normalization = 'var', initialization='random', component_names=names, neighborhood='gaussian', mapsize = (20,20))
sm.train(n_job=1, verbose='debug', train_rough_len=2, train_finetune_len=10)

topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)

#%%

#view2D  = View2D(10,10,"rand data",text_size=10)
#view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

#vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
#vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

#%% SOM structure maps
my_cmap = plt.cm.get_cmap('YlGnBu')

codebook = sm._normalizer.denormalize_by(sm.data_raw, sm.codebook.matrix)

gf.cfig(1, fig_width = 65, fig_height = 80)


for ind in range(0,np.size(codebook,axis=1)):
    weights = codebook[:,ind]
    
    xplot = codebook[:, ind].reshape(sm.codebook.mapsize[0],sm.codebook.mapsize[1])
    
    ax = plt.subplot(3, 2, ind+1)
    
    plt.axis([0, sm.codebook.mapsize[1], 0, sm.codebook.mapsize[0]])
    if(names[ind] in 'Precip (mm)'):
        pl = plt.pcolor(xplot, cmap = my_cmap, norm=colors.LogNorm(vmin=xplot.min(), vmax=xplot.max()))
    else:
        pl = plt.pcolor(xplot, cmap = my_cmap)
        
#    plt.axis([0, sm.codebook.mapsize[1], 0, som.codebook.mapsize[0]])
#    plt.title(names[axis_num - 1])
    plt.yticks([])
    plt.xticks([])
    cbar = plt.colorbar()
    
    cbar.ax.tick_params(labelsize=10) 
    
    plt.title(names[ind], fontsize = 12)

plt.suptitle('SOM nodes/patterns',fontweight='bold', fontsize = 16)

gf.save_fig_file(savefigpath + 'ch5_' + 'SOM_christman_codebook', dpi = tbk.dpi_save)
    
plt.show()


#%% Hits map

def set_labels(cents, ax, labels, fontsize):
    for i, txt in enumerate(labels):
        plt.annotate(txt, (cents[i, 1] + 0.5, cents[i, 0] + 0.5), va="center", ha="center", size=fontsize, color = 'darkorange', fontweight = 'bold')


my_cmap = plt.cm.get_cmap('BuPu')
        
fig = gf.cfig(2, fig_width = 70/1.5, fig_height = 60/1.5)
ax = plt.gca()

bmu_mapping = sm._bmu[0,:]
counts = Counter(bmu_mapping)
counts = [counts.get(x, 0) for x in range(sm.codebook.mapsize[0] * sm.codebook.mapsize[1])]
xplot = np.array(counts).reshape(sm.codebook.mapsize[0],sm.codebook.mapsize[1])

msz = sm.codebook.mapsize
cents = sm.bmu_ind_to_xy(np.arange(0, msz[0] * msz[1]))

set_labels(cents, ax, counts, fontsize=10)

pl = plt.pcolor(xplot, cmap=my_cmap)

plt.axis([0, sm.codebook.mapsize[1], 0, sm.codebook.mapsize[0]])
ax.set_yticklabels([])
ax.set_xticklabels([])

plt.title('SOM Frequency')

gf.save_fig_file(savefigpath + 'ch5_' + 'SOM_christman_hitmap', dpi = tbk.dpi_save)

plt.show()














