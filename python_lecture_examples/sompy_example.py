#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RESOURCE/EXAMPLE from here
#https://github.com/sevamoo/SOMPY/blob/master/sompy/examples/California%20Housing.ipynb
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import scipy.signal as sig
#from scipy import stats
#from scipy import interpolate
#import numpy.ma as ma
#import csv
#from numpy import genfromtxt
#from mpl_toolkits.basemap import Basemap

import general_functions as gf
reload(gf)

#%load_ext autoreload
#%autoreload 2
from sompy.sompy import SOMFactory

from sklearn.datasets import fetch_california_housing
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView


#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt



#%% START CODE


data = fetch_california_housing()
descr = data.DESCR
names = fetch_california_housing().feature_names+["HouseValue"]

data = np.column_stack([data.data, data.target])
print descr
print "FEATURES: ", ", ".join(names)

#%%

sm = SOMFactory().build(data, normalization = 'var', initialization='random', component_names=names, neighborhood='gaussian')
sm.train(n_job=1, verbose='debug', train_rough_len=2, train_finetune_len=5)

#%%
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print "Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error)

#%%
view2D  = View2D(10,10,"rand data",text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

#%%
vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

#%% K-means

sm.cluster(4)
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)
