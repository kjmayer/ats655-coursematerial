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


#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ion()

# need to set as Automatic graphics

#plt.ioff()
#%matplotlib inline
#%matplotlib qt



#%% START CODE

xinc = np.arange(-4,4.5,.5)

x = np.random.normal(0,1,size = (25,))
y = np.random.normal(0,1,size = (1000,))

#%%
#=====================================
# drawing random values from normal distribution
#=====================================

gf.cfig(1)
gf.show_plot()

plt.xlabel('value')
plt.ylabel('arbitrary')
plt.yticks([])
plt.title('values drawn from a normal distribution')
plt.xlim(-3,3)

plt.plot(x,np.arange(0,np.size(x)),'.r', markersize = 15)

plt.pause(2)

plt.plot(y,np.arange(0,np.size(y)),'.k', markersize = 15)

plt.pause(2)

for ind, val in enumerate(xinc[1:]):
    plt.plot([val - 0.25, val-0.25], [0, np.size(y)],'-', color = 'gray')

gf.error()


#%% PDF counts

gf.cfig(2)
gf.show_plot()

hx = np.histogram(x,xinc)
hy = np.histogram(y,xinc)

plt.xlabel('value')
plt.ylabel('counts')

plt.bar(hx[1][:-1],hx[0],edgecolor = 'r', color = [], width = .4, linewidth = 2)
plt.pause(2)
plt.bar(hy[1][:-1],hy[0],edgecolor = 'k', color = [], width = .4, linewidth = 2)

#%% PDF normalized

gf.cfig(3)
gf.show_plot()

hx = np.histogram(x,xinc)
hy = np.histogram(y,xinc)

plt.xlabel('value')
plt.ylabel('frequency')

plt.bar(hx[1][:-1],hx[0].astype(float)/np.size(x),edgecolor = 'r', color = [], width = .4, linewidth = 2)
plt.pause(2)
plt.bar(hy[1][:-1],hy[0].astype(float)/np.size(y),edgecolor = 'k', color = [], width = .4, linewidth = 2)

#%% CDF
gf.cfig(4)
gf.show_plot()

hxx = np.cumsum(hx[0])
hyy = np.cumsum(hy[0])

plt.xlabel('value')
plt.ylabel('frequency')

plt.bar(hx[1][:-1],hxx.astype(float)/np.size(x),edgecolor = 'r', color = [], width = .4, linewidth = 2)
plt.pause(1)
plt.bar(hy[1][:-1],hyy.astype(float)/np.size(y),edgecolor = 'k', color = [], width = .4, linewidth = 2)





