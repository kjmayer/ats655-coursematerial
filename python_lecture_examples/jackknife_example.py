# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""
#.............................................
# IMPORT STATEMENTS
#.............................................
#import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import scipy.signal as sig
import scipy.stats as stats
import scipy.io as sio
#import numpy.ma as ma
#import csv

import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ion()
#%matplotlib inline
#%matplotlib qt

#%% load data
DATA = sio.loadmat('nao_timeseries.mat')

X = DATA['NAO'][:,0] # grab January data only
TIME_NAO = DATA['TIME_NAO'][:,0]

#%% plot data
gf.cfig(1)
plt.plot(TIME_NAO,X,color = 'black', linewidth = 1.5)
pfull = np.polyfit(TIME_NAO,X,1)
plt.plot(TIME_NAO,TIME_NAO*pfull[0]+pfull[1],'--',color = 'black', linewidth = 1.5)
plt.xlabel('year');
plt.ylabel('NAO index');
plt.title('Estimation of NAO best-fit with jackknife');
plt.ylim(-2.4,2.4);
plt.xlim(min(TIME_NAO), max(TIME_NAO));
gf.plot_zero_lines();


gf.show_plot()

#%%
M = np.empty([len(X),2])

for j, val in enumerate(X):
    
    X2 = X
    X2 = np.delete(X2,j)
    
    T2 = TIME_NAO
    T2 = np.delete(T2,j)
    
    pfull = np.polyfit(T2,X2,1)
    
    M[j,0] = pfull[0]
    M[j,1] = pfull[1]
    
    plt.plot(TIME_NAO[j],val,'.',color = 'red', markersize = 15)
    plt.plot(T2,T2*pfull[0] + pfull[1],'--', color = np.random.random_sample(size = 3))
    plt.pause(.1)
    
#%%
gf.cfig(2)
xint = np.arange(.01,.02,.00025)
y, bin_edges = np.histogram(M,xint)
plt.plot(bin_edges[:-1],y/float(len(M)))
plt.xlabel('slope')
plt.ylabel('frequency')
plt.title('Distribution of NAO slopes from jackknife')

gf.show_plot()

gf.cfig(3)
xint = np.arange(-36.,-22.,.5)
y, bin_edges = np.histogram(M,xint)
plt.plot(bin_edges[:-1],y/float(len(M)))
plt.xlabel('y-intercept')
plt.ylabel('frequency')
plt.title('Distribution of NAO slopes from jackknife')

gf.show_plot()
        