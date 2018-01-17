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
import scipy.signal as sig
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
#%matplotlib inline
#%matplotlib qt



#%% START CODE
t = np.arange(1,40)
x = np.zeros(np.shape(t))

x[np.size(t)/2:np.size(t)/2+2] = np.ones((2,))
#x = np.random.rand(np.shape(t))
#x[0] = 1.

#%%
g = [1., 1., 1.]

y1 = sig.lfilter(g,np.sum(g),x)
y2 = sig.filtfilt(g,np.sum(g),x)

gf.cfig(1)

plt.title('boxcar smoothing');
plt.plot(t,x,'-k',linewidth = 2, label ='original data')
plt.plot(t,y1,'--r',linewidth = 3, label = 'smoothed with 1-1-1' )

plt.ylim(0,1.1)

plt.legend(fontsize = 14, frameon = False)

gf.show_plot()

#%%
plt.plot(t,y2,'--b',linewidth = 3, label = 'smoothed with 1-1-1 twice' )

plt.legend(fontsize = 14, frameon = False)

gf.show_plot()
#%%
Z_x = np.fft.fft(x)/np.size(x)
Z_y1 = np.fft.fft(y1)/np.size(y1)
Z_y2 = np.fft.fft(y2)/np.size(y2)

Ck2_x = np.abs(Z_x[0:np.size(Z_x)/2 + 1])**2
Ck2_y1 = np.abs(Z_y1[0:np.size(Z_y1)/2 + 1])**2
Ck2_y2 = np.abs(Z_y2[0:np.size(Z_y2)/2 + 1])**2

#%%
freq = np.arange(0,np.size(x)/2+1)/float(np.size(x))

Rg_y1 = 1./3 + (2./3)*np.cos(freq*2.*np.pi)
Rg2_y1 = Rg_y1**2

Rg_y2 = (1./3 + (2./3)*np.cos(freq*2*np.pi))**2
Rg2_y2 = Rg_y2**2


#%% plot spectrum of the data and filtered data

maxval = np.max(Ck2_x)

gf.cfig(3)
plt.title('spectra')

plt.plot(freq,Ck2_x/maxval,'-k',linewidth = 2, label = 'original data')
plt.plot(freq,Ck2_y1/maxval,'-r',linewidth = 2, label = 'data after applying forward 1-1-1')
plt.plot(freq,Ck2_y2/maxval,'-b',linewidth = 2, label = 'data after applying forward/backward 1-1-1')

plt.legend(fontsize = 14, frameon = False)

plt.xlabel('frequency')
plt.ylabel('normalized power')

gf.show_plot()

#%% plot the response functions squared (to compare with C_k^2

gf.cfig(4)
plt.title('squared response functions for filters')

plt.plot(freq,Rg2_y1,'-k',linewidth = 2, label = '1-1-1 theoretical response')
plt.plot(freq,Ck2_y1/Ck2_x,'--r',linewidth = 2, label = '1-1-1 Ck^2_{output}/Ck^2_{orig}')

plt.plot(freq,Rg2_y2,'-k',linewidth = 2, label = '1-1-1 x2 theoretical response')
plt.plot(freq,Ck2_y2/Ck2_x,'--',color = 'cornflowerblue',linewidth = 2, label = '1-1-1 x2 Ck^2_{output}/Ck^2_{orig}')

plt.legend(fontsize = 14, frameon = False)

plt.ylabel('filter power factor')
plt.xlabel('frequency')


gf.show_plot()
