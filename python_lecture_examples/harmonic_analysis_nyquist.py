#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:45:51 2017

@author: eabarnes
"""
#.............................................
# IMPORT STATEMENTS
#.............................................
#import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import scipy.signal as sig
#import scipy.stats as stats
#import numpy.ma as ma
#import csv
#import scipy.io as sio
#import numpy.linalg as LA
#from mpl_toolkits.basemap import Basemap
#from matplotlib.patches import Polygon


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

#%% input by user
T = 4.0
t = np.arange(0.,T+0.05,0.05)
N = np.size(t)-1

COLOR_MAT = ['darkgreen','cornflowerblue','red','orange','navy','hotpink','grey']
#%%
gf.cfig(1)
plt.xlim(np.min(t),np.max(t))
plt.ylim(-1.75,1.75)
plt.xlabel('time')
plt.title('N = ' + str(N))
gf.plot_zero_lines()

plt.show()

#%% example showing waves of different wave-lengths, and why k=N/2 is special

count = -1
print '--------------------------'

for k in (1,3,N/2):
    count = count + 1
    
    y1 = np.cos(2.*np.pi*k*t/T)
    y2 = np.sin(2.*np.pi*k*t/T)

    #gf.fig(1)
    plt.plot(t,y1,'x--',color = COLOR_MAT[count],linewidth = 0.75, label = 'k='+str(k))
    plt.plot(t,y2,'o--',color = COLOR_MAT[count],linewidth = 0.75)

    print 'Variance of cos for k = ' + str(k) + ': ' + str(np.var(y1))
    print 'Variance of sin for k = ' + str(k) + ': ' + str(np.var(y2))
    print '.........'

    plt.legend(frameon = False)

    plt.pause(5)
    
plt.plot(t,y1,'x--',color = COLOR_MAT[count], linewidth = 2.)
plt.plot(t,y2,'o--',color = COLOR_MAT[count], linewidth = 2.)    

#%% Aliasing example
gf.cc()

T = 10.
t = np.arange(0.,T+.01,.01)
k = 53 #4, 5, 6, 11

gf.cfig(2)
plt.xlim(np.min(t),np.max(t))
plt.ylim(-1.3,1.3)
plt.xlabel('time')
plt.title('Nyquist cutoff: 5 total waves, 1 per 2 time steps')

for i in np.arange(0,11,1):
    plt.plot((i*1.,i*1.),(-2.,2.),'-',color = 'black',linewidth = .5)

gf.plot_zero_lines()

y1 = np.cos(2.*np.pi*k*t/T)
y2 = np.sin(2.*np.pi*k*t/T)

plt.plot(t,y1,'--',color = 'red', linewidth = 1.)

t1 = np.arange(np.min(t),np.max(t)+1,1)

y11 = np.cos(2.*np.pi*k*t1/T)
y22 = np.sin(2.*np.pi*k*t1/T)

plt.pause(10)

plt.plot(t1,y11,'.',color = 'blue',markersize = 20)

plt.pause(3)

plt.plot(t1,y11,'-',color = 'blue',markersize = 20, linewidth = 1.)

#%% Example demonstrating that sines and cosines are uncorrelated
gf.cc()

T = 100.
t = np.arange(0.,T,.1)

for k in np.arange(1,11,1):
    
    y0 = np.cos(2*np.pi*(k+1)*t/T)
    y1 = np.cos(2*np.pi*(k)*t/T)
    
    y2 = np.sin(2*np.pi*(k)*t/T)

    c01 = np.corrcoef(y0,y1)[1][0]
    c12 = np.corrcoef(y1,y2)[1][0]
    
    print ''
    print 'correlation of cos(' + str(k) + ') and cos(' + str(k+1) + ') = ' + str(gf.round_to_nearest(c01,.0001))
    print 'correlation of cos(' + str(k) + ') and sin(' + str(k) + ') = ' + str(gf.round_to_nearest(c12,.0001))







