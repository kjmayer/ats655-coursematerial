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

#%%

T = 10.1
t = np.arange(0.,T,.1)
N = np.size(t)-1

A1 = 8.
A3 = 10.
B2 = 7.

y = A1*np.cos(2.*np.pi*1.*t/T) + A3*np.cos(2.*np.pi*3.*t/T) + B2*np.sin(2.*np.pi*2.*t/T)

gf.cfig(1)
plt.plot(t,y,'--k',linewidth = 5, label = 'original data')
gf.plot_zero_lines()
plt.xlabel('time')
plt.ylabel('wave amplitude')

#%%
num_coeffs = 7
A_guess = np.empty(num_coeffs)
B_guess = np.empty(num_coeffs)

print '--------------------'

for k in np.arange(1,num_coeffs,1):
   
    A_guess[k] = 2.*np.mean(np.cos(2.*np.pi*k*t/T) * y)
    B_guess[k] = 2.*np.mean(np.sin(2.*np.pi*k*t/T) * y)

    if(A_guess[k]>=1):
        print '   '
        print '.....................'
      
    print 'k = ' + str(k) + ', A_k = ' + str(gf.round_to_nearest(A_guess[k],.01))    

    if(A_guess[k]>=1):
        print '.....................'        
        print '   '

    if(B_guess[k]>=1):
        print '   '
        print '.....................'
      
    print 'k = ' + str(k) + ', B_k = ' + str(gf.round_to_nearest(B_guess[k],.01))    

    if(B_guess[k]>=1):
        print '.....................'        
        print '   '

        
#%% fill-in values you found in previous loop

A1_guess = A_guess[1]
A3_guess = A_guess[3]
B2_guess = B_guess[2]

#gf.fig(1)

plt.plot(t,A1_guess*np.cos(2.*np.pi*1.*t/T),'-b',linewidth = 2,label = 'cos(1x)')
plt.plot(t,A3_guess*np.cos(2.*np.pi*3.*t/T),'-r',linewidth = 2, label = 'cos(3x)')
plt.plot(t,B2_guess*np.sin(2.*np.pi*2.*t/T),'-g',linewidth = 2, label = 'sin(2x)')

plt.legend(frameon = False, loc = 'lower right')
plt.title('components of y in terms of harmonics')

#%%
#gf.fig(1)

y_guess = A1_guess * np.cos(2.*np.pi*1.*t/T) + A3_guess * np.cos(2.*np.pi*3.*t/T) + B2_guess * np.sin(2.*np.pi*2.*t/T)

plt.plot(t,y_guess,'-',color = 'gray', linewidth = 3, label = 'fitted data')

plt.legend(frameon = False, loc = 'lower left', fontsize = 12)








