#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:47:26 2017

@author: eabarnes
"""

#.............................................
# IMPORT STATEMENTS
#.............................................
#import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import scipy.signal as sig
import scipy.stats as stats
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


#%% CODE FROM TESTING_SPECTRAL_SIGNIFICANCE.PY

T = 256         #length of window
N = 40          #number of realizations

alpha = 0.5     #red noise lag-one autocorrelation

T2 = T/2
freq = np.arange(0.,T2+1.)/T

# contstruct expected rednoise spectrum
rspec = []
for i in np.arange(1,T2+2,1):
    rspec.append((1.-alpha*alpha)/(1.-2.*alpha*np.cos(np.pi*(i-1.)/T2)+alpha*alpha))
    
factor = np.sqrt(1.-alpha*alpha)

x = np.zeros(T,)
pnum = 0
# loop realizations
for ir in np.arange(0,N+1,1):
    
    x[0] = x[-1]*alpha + factor*np.random.randn()

    for j in np.arange(1,T,1):
        x[j] = x[j-1]*alpha + factor*np.random.randn()+0.5*np.cos(2.*np.pi*(1.-0.01*np.random.randn())*50./256.*j)
        

    p = sig.welch(x,window='hanning', nperseg=T);
    if(ir==0):
        psum = p[1]
    else:
        psum = psum + p[1]

    # calculate average    
    pave = psum/(ir+1.0)
    #normalize the spectrum
    pave = pave/np.mean(pave)
 
    
    # calculate significance
    dof = 2.*(ir+1.)
    fstat = stats.f.ppf(.99,dof,1000)
    spec99 = [fstat*m for m in rspec]
    
    if((ir+1.) % 20 == 0 or ir==0):       
        gf.cfig(1)
        plt.xlabel('frequency (cycles per time step)')
        plt.ylabel('power')
        plt.title('# Realizations = ' + str(ir+1))
        plt.ylim(0,rspec[0]*2.)
        plt.plot(freq,pave,'-k', label = 'data')
        plt.plot(freq,rspec,'-', label = 'red-noise fit', color = 'blue')
        plt.plot(freq,spec99,'--', label = '99% confidence', color = 'red')
        plt.legend(frameon = False)
        gf.show_plot()
        plt.pause(1)

#%%
p_len = 3

# true data spectrum of infinite time series
gf.cfig(2)
plt.plot(freq,pave, '.-k', linewidth = 1.5, markersize = 6, label = 'data spectrum')
plt.xlabel('frequency (cycles per time step)')
plt.ylabel('power')
plt.xlim(0,.5)
plt.ylim(-1., 7.)
gf.plot_zero_lines()
gf.show_plot()

#================
# now assume you only have 30 days instead of an infinite number
T = 30
#================

omega = np.append([-1.*freq*2.*np.pi],[freq*2.*np.pi])
omega = np.sort(omega)
omega = omega[np.size(omega)/4:3*np.size(omega)/4:1]

# use sinc function
B = np.sinc(omega*T/(2.*np.pi))

# plot results
plt.plot(omega/(2.*np.pi)+.1, B/np.max(B), '--r', linewidth = 1.25, label = 'response function of rectangular window')
j = gf.isclosest(freq,.1)[1]
plt.plot(freq[j],pave[j],'.r',markersize = 25)
plt.pause(p_len)

plt.plot(omega/(2.*np.pi)+.2, B/np.max(B), '--r', linewidth = 1.25)
j = gf.isclosest(freq,.2)[1]
plt.plot(freq[j],pave[j],'.r',markersize = 25)
plt.pause(p_len)

plt.plot(omega/(2.*np.pi)+.3, B/np.max(B), '--r', linewidth = 1.25)
j = gf.isclosest(freq,.3)[1]
plt.plot(freq[j],pave[j],'.r',markersize = 25)
plt.pause(p_len)

Cb = np.convolve(pave,B/np.sum(B), mode = 'same')
plt.plot(freq,Cb,'-r',linewidth = 4, label = 'convolution')

plt.legend(frameon = False)
