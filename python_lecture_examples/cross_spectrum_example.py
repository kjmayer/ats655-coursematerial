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
from scipy import stats
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

#%% START CODE
chunk_length = 256
num_chunks= 20
n = chunk_length*num_chunks

# generate red noise time series with autocorrelation
alpha = 0.5
height = 2.0
factor = np.sqrt(1.-alpha*alpha)

x = np.zeros((n,))
y = np.zeros((n,))
pnum = 0

x[0] = x[-1]*alpha + factor*np.random.randn()
y[0] = y[-1]*alpha + factor*np.random.randn()
for j in np.arange(1,n):
    x[j] = x[j-1]*alpha + factor*np.random.randn()+1.0*np.cos(2.*np.pi*(1.-0.01*np.random.randn())*30./256.*j) + 0.75*np.cos(2.*np.pi*(1.-.001*np.random.randn())*63./256*j-np.pi/4.)
    y[j] = y[j-1]*alpha + factor*np.random.randn()+1.0*np.cos(2.*np.pi*(1.-0.01*np.random.randn())*10./256.*j) + 0.75*np.cos(2.*np.pi*(1.-.001*np.random.randn())*63./256*j)
        
xa = x - np.mean(x)
ya = y - np.mean(y)

#%% take a look at the data

t = np.arange(1,np.size(x)+1)

gf.cfig(1)
plt.plot(t,xa,'-k',linewidth = 1, label = 'x anomalies')
plt.plot(t,ya,'-r',linewidth = 1, label = 'y anomalies')

plt.xlim(0,np.size(t)+1)
plt.ylim(-5,5)
plt.xlabel('time')
plt.ylabel('anomaly')
plt.title('Data series x and y')
gf.plot_zero_lines()

gf.show_plot()

plt.pause(2)
plt.xlim(0,150)

#%% calculate and plot spectrum of x and y

win_rect = np.ones(np.size(t)/float(num_chunks),)

F, Pxx = sig.csd(xa, xa, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), scaling = 'density', detrend = False)
F, Pyy = sig.csd(ya, ya, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), scaling = 'density', detrend = False)

gf.cfig(2)
plt.plot(F,Pxx/np.sum(Pxx),'-k',linewidth = 2, label = 'x')
plt.plot(F,Pyy/np.sum(Pyy),'-r',linewidth = 2, label = 'y')

plt.title('Power Spectra of x and y')
plt.ylabel('power (fraction of variance)')
plt.xlabel('frequency (cycles/time step)')

#--------------------------------------------------
# calculate and plot red-noise fit
alpha = sig.correlate(xa,xa,'full')
alpha = alpha/np.max(alpha)
alpha = alpha[np.size(alpha)/2+1]
Te = -1./np.log(alpha)
rnoise = 2.8*2*Te/(1+(Te**2)*(F*2*np.pi)**2)
rnoise = rnoise/np.sum(rnoise)
dof = 2*num_chunks
fst = stats.f.ppf(.99,dof,1000)
spec99 = fst*rnoise
plt.plot(F,rnoise,'-',color = 'gray', label = 'red-noise fit')
plt.plot(F,spec99,'--',color = 'gray', label = '99% conf. bound')
#--------------------------------------------------

plt.legend(fontsize = 14)
gf.show_plot()

#%% cross-spectrum analysis
 
#   calculate the cross-spectrum (both real and imaginary parts; i.e. Fxy)
#   use an overlap of 0.5
F, Pxy = sig.csd(xa, ya, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), scaling = 'density', detrend = False)

# calculate the squared coherence of x and y
F, Cxy = sig.coherence(xa, ya, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), detrend = False)

# check that we understand scipy's sig.coherence code by calculating the
# coherence ourselves
F, Pxx = sig.csd(xa, xa, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), scaling = 'density', detrend = False)
F, Pyy = sig.csd(ya, ya, window = win_rect, noverlap = chunk_length/2, nperseg = chunk_length, nfft = np.size(t)/float(num_chunks), scaling = 'density', detrend = False)
CoherCheck = (np.abs(Pxy)**2)/(Pxx*Pyy)

gf.cfig(3)
plt.plot(F,Cxy,'-k',linewidth = 2)
plt.plot(F,CoherCheck,'.--r',linewidth = 2)
plt.title('Checking that I understand how scipy works!')
plt.ylabel('coherence')
plt.xlabel('frequency (cycles/time step)')
gf.show_plot()

#%% plot the coherence and phase

j = np.argsort(Cxy)[::-1]

gf.cfig(4)
plt.plot(F,Cxy,'-k',linewidth = 2)
plt.plot(F[j[0]],Cxy[j[0]],'or',linewidth = 2,markersize = 10)
plt.plot(F[j[1]],Cxy[j[1]],'ob',linewidth = 2,markersize = 10)
#plt.plot(F[j[2]],Cxy[j[2]],'og',linewidth = 2,markersize = 10)

plt.ylabel('R^2')
plt.xlabel('frequency (cycles per time step)')
plt.xlim(-.03,.53)

# determine coherence-squared significance
dof = 2*num_chunks
fval = stats.f.ppf(0.99, 2, dof - 2)
r2_cutoff = fval / (num_chunks - 1. + fval)

plt.plot(F,np.ones(np.size(F))*r2_cutoff,'--r',linewidth = 2, label = '99% sig. cutoff')

plt.legend(fontsize = 14)

plt.title('Coherence Squared')

gf.show_plot()

#%% plot the quadrature spectrum

P = -np.angle(Pxy, deg = True)

gf.cfig(5)
plt.plot(F,P,'-k',linewidth = 2)
plt.plot(F[j[0]],P[j[0]],'or',linewidth = 2,markersize = 10)
plt.plot(F[j[1]],P[j[1]],'ob',linewidth = 2,markersize = 10)
#plt.plot(F[j[2]],P[j[2]],'og',linewidth = 2,markersize = 10)

plt.title('Phase difference')
plt.ylabel('degrees')
plt.xlabel('frequency (cycles per time step)')
plt.xlim(-.03,.53)
plt.ylim(-185,185)
gf.plot_zero_lines()
gf.show_plot()

print '-------------------------------------'
print '            PERIOD ........ PHASE'
print '   RED:     ' + str(gf.round_to_nearest(1./F[j[0]],1)) + ' dys.        ' + str(np.round(P[j[0]])) + ' deg.'
print '   BLUE:    ' + str(gf.round_to_nearest(1./F[j[1]],1)) + ' dys.        ' + str(np.round(P[j[1]])) + ' deg.'
#print '   GREEN:   ' + str(gf.round_to_nearest(1./F[j[2]],1)) + ' dys.        ' + str(np.round(P[j[2]])) + ' deg.'


