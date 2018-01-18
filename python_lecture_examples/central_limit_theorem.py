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
#gf.cc()
#plt.ion()

# need to set as Automatic graphics

#plt.ioff()
#%matplotlib inline
#%matplotlib qt



#%% START CODE
N0 = 10000000

xinc = np.arange(-10,10,.01)

Rblack= np.random.lognormal(0.,2.,size=(N0,))
Rred = np.random.uniform(-5.,5.,size=(N0,))
Rgreen = np.random.normal(0,1,size = (N0,))

#%%
gf.cfig(11)

hx = np.histogram(Rblack,xinc)
plt.plot(hx[1][:-1],hx[0]/float(N0),'-',color='orange', label='lognormal')

hx = np.histogram(Rred,xinc)
plt.plot(hx[1][:-1],hx[0]/float(N0),'-',color='blue',label = 'uniform')

hx = np.histogram(Rgreen,xinc)
plt.plot(hx[1][:-1],hx[0]/float(N0),'-',color='black', label = 'normal')

plt.legend(fontsize = 15)
plt.ylabel('frequency')

plt.xlim(-8,8)

plt.show()

#%%

bin_width = .05
gf.cfig(12)


for N in (5, 25, 100):

    y2 = []
    
    if(N == 5):
        clr = 'black'
    elif(N==25):
        clr = 'orange'
    elif(N==100):
        clr = 'blue'
        
    
    for i in np.arange(0,10000):
    
        y2.append(np.mean(np.random.normal(loc = 0, scale = 1., size = N)))        
    
    # calc histograms
    bins = np.arange(-8,8,bin_width)
    y2hist, x = np.histogram(y2,bins = bins)
    
    x = bins[0:-1]
    plt.plot(x,y2hist/(float(len(y2))), color = clr, label = 'N = ' + str(N))
    plt.plot(x,(bin_width)*stats.norm.pdf(x+bin_width/2, loc = 0, scale = 1./np.sqrt(N)), color = 'green', linestyle = '--', linewidth = 1.75)

    

plt.xticks(np.arange(-10,10,2))
plt.xticks(np.arange(-2,2,.5))
plt.xlim(-1.7,1.7)

plt.xlabel(r'$\overline{x}_N$')
plt.ylabel('frequency')

plt.legend(fontsize = 15)
plt.title('Standard Normal distribution of means')

plt.show()

#%%

bin_width = .01

for N in (5, 25, 100, 200):

    y1, y2, y3, y4 = [], [], [], []
    
    for i in np.arange(0,10000):
    
        y2.append(np.mean(np.random.normal(loc = 0, scale = 1., size = N)))    
        y1.append(np.mean(np.random.chisquare(3., size = N)))
        y3.append(np.mean(np.random.lognormal(mean = 0, sigma = 1, size = N)))
        y4.append(np.mean(np.random.uniform(low = -3, high = 6, size = N)))
    
    
    sigma_y1 = np.std(np.random.chisquare(3., size = N))
    sigma_y3 = np.std(np.random.lognormal(mean = 0, sigma = 1, size = N))
    sigma_y4 = np.std(np.random.uniform(low = -3, high = 6, size = N))
    
    # calc histograms
    bins = np.arange(-8,8,bin_width)
    y1hist, x = np.histogram(y1,bins = bins)
    y2hist, x = np.histogram(y2,bins = bins)
    y3hist, x = np.histogram(y3,bins = bins)
    y4hist, x = np.histogram(y4,bins = bins)
    
    x = bins[0:-1]
    
    gf.cfig(1)
    plt.plot(x,y2hist/(float(len(y2))), color = 'black', label = 'Standard Normal')
    plt.plot(x,y3hist/(float(len(y3))),  color = 'orange', label = 'Lognormal')
#    plt.plot(x,y1hist/(float(len(y1))), color = tbk.clr1, label = 'Chi-Squared')
    plt.plot(x,y4hist/(float(len(y4))), color = 'blue', label = 'Uniform')
    
    if(N>=0):
#        plt.plot(x,(bin_width)*stats.norm.pdf(x+bin_width/2, loc = np.mean(y1), scale = sigma_y1/np.sqrt(N)), color = 'green', linestyle = '--', linewidth = 1.75)
        plt.plot(x,(bin_width)*stats.norm.pdf(x+bin_width/2, loc = np.mean(y2), scale = 1./np.sqrt(N)), color = 'green', linestyle = '--', linewidth = 1.75)
        plt.plot(x,(bin_width)*stats.norm.pdf(x+bin_width/2, loc = np.mean(y3), scale = sigma_y3/np.sqrt(N)), color = 'green', linestyle = '--', linewidth = 1.75)
        plt.plot(x,(bin_width)*stats.norm.pdf(x+bin_width/2, loc = np.mean(y4), scale = sigma_y4/np.sqrt(N)), color = 'green', linestyle = '--', linewidth = 1.75)    
    
    plt.legend(fontsize = 15)
    
    plt.xticks(np.arange(-10,10,2))
    plt.xlim(-1.6,4.4)
    
    textprint = r'$\overline{x}_{N}$'
    plt.xlabel(textprint)
    plt.ylabel('frequency')
    
    plt.title('N = ' + str(N))
    plt.legend(fontsize = 15)
    
    plt.show()   
    plt.pause(4)
    
    
    
    


