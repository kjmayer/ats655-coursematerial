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
plt.ioff()
#%matplotlib inline
#%matplotlib qt



#%% START CODE

filename = 'homework_1_data_Y1.csv'
X = np.genfromtxt(filename, delimiter = ',')

#X = X[3999::1]

xindex = (np.arange(0,np.size(X),1))/24.

#%% ------------------------------------------------------------------------------------------------

gf.cfig(1)
plt.plot(xindex,X, linewidth = 0.75)

v = np.std(X)
plt.text(0.95, 0.925,'$\sigma = $' + str(np.round(v)) + '$^o$F', ha='right', transform=plt.gca().transAxes, color = 'k')

gf.plot_zero_lines()
plt.ylabel('temperature (deg. F)')
plt.xlabel('days')
plt.title('Hourly temperature at Christman Field (2011-2013)')
gf.show_plot()

#%% ------------------------------------------------------------------------------------------------
gf.cfig(2)
plt.plot(xindex[::24],X[::24], linewidth = 0.74, color = 'red')

v = np.std(X[::24])
plt.text(0.95, 0.925,'$\sigma = $' + str(np.round(v)) + '$^o$F', ha='right', transform=plt.gca().transAxes, color = 'red')

gf.plot_zero_lines()
plt.ylabel('temperature (deg. F)')
plt.xlabel('days')

plt.title('Sampled once-a-day')

gf.show_plot()

#%% ------------------------------------------------------------------------------------------------
gf.cfig(3)
plt.plot(xindex[::720],X[::720], linewidth = 0.74, color = 'blue')

v = np.std(X[::720])
plt.text(0.95, 0.925,'$\sigma = $' + str(np.round(v)) + '$^o$F', ha='right', transform=plt.gca().transAxes, color = 'blue')

gf.plot_zero_lines()
plt.ylabel('temperature (deg. F)')
plt.xlabel('days')

plt.title('Sampled once-a-month')

gf.show_plot()


#%% ------------------------------------------------------------------------------------------------
gf.cfig(4)
plt.plot(xindex[4000:4000+720:24],X[4000:4000+720:24], linewidth = 0.74, color = 'green')

v = np.std(X[4000:4000+720:24])
plt.text(0.95, 0.925,'$\sigma = $' + str(np.round(v)) + '$^o$F', ha='right', transform=plt.gca().transAxes, color = 'green')

gf.plot_zero_lines()
plt.ylabel('temperature (deg. F)')
plt.xlabel('days')

plt.title('Sampled daily over one month')

gf.show_plot()


#%% ------------------------------------------------------------------------------------------------
gf.cfig(5)
plt.plot(xindex[4000:4000+24:1],X[4000:4000+24:1], linewidth = 0.74, color = 'magenta')

v = np.std(X[4000:4000+24:1])
plt.text(0.95, 0.925,'$\sigma = $' + str(np.round(v)) + '$^o$F', ha='right', transform=plt.gca().transAxes, color = 'magenta')

gf.plot_zero_lines()
plt.ylabel('temperature (deg. F)')
plt.xlabel('days')

plt.title('Sampled daily over one day')

gf.show_plot()















