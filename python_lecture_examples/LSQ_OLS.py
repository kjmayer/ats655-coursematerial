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

from sklearn import datasets, linear_model
#import matplotlib.mlab as mlab
#import scipy.signal as sig
from scipy import stats, odr
import numpy.linalg as LA
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
plt.ion()
#%matplotlib inline
#%matplotlib qt



#%% START CODE
LW = 2

x = 5.*np.random.randn(50)
y = x+5.+3.*np.random.randn(len(x))

x = x - np.mean(x)
y = y - np.mean(y)

gf.cfig(1)
plt.plot(x,y,'ok',markersize = 15, label = 'DATA')
plt.ylabel('y value')
plt.xlabel('x value')

plt.legend(frameon = 0, loc = 'upper left', fontsize = 15)

plt.xlim([-15,15])
plt.ylim([-30, 30])

gf.show_plot()


#%% fit LSQ lines

gf.fig(1)
# X VERSUS Y
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
plt.plot(x,intercept+x*slope,'-',color = 'blue', label = 'LSQ: x vs y', linewidth = LW)


# Y VERSUS X
slope, intercept, r_value, p_value, std_err = stats.linregress(y,x)
plt.plot(x,(1./slope)*x - intercept/slope,'-',color = 'green', label = 'LSQ: y vs x', linewidth = LW)

plt.legend(frameon = 0, loc = 'upper left', fontsize = 15)
# USING SKLEARN
# Create linear regression object
#regr = linear_model.LinearRegression()
# Train the model using the training sets
#regr.fit(x.reshape(-1,1), y.reshape(-1,1))

#%% fit OLS lines

gf.fig(1)
C = np.cov([x,y], rowvar = 1)
LAM, E = LA.eig(C)

plt.plot(np.array([-E[0][1],E[0][1]])*30.,np.array([-E[1][1],E[1][1]])*30.,'-', color = 'deeppink', linewidth = LW, label  = 'OLS')

plt.legend(frameon = 0, loc = 'upper left', fontsize = 15)

