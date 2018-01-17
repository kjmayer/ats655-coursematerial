#%% START CODE
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
#import numpy.ma as ma
#import csv

import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

import textbook_params as tbk
reload(tbk)

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt

#.............................................
# INTIAL SETTINGS
savefigpath = tbk.fig_dir + 'figs_ch2/'


#%% START CODE

mu_w = 0
var_w = .75
inc = .05

index = np.arange(-6.,6.+inc,inc)
index_plot = index
X = stats.uniform.pdf(index,-2.,4.)
#X = stats.norm.pdf(index,20.,var_w*0.5)
W = stats.norm.pdf(index,mu_w,var_w)
Y = np.convolve(X,W, mode = 'same')
Y = (Y/np.sum(Y))/inc

gf.cfig(1)
plt.plot(index_plot,X, color = 'black', label = '$X(x)$')
plt.plot(index_plot,W, color = 'blue', label = '$W(x)$')
plt.plot(index_plot,Y, color = 'red', label = '$Y(x) = X(x) + W(x)$')


plt.legend(fontsize = 15)

plt.xlabel('x')
#plt.xticks(np.arange(-10,10,2))
plt.xlim(-6,6)

plt.ylabel('probability')

plt.legend(fontsize = 15)

#gf.save_fig_file_cmyk(savefigpath + 'ch2_' + 'bayes_signal_pdfs', dpi = tbk.dpi_save)

plt.title('Distribution of $Y(x)$')

gf.show_plot()

#%% NOT QUITE RIGHT


#
#fx_y_expectation = []
#yout_vec = index
#
#for yout in yout_vec:
#    
#    if(yout % 100 == 0):
#        print 'yout = ' + str(yout)
#    
#    fx_y = []
#    
#    for (ind, val) in enumerate(index):
#        
#        # probability of Y=yout given X=x is just the error distribution about x
#        fy_x = (stats.norm.cdf(yout+inc/2.,val,var_w) - stats.norm.cdf(yout-inc/2.,val,var_w))
#            
#        # probability of Y = yout
#        k = np.where(index==yout)
#    #    k = np.where(np.logical_and(index>yout-inc,index<=yout))
#        fy = Y[k]*inc
#            
#        # probability of X = val
#        #j = np.where(np.logical_and(index>val-inc,index<=val))
#        fx = X[ind]*inc
#        
#        #if(len(j)>1):
#        #    print val, X[j], j
#        #    break
#        
#        a = (fy_x * fx)/fy
#        fx_y.append(a)
#        
#    fx_y = np.array(fx_y)        
#    fx_y_expectation.append(np.sum(index.astype(float) * fx_y[:,0])/100.)
#
#    example_val = 2.2
#
#    isclose_bool, j = gf.isclose(yout/100., example_val)
#    
#    
#    if(isclose_bool):
#        gf.cfig(3)
#        m = np.sum(fx_y[:,0]*inc)
#        fx_y = fx_y/m
#        plt.plot(yout_vec/100.,fx_y[:,0]/(inc/100.),'*-',linewidth = 1)
#        print np.sum(fx_y)
#        print np.trapz(fx_y[:,0], dx=inc/100.)
#        
#        b = fx_y_expectation[-1]
#        plt.plot([b,b],[0,.14],'--',color = 'gray')
#        
#        plt.xlim(-8.,8.)
#        plt.ylim(0,1.2)
#        plt.xlabel('x')
#        plt.title('$\mathit{f}(X = x | Y = y = ' + str(example_val) + ')$')
#        plt.ylabel('probability')
#
#        gf.show_plot()
#        gf.error()
#    
#
#    
#
#    
#
##%%
##gf.cfig(1)
##plt.plot(index_plot,fx_y)    
##plt.show()
#    
#gf.cfig(2)
#plt.plot(yout_vec/100.,fx_y_expectation)    
#plt.xlabel('y')
#plt.ylabel('E(X|Y=y)')
#
#plt.xlim(-3.5,3.5)
#plt.ylim(-2.5,2.5)
#gf.plot_zero_lines()
#
#gf.save_fig_file_cmyk(savefigpath + 'ch2_' + 'bayes_signal_expectation', dpi = tbk.dpi_save)
#plt.show()
#
#
#
#
#
