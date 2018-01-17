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

# http://pacal.sourceforge.net/getting_started.html
import pacal as pc

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
savefigpath = '/Users/eabarnes/GoogleDrive/WORK/TEACHING/ATS655/S17/python_code/figures/'


#%% START CODE

mu_w = 0
var_w = .75
inc = .1

X = pc.UniformDistr(-2,2)
#X= pc.NormalDistr(mu_w+.25,var_w*.5)
#W = pc.UniformDistr(-1,1)
W = pc.NormalDistr(mu_w,var_w)

Y = X + W

gf.fig(11)

X.plot(color = 'black', label = '$X(x)$')
W.plot(color = 'blue', label = '$W(x)$')
Y.plot(color = 'red', label = '$Y(x) = X(x) + W(x)$')

plt.legend(fontsize = 15)

plt.xlabel('x')
plt.xlim(-6,6)
#plt.xticks(np.arange(-10,10,2))
#plt.xlim(-7,7)

plt.ylabel('probability')

plt.legend(fontsize = 15)

plt.title('Distribution of $Y(x)$')

gf.save_fig_file_cmyk(savefigpath + 'ch2_' + 'bayes_signal_pdfs', dpi = tbk.dpi_save)

pc.show()

#%%
inc = .05

xvec = np.arange(-6.+inc/2.,6.+inc/2.,inc)
yout_vec = xvec

fx_y_expectation = []

for yout in yout_vec:

    fx_y = []       
    
    # probability of Y = yout
    fy = Y.cdf(yout+inc/2.) - Y.cdf(yout-inc/2.)
    
    for xout in xvec:
        
        # probability of Y=yout given X=x is just the error distribution about x
        fy_x = stats.norm.cdf(yout+inc/2.,loc = xout,scale = var_w) - stats.norm.cdf(yout-inc/2.,loc = xout, scale = var_w)
#        fy_x = fy
        
        # probability that x = xout
        fx = X.cdf(xout+inc/2.) - X.cdf(xout-inc/2.)
        
        #print [xout-inc/2., xout+inc/2.]
        
        a = (fy_x * fx/fy)
        
        fx_y.append(a)
    
    fx_y = np.array(fx_y)
    fx_y_expectation.append(np.sum(yout_vec.astype(float) * fx_y))

    example_val = 2.2-inc/2.
    isclose_bool, j = gf.isclose(yout, example_val)
    
    if(isclose_bool):
        gf.cfig(5)
        plt.plot(xvec,fx_y, 'o-', markersize = 5, linewidth = 1)

        gf.show_plot()
        print np.sum(fx_y)
        print np.trapz(fx_y/inc, dx=inc)
        plt.ylabel('probability within interval')
        b = fx_y_expectation[-1]
        plt.plot([b,b],[0,.14],'--',color = 'gray')
        
        plt.xlim(-4.5,4.5)
        plt.ylim(0,0.095)
        plt.xlabel('x')
        plt.title('$\mathit{f}(X = x \pm \epsilon | y = ' + str(example_val+inc/2.) + ' \pm \epsilon)$')
        plt.ylabel('probability')
        
        gf.show_plot()

        gf.save_fig_file_cmyk(savefigpath + 'ch2_' + 'bayes_signal_soln_a', dpi = tbk.dpi_save)    

#
#%%
    
gf.cfig(2)
plt.plot(yout_vec,fx_y_expectation)    
plt.xlabel('y')
plt.title('E(X|Y=y)')
plt.ylabel('x')

plt.xlim(-4.5,4.5)
plt.ylim(-2.5,2.5)
gf.plot_zero_lines()

plt.plot([-60,60],[-2,-2],'--',linewidth = 1.5, color = 'black')
plt.plot([-60,60],[2,2],'--',linewidth = 1.5, color = 'black')

gf.save_fig_file_cmyk(savefigpath + 'ch2_' + 'bayes_signal_expectation', dpi = tbk.dpi_save)
gf.show_plot()





