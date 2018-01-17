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

#%% plot Z versus t distributions

x = np.arange(-6,6,.01)
z = stats.norm.pdf(x,0,1)
t = stats.t.pdf(x,4)

gf.cfig(1)
plt.plot(x,z, color = tbk.clr2, label = 'Z')
plt.plot(x,t,linestyle = '-', color = tbk.clr1, label = r"t ($\nu$ = 4)")

plt.title('Z and Student-t probability density functions')
plt.ylabel('f(Z)')
plt.legend(frameon = 0)

plt.xlim(-5,5)
plt.ylim(0,.45)
plt.yticks(np.arange(0,.5,.1))

gf.plot_zero_lines()

gf.show_plot()
