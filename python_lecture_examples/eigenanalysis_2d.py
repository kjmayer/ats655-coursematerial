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
#import scipy.stats as stats
#import numpy.ma as ma
#import csv
import numpy.linalg as LA

import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

import textbook_params as tbk
reload(tbk)

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ion()
#%matplotlib inline
#%matplotlib qt

#.............................................
# INTIAL SETTINGS


#%%
mult_fact = 15
#%%

FS = 15
inc = .2

A = [[-1, 1],[1, -1]]
C = np.cov(A, rowvar = 0)
LAM, E = LA.eig(C)
A_new = np.dot(A,E)


gf.cfig(1, fig_width = 70, fig_height = 70)
plt.plot(zip(*A)[0],zip(*A)[1], marker = 'o', linestyle = '', markersize = 20, color = tbk.clr1,markeredgecolor = 'lightgray')

plt.text(zip(*A)[0][0]-inc, zip(*A)[1][0], r"($-1$,$1$)", color = 'black', fontsize = FS, horizontalalignment = 'right',verticalalignment = 'bottom')
plt.text(zip(*A)[0][0]-inc, zip(*A)[1][0], r"($-\sqrt{2}$,$0$)", color = tbk.clr3, fontsize = FS, horizontalalignment = 'right', verticalalignment = 'top')

plt.text(zip(*A)[0][1]+inc, zip(*A)[1][1], r"($1$,$-1$)", color = 'black', fontsize = FS, horizontalalignment = 'left',verticalalignment = 'bottom')
plt.text(zip(*A)[0][1]+inc, zip(*A)[1][1], r"($\sqrt{2}$,$0$)", color = tbk.clr3, fontsize = FS, horizontalalignment = 'left', verticalalignment = 'top')

var_exp = 100.*LAM[0]/np.sum(LAM)
plt.text(.75,3.75,'Total variance = ' + str(np.sum(LAM)) + ',  $\lambda_1$ = ' + str(LAM[0]), color = tbk.clr3, fontsize = 14)

var_exp = 100.*LAM[0]/np.sum(LAM)
plt.text(.75,3.35,'variance explained = ' + str(var_exp) + '%', color = tbk.clr3, fontsize = 14)

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.xlabel('x')
plt.ylabel('y')    

plt.plot([E[0,0]*-mult_fact, E[0,0]*mult_fact],[E[-1,0]*-mult_fact, E[-1,0]*mult_fact],linestyle = '--', linewidth = 2, color = tbk.clr3)

gf.plot_zero_lines(linewidth = 2)


#%%

inc = .15


A = [[-1, 1],[1, -1],[-1.1, 2]]
C = np.cov(A, rowvar = 0)
LAM, E = LA.eig(C)
E[:,[0,1]] = E[:,[1,0]]
E[:,1] *=-1
A_new = np.dot(A,E)


gf.cfig(2, fig_width = 70, fig_height = 70)
plt.plot(zip(*A)[0],zip(*A)[1], marker = 'o', linestyle = '', markersize = 20, color = tbk.clr1,markeredgecolor = 'lightgray')

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.plot([E[0,0]*-mult_fact, E[0,0]*mult_fact],[E[-1,0]*-mult_fact, E[-1,0]*mult_fact],linestyle = '--', linewidth = 2, color = tbk.clr3)
plt.plot([E[0,1]*-mult_fact, E[0,1]*mult_fact],[E[-1,1]*-mult_fact, E[-1,1]*mult_fact],linestyle = '--', linewidth = 2, color = tbk.clr3)

txt_str = r"($" + str(zip(*A)[0][0]) + r"$, $" + str(zip(*A)[1][0]) + r"$)"
plt.text(zip(*A)[0][0]-inc, zip(*A)[1][0], txt_str, color = 'black', fontsize = FS, horizontalalignment = 'right',verticalalignment = 'bottom')

txt_str = r"($" + str(zip(*A)[0][1]) + r"$, $" + str(zip(*A)[1][1]) + r"$)"
plt.text(zip(*A)[0][1]+inc, zip(*A)[1][1], txt_str, color = 'black', fontsize = FS, horizontalalignment = 'left',verticalalignment = 'bottom')

txt_str = r"($" + str(zip(*A)[0][2]) + r"$, $" + str(zip(*A)[1][2]) + r"$)"
plt.text(zip(*A)[0][2]+inc, zip(*A)[1][2], txt_str, color = 'black', fontsize = FS, horizontalalignment = 'left',verticalalignment = 'bottom')



txt_str = r"($" + str(round(zip(*A_new)[0][0],1)) + r"$, $" + str(round(zip(*A_new)[1][0],1)) + r"$)"
plt.text(zip(*A)[0][0]-inc, zip(*A)[1][0], txt_str, color = tbk.clr3, fontsize = FS, horizontalalignment = 'right',verticalalignment = 'top')

txt_str = r"($" + str(round(zip(*A_new)[0][1],1)) + r"$, $" + str(round(zip(*A_new)[1][1],1)) + r"$)"
plt.text(zip(*A)[0][1]+inc, zip(*A)[1][1], txt_str, color = tbk.clr3, fontsize = FS, horizontalalignment = 'left',verticalalignment = 'top')

txt_str = r"($" + str(round(zip(*A_new)[0][2],1)) + r"$, $" + str(round(zip(*A_new)[1][2],1)) + r"$)"
plt.text(zip(*A)[0][2]+inc, zip(*A)[1][2], txt_str, color = tbk.clr3, fontsize = FS, horizontalalignment = 'left',verticalalignment = 'top')


plt.text(.05,3.75,'Total variance = ' + str(gf.round_to_nearest(np.sum(LAM),.01)) + ',  $\lambda_1$ = ' + str(gf.round_to_nearest(LAM[1],.01))+ ',  $\lambda_2$ = ' + str(gf.round_to_nearest(LAM[0],.01)), color = tbk.clr3, fontsize = 14)


var_exp = 100.*LAM[0]/np.sum(LAM)
var_exp2 = 100.*LAM[1]/np.sum(LAM)
plt.text(.05,3.35,'variance explained = ' + str(gf.round_to_nearest(var_exp2,.1)) + '%, ' + str(gf.round_to_nearest(var_exp,.1)) + '%', color = tbk.clr3, fontsize = 14)


plt.text(-3.5, -3, "eigenvector 2", color = tbk.clr3, fontsize = FS, fontweight = 'bold')   
plt.text(-2.5, 3.5, "eigenvector 1", color = tbk.clr3, fontsize = FS, fontweight = 'bold')      

plt.xlabel('x')
plt.ylabel('y')    

gf.plot_zero_lines(linewidth = 2)

gf.save_fig_file('../../Lecture_notes/included_figures/ch5_eof_intro_2lines_rgb')











