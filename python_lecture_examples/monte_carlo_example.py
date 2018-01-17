# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""
#.............................................
# IMPORT STATEMENTS
#.............................................
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gf

reload(gf)

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt


#%% START CODE

bin_width = .1
drawn_max = 2.2
sample_length = 31

Z = np.random.randn(10000,1)
M = np.empty([100000, 1])

for iloop in range(100000):
    ip = np.random.randint(low=0,high=Z.shape[0],size=sample_length) 
    M[iloop] = np.max(Z[ip])
    

#%% plot results

gf.fig(2)
bin_list = np.arange(-1,5,bin_width)
n, bins = np.histogram(M, bins=bin_list, density=False)
plt.plot([drawn_max, drawn_max],[0, .12],color='red',linewidth=3,linestyle='--')
plt.bar(bins[0:-1],n/float(len(M)),bin_width, facecolor='blue', alpha=0.4)
plt.plot(bins[0:-1]+bin_width/2,n/float(len(M)),color='blue')

plt.xlabel('maximum value')
plt.ylabel('frequency')
titlename = 'Maximum value for sample of length N = ' + str(sample_length) 
plt.title(titlename)

plt.show()
#plt.close()