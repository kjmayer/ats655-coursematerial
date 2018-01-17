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
#import numpy.linalg as LA
#from scipy.stats import ortho_group
from scipy.cluster.vq import kmeans,vq


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
savefigpath = '../figures/'
#np.set_printoptions(suppress=False)

#%%

def init_board(N,R1,R2):
    X = np.array([(np.random.uniform(R1, R2), np.random.uniform(R1, R2)) for i in range(N)])
    return X

#%%

M = np.empty((0,2))

for i in np.arange(0,7,1):
    randval1 = np.random.uniform(-1,1)
    randval2 = np.random.uniform(-1,1)
    
    R1 = -1. + randval1*1.
    R2 = 1. + randval2*1.
    M = np.append(M,init_board(100,R1,R2), axis = 0)
    
    
gf.cfig(1)
plt.plot(M[:,0],M[:,1],'.',markersize = 8)

plt.xticks([])
plt.yticks([])

yvals = plt.ylim()
xvals = plt.xlim()

plt.title('Data')

gf.save_fig_file(savefigpath + 'ch5_' + 'kmeans_rawdata', dpi = tbk.dpi_save)

plt.show()
gf.cc()
#%%

for NUM_CLUSTERS in [7,4]:

    COLOR_MAT = ['darkgreen','cornflowerblue','red','orange','navy','hotpink','grey']
    
    centroids, _ = kmeans(M, NUM_CLUSTERS, iter=20)
    idx, _ = vq(M,centroids)
    
    gf.cfig(i)
    plt.title(str(NUM_CLUSTERS) + ' Clusters')
    
        
    for (ind,val) in enumerate(M):
        plt.plot(val[0],val[1],'.',color = COLOR_MAT[idx[ind]], markersize = 8)
    
    for (ind,val) in enumerate(centroids):
        plt.plot(val[0],val[1],'*',color = COLOR_MAT[ind],markersize = 25)
        
    plt.xticks([])
    plt.yticks([])
        
    plt.xlim(xvals)
    plt.ylim(yvals)    
    
    gf.save_fig_file(savefigpath + 'ch5_' + 'kmeans_' + str(NUM_CLUSTERS) + 'clusters', dpi = tbk.dpi_save)
    
    plt.show()    
    
    
    
    
    
    
    