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
import random
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

#scipy.linalg

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt

savefigpath = '../figures/'

#%% Define functions

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
     
#def find_centers(X, K):
#    # Initialize to K random centers
#    oldmu = random.sample(X, K)
#    mu = random.sample(X, K)
#    while not has_converged(mu, oldmu):
#        oldmu = mu
#        # Assign all points in X to clusters
#        clusters = cluster_points(X, mu)
#        # Reevaluate centers
#        mu = reevaluate_centers(oldmu, clusters)
#    return(mu, clusters)

# EAB: code modified to show only one step at a time
def find_centers(X, K):
    if isinstance(K, list):
        oldmu = K
        mu = K
    else:
        print 'getting random centers'
        # Initialize to K random centers
        oldmu = random.sample(X, K)
        mu = random.sample(X, K)
    for i in range(0,1):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
    
    
#%%

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X
    
def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([random.gauss(c[0], s), random.gauss(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X    

#%%
COLOR_MAT = ['red','limegreen','blue','cornflowerblue','red','orange','navy','hotpink','grey']
MS = 20

#X = init_board_gauss(10,3)
#X = init_board(20)
X = np.array([[0,3],[0,2],[0,1],[2,-1],[1.9,-2],[3,-3],[4,-3],[5.1,-2],[5,-1],[7,1],[7,1.5],[7,3.5]])
initial_guess = [ [2,5],[2,0],[6.5,0] ]


gf.cfig(15, fig_width=60, fig_height=50)        
plt.plot(X[:,0],X[:,1],'.', markeredgecolor = 'gray', color = 'gray', markersize = MS)
for (ind,val) in enumerate(initial_guess):
    plt.plot(val[0],val[1],'*', color = COLOR_MAT[ind],markersize = MS, markeredgecolor = COLOR_MAT[ind], linewidth = 2)
plt.xlim(-1,8)
plt.ylim(-4.5,6.5) 
plt.yticks([])
plt.xticks([]) 
xvals = plt.xlim()
yvals = plt.ylim()
plt.title('Data and Initial Guess')
gf.save_fig_file(savefigpath + 'ch5_' + 'kmeans_alg_init', dpi = tbk.dpi_save)
#plt.show()

for j in range(0,5):
    if j==0:
        old_centers = initial_guess
        centers, idx = find_centers(X,initial_guess)
    else:
        old_centers = centers
        centers, idx = find_centers(X,old_centers)
        
    gf.cfig(j, fig_width=60, fig_height=50)        
    for (ind,val) in enumerate(centers):
        x = [item[0] for item in idx[ind]]
        y = [item[1] for item in idx[ind]]
        
        if(j != 0):
            plt.title('Iteration ' + str(j))
            plt.plot(x,y,'.',markeredgecolor = COLOR_MAT[ind], color = COLOR_MAT[ind], markersize = MS)
            plt.plot(val[0],val[1],'*', color = COLOR_MAT[ind],markersize = MS, markeredgecolor = COLOR_MAT[ind], linewidth = 2)
            ax = plt.axes()
            plt.plot(old_centers[ind][0],old_centers[ind][1],'*',markeredgecolor = COLOR_MAT[ind],markersize = MS, color = 'none', linewidth = 2)
            plt.plot([old_centers[ind][0], val[0]], [old_centers[ind][1], val[1]], '--', color = COLOR_MAT[ind], linewidth = 1)
        else:
            plt.title('Initial Clusters')
            plt.plot(x,y,'.',markeredgecolor = COLOR_MAT[ind], color = COLOR_MAT[ind], markersize = MS)
            plt.plot(old_centers[ind][0],old_centers[ind][1],'*', color = COLOR_MAT[ind],markersize = MS, markeredgecolor = COLOR_MAT[ind], linewidth = 2)
            

    plt.yticks([])
    plt.xticks([])           
    plt.xlim(xvals)
    plt.ylim(yvals)      
    
    gf.save_fig_file(savefigpath + 'ch5_' + 'kmeans_alg' + str(j), dpi = tbk.dpi_save)
#    plt.show()
    
    