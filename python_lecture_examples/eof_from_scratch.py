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
import scipy.io as sio
import numpy.linalg as LA
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon


import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ioff()
#%matplotlib inline
#%matplotlib qt

#%% input by user
#-------------------------------------
# which EOF do you want to plot?
eof_num = 1
#-------------------------------------

#%% DATA description

# This data is composed of 7 weather variables averaged over one year for
# each state in the US (thus, 50 states). 

# names of the different variables for each state
descriptor_names = ['temp','precip','% sun','sun hours','clear dys','humid AM','humid PM']

DATA = sio.loadmat('state_data_raw.mat')
Y = DATA['X']

# UNCOMMENT if you want to see what happens when you put-in RANDOM data
#Y = np.random.rand(np.shape(Y))


#%%

# UNCOMMENT if you want to see what happens when you put-in RANDOM data
#Y = np.random.rand(np.size(Y,0),np.size(Y,1));

# calculate anomalies from the state-mean (sample-mean) - call this "X"
Ymean = np.nanmean(Y,axis = 0)
X = Y - Ymean

# remove NaN
i = np.isnan(X)
X[i] = 0. # this step does not make a lot of sense!!

# standardize the data - call it "Xw" - why should we standardize this data?
Xstd = np.std(X,axis = 0)
Xw = X/Xstd

# don't standardize the data
#Xw = X;

#%% calculate EOF using temporal covariance matrix (covariance along the sampling dimension)

# calculate the temporal covariance matrix, dimensions should be [variable x variable]
C = 1./np.size(X,axis = 0)*np.dot(np.transpose(Xw),Xw)

# calculate eigenvalues and eigenvectors of C
lam, E = LA.eig(C)

# sort eigenvalues and vector by the largest to smallest eigenvalues
i = np.flipud(np.argsort(lam))
lam = lam[i]
E = E[:,i]

# convert eigenvalues to percent variance explained
pve = 100.*lam/np.sum(lam)

#%%

# take only one eigenvector, user specified by "eof_num" above
e1 = E[:,eof_num-1]

# calculate the the PC associated with the EOF of interest
z1 = np.dot(Xw,e1)

# standardize z1
z1 = (z1-np.mean(z1))/np.std(z1)

# calculate d1 for plotting in physical units, not standardized/weighted units,
# thus it uses the original "X" anomaly data
d1 = (1./np.size(X,axis=0))*np.dot(np.transpose(z1),X)

# calculate d1 for plotting in standardized/weighted units,
# thus it uses the "Xw" anomaly data
d1s = (1./np.size(Xw, axis = 0))*np.dot(np.transpose(z1),Xw)

#%% plot the results: EIGENVALUES

gf.cfig(1)
plt.plot(np.arange(1,np.size(pve)+1.),pve,'o-',linewidth = 2, color = 'black')

plt.xlim(0.5, 7.5)
plt.xlabel('eigenvalue position')
plt.ylabel('percent variance explained (%)')

# plot error bars according to North et al.abs
# here we will assume that all of the data is independent (is that a good assumption?)
# such that Nstar = N
Nstar = np.size(X,axis = 0)
eb = pve*np.sqrt(2./Nstar)
plt.errorbar(np.arange(1,np.size(pve)+1.),pve,yerr = eb/2, xerr = None, linewidth = 1, color = 'black')

gf.show_plot()
#plt.show()


#%% plot the results: EIGENVECTOR 1 standardized
gf.cfig(2)
plt.plot(d1s,'s-k',linewidth = 2, label = 'd1s')
plt.plot(e1,'s-r',linewidth = 2, label = 'e1')

plt.xticks(np.arange(len(descriptor_names)),descriptor_names)
plt.xlim(-0.5, 6.5)

plt.legend(loc = 'upper right')
gf.plot_zero_lines()
plt.ylabel('sigma')
plt.title('d standardized')

gf.show_plot()
#plt.show()


#%% plot the results: EIGENVECTOR 1 not-standardized (does not make sense for this problem though)
gf.cfig(3)
plt.plot(d1,'s-k',linewidth = 2, label = 'd1')

plt.xticks(np.arange(len(descriptor_names)),descriptor_names)
gf.plot_zero_lines()
plt.legend(loc = 'upper right')
plt.ylabel('physical units')
plt.title('d in physical units')
plt.xlim(-0.5, 6.5)


gf.show_plot()
#plt.show()


#%% plot the PC1 as a US map

gf.cfig(4)

# create the map
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
m.readshapefile('st99_d00', name='states', drawbounds=True)

# collect the state names from the shapefile attributes so we can
# look up the shape obect for a state by it's name
state_names = []
for shape_dict in m.states_info:
    state_names.append(shape_dict['NAME'])

state_names_list = sorted(list(set(state_names)))
state_names_list.pop(state_names_list.index('District of Columbia'))
state_names_list.pop(state_names_list.index('Puerto Rico'))

ax = plt.gca() # get current axes instance

colors={}
cmap = plt.cm.get_cmap('seismic')
vmin = -3.
vmax = 3.

for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        z = z1[state_names_list.index(statename)]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        colors[statename] = cmap((z-vmin)/(vmax-vmin))[:3]

for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if state_names[nshape] not in ['District of Columbia','Puerto Rico']:
        color = colors[state_names[nshape]]
        
        if state_names[nshape] == 'Alaska':
        # Alaska is too big. Scale it down to 35% first, then transate it. 
            seg = list(map(lambda (x,y): (0.35*x + 1300000, 0.32*y-1300000), seg))
        if state_names[nshape] == 'Hawaii':
            seg = list(map(lambda (x,y): (.45*x + 5100000, .7*y-900000), seg))

        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)

        
plt.title('PC' + str(eof_num) + ' value')

gf.show_plot()
#plt.show()






















