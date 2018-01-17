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
#from random import randint
import scipy.io as sio
import h5py
#import numpy.ma as ma
#import csv
from mpl_toolkits.basemap import Basemap, cm

import general_functions as gf
reload(gf)
gf.add_parent_dir_to_path()

#.............................................
# PLOTTING COMMANDS
#.............................................
gf.cc()
plt.ion()
#%matplotlib inline
#%matplotlib qt



#%% START CODE

DATA = h5py.File('era_interim_eke_nh.mat')

X = np.array(DATA['EKE'])
X = np.swapaxes(X,2,0)
LAT = np.array(DATA['LAT'])[0,:]
LONG = np.array(DATA['LONG'])[0,:]


# get data so there is no gap at 360E
X = np.insert(X, 0, values=X[:,:,0], axis=2)
LONG = np.insert(LONG,0,values = 0.)

X = np.insert(X, -1, values=X[:,:,-1], axis=2)
LONG = np.insert(LONG,-1,values = LONG[-1])
LONG[-1] = 360


del DATA

#%% get January days only

itime = []
for j in np.arange(0,32,1):
    m = np.arange(j*365,j*365+31,1)
    itime = np.append(itime,m)

    
X = X[np.ndarray.astype(itime,int),:,:]



#%% get score of composite
N = 30
Xmean = np.nanmean(X,axis=0)
sigma = np.std(X,axis=0)

#rand = np.random.randint(0,X.shape[0],N)
irand = np.arange(0,N)


Xcomposite = np.nanmean(X[irand,:,:],axis = 0)

P = np.empty((X.shape[1],X.shape[2]))

for ilat,vallat in enumerate(LAT):
    for ilon,vallon in enumerate(LONG):
        
        zscore = (Xcomposite[ilat,ilon] - Xmean[ilat,ilon])/(sigma[ilat,ilon]/np.sqrt(N))
        P[ilat,ilon] = stats.norm.sf(np.abs(zscore))*2.
 

  
#%% plot initial results

# lon_0 is central longitude of projection.
# resolution = 'c' means use crude resolution coastlines.
gf.cfig(1, fig_width = 100, fig_height = 100)

m = Basemap(projection='nplaea',lon_0=180, boundinglat = 10, resolution='l')
m.drawcoastlines(color = 'gray')

# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,120.,30.))
#m.drawmeridians(np.arange(0.,360.,60.))
lon_2d,lat_2d = np.meshgrid(LONG,LAT)
x,y = m(lon_2d,lat_2d)
cs = plt.pcolor(x,y,Xcomposite-Xmean, cmap = cm.GMT_haxby)

for ilat,vallat in enumerate(LAT):
    for ilon,vallon in enumerate(LONG):
        if(P[ilat,ilon]<0.05):
            x,y = m(vallon,vallat)
            m.plot(x,y,'o',markersize = 3, color = 'blue')


cb = plt.colorbar(cs,orientation='horizontal',fraction=0.046, pad=0.04)
cb.set_label(label = 'eddy kinetic energy ($m^2/s^2$)', size = 14)
cb.ax.tick_params(labelsize=12)
plt.clim(-60,60)

plt.title('Composite of ' + str(N) + ' random January days')

gf.show_plot()


#%% calculate wilks significance

alpha = 0.15

Pvals = P.flatten()
Pvals = np.sort(Pvals)

x = np.arange(1,len(Pvals)+1,1)
x = x.astype(float)

gf.cfig(2)
plt.plot(x,Pvals,'.g', markersize = 6, linewidth = 1.5)

y = (x/len(x))*alpha
plt.plot(x,y,'-',color = 'black', linewidth = 1.5)

plt.xlim(0,1200)
plt.ylim(0,.12)

plt.xlabel('index')
plt.ylabel('p-value')

d = Pvals - y
k = np.where(d>0)[0][0] - 1 + 1

plt.plot([k,k],[0,1],'--',color = 'gray')

plt.title('sorted p-values: $p_{crit} = $' + str(gf.round_to_nearest(Pvals[k-1],.001)))

gf.show_plot() 
#%% plot wilks significance
     
gf.fig(1)
for ilat,vallat in enumerate(LAT):
    for ilon,vallon in enumerate(LONG):
        if(P[ilat,ilon]<Pvals[k-1]):
            x,y = m(vallon,vallat)
            m.plot(x,y,'o',markersize = 6, color = 'limegreen')
