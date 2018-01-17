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
from mpl_toolkits.basemap import Basemap

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

#.............................................
# INTIAL SETTINGS


#%% get data together

filename = 'Temperature_2meters_monthly_ERAInterim.csv'
X = np.genfromtxt(filename, delimiter = ',')
X = X[:,0:-1:4]

filename = 'LAT_LONG.csv'
LAT_LONG = np.genfromtxt(filename, delimiter = ',')
LAT = LAT_LONG[0:-1:4,0]
LONG = LAT_LONG[0:-1:4,1]

#%%
#======== INPUTS ===========
eof_number = np.arange(0,5)
month_to_plot = 100
#===========================

#spatial dimension
C = 1./np.size(X,axis = 1)*(np.dot(X,np.transpose(X)))

# calculate eigenvalues and eigenvectors of C
lam, Z = LA.eig(C)
E = np.dot(np.transpose(Z),X)

# sort eigenvalues and vector by the largest to smallest eigenvalues
i = np.flipud(np.argsort(lam))
lam = lam[i]
E = E[i,:]

# convert eigenvalues to percent variance explained
pve = 100.*lam/np.sum(lam)

# reduce E and Z and reconstruc X

# retain only certain eofs
Z = Z[:,eof_number]
E = E[eof_number,:]

# reconstruct X
Xrecon = np.dot(Z,E)

# plot results: EIGENVALUES
gf.cfig(1)
plt.plot(np.arange(1,np.size(pve)+1.),pve,'o-',linewidth = 2, color = 'black')

plt.plot([np.max(eof_number)+1.5,np.max(eof_number)+1.5],[0,20],'--k')

plt.title('Variance Retained = ' + str(np.round(np.sum(pve[eof_number]))) + '%')

plt.xlim(0.5, 110.5)
plt.ylim(0,17.)
plt.xlabel('eigenvalue position')
plt.ylabel('percent variance explained (%)')

# plot error bars according to North et al.abs
# here we will assume that all of the data is independent (is that a good assumption?)
# such that Nstar = N
Nstar = np.size(X,axis = 1)
eb = pve*np.sqrt(2./Nstar)
plt.errorbar(np.arange(1,np.size(pve)+1.),pve,yerr = eb/2, xerr = None, linewidth = 1, color = 'black')

plt.show()
#gf.show_plot()


# plot results
LATu = np.unique(LAT)
LONGu = np.unique(LONG)

P_PLOT = np.empty((np.size(LATu),np.size(LONGu)))
P_PLOT_orig = np.empty((np.size(LATu),np.size(LONGu)))

for ilat,lat in enumerate(LATu):
    for ilon,lon in enumerate(LONGu):
        
        j = np.where(np.logical_and(LAT==lat,LONG==lon))
        P_PLOT[ilat,ilon] = Xrecon[month_to_plot-1,j]
        P_PLOT_orig[ilat,ilon] = X[month_to_plot-1,j]

lons, lats = np.meshgrid(LONGu,LATu)

fig = gf.cfig(3)

#------------------- RAW DATA -------------------------

#fig = gf.cfig(4)
#ax = fig.add_axes([0.05,0.05,0.9,0.9])
# create Basemap instance.
# coastlines not used, so resolution set to None to skip
# continent processing (this speeds things up a bit)
ax = fig.add_subplot(211)
m = Basemap(projection='kav7',lon_0=0,resolution='c')
m.drawcoastlines(linewidth = 1)
im1 = m.pcolormesh(lons,lats,P_PLOT_orig,shading='flat',cmap=plt.cm.RdBu,latlon=True)

#cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
plt.clim(-8,8)
ax.set_title('Month = ' + str(month_to_plot) + '\nRaw data', fontsize = 12)

#plt.show()
#------------------- SMOOTHED DATA -------------------------

ax = fig.add_subplot(212)
#ax = fig.add_axes([0.05,0.05,0.9,0.9])
# create Basemap instance.
# coastlines not used, so resolution set to None to skip
# continent processing (this speeds things up a bit)
m = Basemap(projection='kav7',lon_0=0,resolution='c')
m.drawcoastlines(linewidth = 1)
im1 = m.pcolormesh(lons,lats,P_PLOT,shading='flat',cmap=plt.cm.RdBu,latlon=True)

#cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
plt.clim(-8,8)
ax.set_title('Month = ' + str(month_to_plot) + '\nSmoothed by retaining ' + str(np.max(eof_number)+1) + ' of ' + str(np.size(lam)) + ' EOFs\n variance retained = ' + str(np.round(np.sum(pve[eof_number]))) + '%', fontsize = 12)
#plt.show()

gf.show_plot()


