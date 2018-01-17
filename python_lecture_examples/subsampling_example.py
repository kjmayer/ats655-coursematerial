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
#import numpy.ma as ma
#import csv

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
DATA = sio.loadmat('subsampling_example_Z500_August.mat')
X = DATA['X'][:,0]
LAT = DATA['LAT'][0][0]
LONG = DATA['LONG'][0][0]

sample_length = 20
P = np.empty(2500)

for j, val in enumerate(P):
    ir = stats.randint.rvs(0,len(X)-1,size = sample_length)
    P[j] = np.nanmean(X[ir])

#%%
gf.cfig(1)
h, bins = np.histogram(X,20)
plt.plot(bins[:-1],h)

plt.xlabel('geopotential height (m)')
plt.ylabel('frequency')

plt.plot([np.mean(X), np.mean(X)],[0., 150],'--', color = 'yellow')

plt.title('August Z500 at ' + str(np.round(LAT)) + '$^o$N, ' + str(round(LONG)) + '$^o$E')

Z = (X-np.mean(X))/np.std(X)
plt.text(5700, 150, 'skewness = ' + str(gf.round_to_nearest(stats.skew(Z[:]), 0.01))) 

plt.xlim(5700, 6000)


gf.show_plot()
    
#%%

#mp = np.mean(P)
mp = 0.

gf.cfig(2)
h, bins = np.histogram(P-mp,20)
plt.plot(bins[:-1],h, color = 'blue', label = 'means')
plt.plot((np.mean(X), np.mean(X)),(0., 400),'--', color = 'yellow', label = 'mean of sample means');

a1 = np.percentile(P-mp,2.5)
a2 = np.percentile(P-mp,100.-2.5)

plt.plot((a1,a1),(0,400),'--',color = 'red', linewidth = 2, label = '95% confidence bounds')
plt.plot((a2,a2),(0,400),'--',color = 'red', linewidth = 2)

t_inc = (stats.t.ppf(0.975, sample_length - 1))*np.std(X)/np.sqrt(sample_length-1)

plt.plot(np.ones((2,))*(np.mean(X)-t_inc), (0,400), '--',color = 'black', label = 'critical t')
plt.plot(np.ones((2,))*(np.mean(X)+t_inc), (0,400), '--',color = 'black')

plt.legend(fontsize = 12)

plt.xlabel('geopotential height (m)')
plt.ylabel('frequency')

plt.title('distribution of random sample means of ' + str(sample_length) + ' days')
plt.xlim(5700, 6000)


gf.show_plot()

#%%

#clear all
#close all
#
#% load('/Users/eabarnes/Documents/RESEARCH/2012/WAVE_HEIGHTS/ERA_INTERIM/z500_daily.mat');
#% disp('only use August days');
#% itime = find(TIME(:,3)==8);
#% X = X(itime,:,:);
#% 
#% ilat = find(isbetween(LAT,37,38));
#% ilong = find(isbetween(LONG,275,276));
#% 
#% X = X(:,ilat,ilong);
#% LAT = LAT(ilat);
#% LONG = LONG(ilong);
#
#%save('subsampling_example_Z500_August','X','LAT','LONG');
#
#load('subsampling_example_Z500_August.mat');
#
#%%
#
#sample_length = 20;
#
#P = nan(2500,1,1);
#
#for iloop = 1:size(P,1)
#       
#    ir = randi(size(X,1),[sample_length 1]);
#    
#    P(iloop,:,:) = nanmean(X(ir),1);
#    
#end
#
#
#%%
#
#
#cfig(2);
#hist(X,20);
#xlabel('geopotential height (m)');
#ylabel('frequency');
#
#plot(ones(1,2)*mean(X),[0 150],'--y');
#
#title(['August Z500 at ' num2str(round(LAT)) '^oN, ' num2str(round(LONG)) '^oE']);
#Z = (X-mean(X))/std(X);
#add_figure_text(['skewness = ' num2str(round(skewness(Z)*100)/100)],2);
#
#xlim([5700 6000]);
#
#mp = mean(P);
#mp = 0;
#
#
#cfig(1);
#
#hist(P-mp,20);
#plot(ones(1,2)*mean(X),[0 400],'--y','LineWidth',2);
#
#a1 = percentile(P-mp,2.5);
#a2 = percentile(P-mp,100-2.5);
#
#plot([a1 a1],[0 400],'--r','LineWidth',2);
#h=plot([a2 a2],[0 400],'--r','LineWidth',2);
#not_in_legend(h);
#
#t_inc = tinv(0.975,sample_length-1)*std(X)/(sqrt(sample_length)-1);
#
#plot([mean(X) - t_inc]*ones(1,2),[0 400],'--k','LineWidth',2);
#plot([mean(X) + t_inc]*ones(1,2),[0 400],'--k','LineWidth',2);
#
#a=legend('means','mean of sample means','95% confidence bounds','t-test');
#legend boxoff
#set(a,'Location','NorthWest');
#
#xlabel('geopotential height (m)');
#ylabel('frequency');
#
#title(['distribution of random sample means of ' num2str(sample_length) ' days']);
#xlim([5700 6000]);
#
#save_print_plot_width_height('../FIGURES/bootstrap_ex1_37N_275E',12,8);