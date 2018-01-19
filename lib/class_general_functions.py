# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

#%% textbook directories

fig_dir = '/Users/eabarnes/GoogleDrive/WORK/TEACHING/ATS655/S18/python_code/figures'

#%% SPECIFICIATIONS FOR TEXTBOOK FIGURES

# colors for plots
#global clr1, clr2

# original color scheme
clr0, clr1, clr2, clr3 = 'black', 'royalblue', 'darkorange', 'mediumorchid'

#clr0, clr1, clr2, clr3, clr4 = 'black', (0/256.,255/256.,255/256.), (255/256.,224/256.,102/256.), (217/256.,209/256.,255/256.), (196/256.,140/256.,255/256.)
#clr1, clr2, clr3 = (128/256.,171/256.,255/256.), (255/256.,224/256.,102/256.), (196/256.,140/256.,255/256.)

# yellow: (255/256.,255/256.,0/256.)
# bluish: (0/256.,255/256.,255/256.)
# purple: (196/256.,140/256.,255/256.)
# green: (82/256.,255/256.,25/256.)
# darker blue: (128/256.,171/256.,255/256.)

dpi_save = 600.
dpi_mpl = 250.

mpl.rcParams['figure.dpi']= dpi_mpl

fig_text_size = 13.
fig_title_size = fig_text_size*1.5

lw = 2.0
markersize = 10.

fig_width = 45.0
fig_height = fig_width/1.5

#%%
def pica_convert(len_pica):

    len_in = len_pica/6.
    return len_in


def fig(fig_num=None, fig_width=fig_width, fig_height=fig_height):

    # convert picas to inches    
    # AMS takes width = 19, 33 or 39 picas
    fig_width_in = pica_convert(fig_width)
    fig_height_in = pica_convert(fig_height)    
    
    # make new figure
    ax_f = plt.figure(num=fig_num, figsize = (fig_width_in, fig_height_in))    
    ax = plt.gca()
    
#    # set default values for figures
    plt.rc('lines', linewidth=lw)    
#    plt.rc('text', usetex=True)
    plt.rc('text', usetex=False)
    plt.rc('font', size=fig_text_size, weight='normal',family='sans-serif')
    plt.rc('axes',titlesize=fig_text_size,titleweight='bold')
    plt.rc('axes', prop_cycle = cycler('color',['black','#A2132F','#0072BE','#EDB120','#D95319','#7E2F8E','#77AC30','#4DBEEE']))        
    
    return ax_f, ax
    
def cfig(fig_num=None, fig_width=fig_width, fig_height=fig_height):

    # convert picas to inches    
    # AMS takes width = 19, 33 or 39 picas
    fig_width_in = pica_convert(fig_width)
    fig_height_in = pica_convert(fig_height)    
    
    # make new figure
    ax_f = plt.figure(num=fig_num, figsize = (fig_width_in, fig_height_in))    
    plt.clf()   
    ax = plt.gca()
   
    
#    # set default values for figures
    plt.rc('lines', linewidth=lw)    
#    plt.rc('text', usetex=True)
    plt.rc('text', usetex=False)
    plt.rc('font', size=fig_text_size, weight='normal',family='sans-serif')
    plt.rc('axes',titlesize=fig_text_size,titleweight='bold')
    plt.rc('axes', prop_cycle = cycler('color',['black','#A2132F','#0072BE','#EDB120','#D95319','#7E2F8E','#77AC30','#4DBEEE']))        
    
    return ax_f, ax