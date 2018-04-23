# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 14:44:44 2017

@author: lwang
"""
import pylab as pl
import numpy as np
from scipy.stats.stats import pearsonr 

def plot_single(x, x_label, x_subs):
    
    fig = pl.figure()
    ax = fig.add_subplot(221)
    ax.scatter(x[0,:], x[1,:])
    ax.plot(ax.get_xlim(), ax.get_xlim(),'k')
    ax.set_xlabel('Low band ' + x_label)
    ax.set_ylabel('Mid band ' + x_label)
    ax = fig.add_subplot(222)
    ax.scatter(x[0,:], x[2,:])
    ax.plot(ax.get_xlim(), ax.get_xlim(),'k')
    ax.set_xlabel('Low band ' + x_label)
    ax.set_ylabel('High band ' + x_label)
    ax = fig.add_subplot(223)
    ax.scatter(x[1,:], x[2,:])
    ax.plot(ax.get_xlim(), ax.get_xlim(),'k')
    ax.set_xlabel('Mid band ' + x_label)
    ax.set_ylabel('High band ' + x_label)
    
def plot_pairwithSpeech(x1, x1_label, x1_subs, x2, x2_label, x2_subs):
    
   
    comm_subs = np.intersect1d(x1_subs, x2_subs)
    subs_x1_ind = np.in1d(x1_subs, comm_subs)
    x1_comm = x1[:,subs_x1_ind]
    subs_x2_ind = np.in1d(x2_subs, comm_subs)
    x2_comm = x2[subs_x2_ind]
    fig = pl.figure()
    ax = fig.add_subplot(221)
    ax.scatter(x1_comm[0,:], x2_comm)
    ax.set_xlabel('Low band ' + x1_label)
    ax.set_ylabel(x2_label)
    ax = fig.add_subplot(222)
    ax.scatter(x1_comm[1,:], x2_comm)
    ax.set_xlabel('Mid band ' + x1_label)
    ax.set_ylabel(x2_label)
    ax = fig.add_subplot(223)
    ax.scatter(x1_comm[2,:], x2_comm)
    ax.set_xlabel('High band ' + x1_label)
    ax.set_ylabel(x2_label)
    
    print pearsonr(x1_comm[0,:], x2_comm)
    print pearsonr(x1_comm[1,:], x2_comm)
    print pearsonr(x1_comm[2,:], x2_comm)