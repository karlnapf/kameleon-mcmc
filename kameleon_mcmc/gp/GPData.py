"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""

from numpy import hstack, arange, unique, float64
from numpy.core.function_base import linspace
from numpy.lib.npyio import loadtxt
from numpy.ma.core import sin, cos, asarray, zeros, array
from numpy.random import randn, randint, seed, rand
import os
from scipy.constants.constants import pi
import scipy.io


class GPData(object):
    @staticmethod
    def sample_circle_data(n, noise_level=0.025, offset=0.05, seed_init=None):
        if seed_init is not None:
            seed(seed_init)
        
        # decision surface for sampled data
        thetas = linspace(0, pi / 2, n)
        X = sin(thetas) * (1 + randn(n) * noise_level)
        Y = cos(thetas) * (1 + randn(n) * noise_level)
        
        # randomly select labels and distinguish data
        labels = randint(0, 2, n) * 2 - 1
        idx_a = labels > 0
        idx_b = labels < 0
        X[idx_a] *= (1. + offset)
        Y[idx_a] *= (1. + offset)
        X[idx_b] *= (1. - offset)
        Y[idx_b] *= (1. - offset)
        
        return asarray(zip(X, Y)), labels
    
    @staticmethod
    def sample_rectangle_data(n, noise_level=0.015, offset=0.05, seed_init=None):
        if seed_init is not None:
            seed(seed_init)
        
        # rectangle data
        a = rand(n / 2) * (1 - offset)
        b = rand(n / 2) * (1 - offset)
        data = zeros((n, 2))
        labels = zeros(n)
        for i in range(len(a)):
            labels[i] = 1.0 if rand() > 0.5 else -1.0
            data[i, 0] = a[i]
            data[i, 1] += labels[i] * offset + randn() * noise_level
            
        for i in range(len(b)):
            j = i + len(b)
            labels[j] = 1.0 if rand() > 0.5 else -1.0
            data[j, 1] = b[i]
            data[j, 0] += labels[j] * offset + randn() * noise_level
            
        return data, labels
    
    @staticmethod
    def get_pima_data():
        data_dir = os.sep.join(__file__.split(os.sep)[:-3] + ["data"])
        filename = data_dir + os.sep + "pima-indians-diabetes.data"
        data = loadtxt(filename, delimiter=",")
        
        # create labelling
        lab = data[:, -1]
        lab = array([1. if x == 1 else -1.0 for x in lab])
        
        # cut off labeling
        data = data[:, :-1]
        
        return data, lab
    
    @staticmethod
    def get_glass_data():
        data_dir = os.sep.join(__file__.split(os.sep)[:-3] + ["data"])
        filename = data_dir + os.sep + "glass.data"
        data = loadtxt(filename, delimiter=",")
        
        # create a binary "window glass" vs "non-window glass" labelling
        lab = data[:, -1]
        lab = array([1. if x <= 4 else -1.0 for x in lab])
        
        # cut off ids and labeling
        data = data[:, 1:-1]
        
        return data, lab

    @staticmethod
    def get_madelon_data():
        data_dir = os.sep.join(__file__.split(os.sep)[:-3] + ["data"])
        filename_dat = data_dir + os.sep + "madelon_train.data"
        filename_lab = data_dir + os.sep + "madelon_train.labels"
        data = loadtxt(filename_dat)
        lab = loadtxt(filename_lab)
        
        return data, lab

    @staticmethod
    def get_usps_data():
        data_dir = os.sep.join(__file__.split(os.sep)[:-3] + ["data"])
        filename = data_dir + os.sep + "usps3vs5.mat"
        mat = scipy.io.loadmat(filename)
        data = mat['usps3vs5_data']
        lab = array([float(x) for x in mat['usps3vs5_labels'][:, 0]])
        return data, lab
    
    @staticmethod
    def get_mushroom_data():
        data_dir = os.sep.join(__file__.split(os.sep)[:-3] + ["data"])
        filename = data_dir + os.sep + "agaricus-lepiota.data"
        f = open(filename)
        X = asarray(["".join(x.strip(os.linesep).split(",")) for x in f.readlines()], dtype="c")
        f.close()
        
        # first column is labels
        labels=asarray([+1. if X[i,0]=="e" else -1. for i in range(len(X))])
        
        # remove attribute eleven which contains loads of missing values
        remove_idx = 11
        indices = hstack((arange(remove_idx), arange(remove_idx + 1, X.shape[1])))
        X = X[:, indices[1:]]
        
        # generate a map of categorical to int
        cat2num={}
        U=unique(X)
        for num in range(len(U)):
            cat2num[U[num]]=num
        
        # produce an integer representation of the categorical data
        X_int=asarray([[cat2num[X[i,j]] for i in range(X.shape[0])] for j in range(X.shape[1])]).T
        
        return array(X_int, dtype=float64) ,labels
