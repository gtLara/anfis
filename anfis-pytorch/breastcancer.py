#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: Examples from Jang's paper
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

from matplotlib import pyplot as plt
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import GaussMembFunc, make_gauss_mfs
from experimental import train_anfis, test_anfis

dtype = torch.float

def data(partition):
    # loading data set
    data = pd.read_csv('data/breastcancer.csv').dropna().drop(columns = ['id'])
    
    # 70 / 30
    data = data.to_numpy()
    length = len(data)
    idx = np.arange(length)
    np.random.shuffle(idx)
    
    eta = 0.7
    if partition == 'train':
        idx = idx[:int(eta * length)]
    if partition == 'test':
        idx = idx[int(eta * length):]
    
    # x, y
    x = data[idx, 1:data.shape[1]]
    y = data[idx, 0]
    y[y == 'B'] = -1
    y[y == 'M'] = 1
    
    # torch
    x = torch.tensor(np.array(x, dtype = np.float))
    y = torch.tensor(np.array(y, dtype = np.float))
    
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size = 1024, shuffle = True)
        
def model():
    invardefs = [
        ('radius_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('texture_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('perimeter_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('area_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('smoothness_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('compactness_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concavity_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concave points_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('symmetry_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('fractal_dimension_mean', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        
        ('radius_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('texture_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('perimeter_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('area_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('smoothness_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('compactness_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concavity_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concave points_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('symmetry_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('fractal_dimension_se', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        
        ('radius_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('texture_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('perimeter_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('area_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('smoothness_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('compactness_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concavity_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('concave points_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('symmetry_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('fractal_dimension_worst', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2]))
        
        ]
    outvars = ['diagnosis']
    
    model = anfis.AnfisNet('breast-cancer', invardefs, outvars)
    return model

if __name__ == '__main__':
    model = model()
    train_data = data(partition = 'train')
    train_anfis(model, data = train_data, epochs = 20, show_plots = True)
    test_data = data(partition = 'test')
    test_anfis(model, data = test_data, show_plots = True)
