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

##### Problema 3: Previsao de uma serie temporal #####

def training_data(batch_size = 1024):
    '''
        Gera os quatro instantes de dados e saida
    '''
    # loading data set
    data = loadmat('mg.mat')
    data = pd.DataFrame(data['x'])
    lendata = len(data)
    
    # 80%
    data = data.to_numpy()
    limit = int(0.8 * lendata)
    
    # quatro instantes e futuro
    train = np.zeros(shape = [limit - 18, 5])
    for idx, t in enumerate(range(18, limit)):
        train[idx, 0] = data[t - 18]
        train[idx, 1] = data[t - 12]
        train[idx, 2] = data[t - 6]
        train[idx, 3] = data[t - 0]
        train[idx, 4] = data[t + 6]
    
    # return train
    x = torch.tensor(train[:, 0:4])
    y = torch.tensor(train[:, 4])
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size = batch_size, shuffle = True)

def testing_data():
    '''
        Gera os quatro instantes de dados e saida
    '''
    # loading data set
    data = loadmat('mg.mat')
    data = pd.DataFrame(data['x'])
    lendata = len(data)
    
    # 20%
    data = data.to_numpy()
    limit = int(0.8 * lendata)
    
    # quatro instantes e futuro
    test = np.zeros(shape = [lendata - limit - 6, 5])
    for idx, t in enumerate(range(limit, lendata - 6)):
        test[idx, 0] = data[t - 18]
        test[idx, 1] = data[t - 12]
        test[idx, 2] = data[t - 6]
        test[idx, 3] = data[t - 0]
        test[idx, 4] = data[t + 6]
    
    # return test
    x = torch.tensor(test[:, 0:4])
    y = torch.tensor(test[:, 4])
    td = TensorDataset(x, y)
    return DataLoader(td)

def model():
    invardefs = [
        # ainda n escolhi os centros e sigma
        ('x(t-18)', make_gauss_mfs(sigma = 1.2, mu_list = [0, 1])),
        ('x(t-12)', make_gauss_mfs(sigma = 1.2, mu_list = [0, 1])),
        ('x(t-6)', make_gauss_mfs(sigma = 1.2, mu_list = [0, 1])),
        ('x(t)', make_gauss_mfs(sigma = 1.2, mu_list = [0, 1]))
        ]
    outvars = ['ys']
    
    model = anfis.AnfisNet('mackey-glass', invardefs, outvars)
    return model

if __name__ == '__main__':
    model = model()
    train_data = training_data(batch_size = 100)
    train_anfis(model, data = train_data, epochs = 20, show_plots = True)
    test_data = testing_data()
    test_anfis(model, data = test_data, show_plots = True)
