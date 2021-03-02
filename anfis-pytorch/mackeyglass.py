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

def data(partition):
    '''
        Gera os quatro instantes de dados e saida
    '''
    
    # loading data set
    data = loadmat('mg.mat')
    data = pd.DataFrame(data['x'])
    lendata = len(data)
    
    # 80/20
    data = data.to_numpy()
    limit = int(0.8 * lendata)
    
    if partition == 'train':
        const = limit - 18
        cycle = range(18, limit)
    if partition == 'test':
        const = lendata - limit - 6
        cycle = range(limit, lendata - 6)
    
    # quatro instantes e futuro
    x = torch.zeros((const, 4))
    y = torch.zeros((const, 1))
    for idx, t in enumerate(cycle):
        x[idx, 0] = torch.tensor(data[t - 18])
        x[idx, 1] = torch.tensor(data[t - 12])
        x[idx, 2] = torch.tensor(data[t - 6])
        x[idx, 3] = torch.tensor(data[t - 0])
        y[idx, 0] = torch.tensor(data[t + 6])
    
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size = 1024, shuffle = True)
        
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
    x = torch.zeros((limit - 18, 4))
    y = torch.zeros((limit - 18, 1))
    for idx, t in enumerate(range(18, limit)):
        x[idx, 0] = torch.tensor(data[t - 18])
        x[idx, 1] = torch.tensor(data[t - 12])
        x[idx, 2] = torch.tensor(data[t - 6])
        x[idx, 3] = torch.tensor(data[t - 0])
        y[idx, 0] = torch.tensor(data[t + 6])
    
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
    x = torch.zeros((lendata - limit - 6, 4))
    y = torch.zeros((lendata - limit - 6, 1))
    for idx, t in enumerate(range(limit, lendata - 6)):
        x[idx, 0] = torch.tensor(data[t - 18])
        x[idx, 1] = torch.tensor(data[t - 12])
        x[idx, 2] = torch.tensor(data[t - 6])
        x[idx, 3] = torch.tensor(data[t - 0])
        y[idx, 0] = torch.tensor(data[t + 6])
    
    td = TensorDataset(x, y)
    return DataLoader(td)

def model():
    invardefs = [
        # ainda n escolhi os centros e sigma
        ('x(t-18)', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('x(t-12)', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('x(t-6)', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('x(t)', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2]))
        ]
    outvars = ['ys']
    
    model = anfis.AnfisNet('mackey-glass', invardefs, outvars)
    return model

if __name__ == '__main__':
    model = model()
    train_data = data(partition = 'train')
    train_anfis(model, data = train_data, epochs = 20, show_plots = True)
    test_data = data(partition = 'test')
    test_anfis(model, data = test_data, show_plots = True)
