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
    # loading data set
    data = pd.read_csv('data/iris.csv').dropna().drop(columns = ['Id'])
    
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
    
    # # x, y
    # x = data[idx, 0:data.shape[1] - 1]
    # y = data[idx, data.shape[1]- 1]
    # y[y != 'Iris-setosa'] = -1
    # y[y == 'Iris-setosa'] = 1
    
    # # torch
    # x = torch.tensor(np.array(x, dtype = np.float))
    # y = torch.tensor(np.array(y, dtype = np.float))
    
    # copiei do anterior pq acima deu problema de float / double
    data[data[:, data.shape[1] - 1] != 'Iris-setosa', data.shape[1]- 1] = 1
    data[data[:, data.shape[1] - 1] == 'Iris-setosa', data.shape[1]- 1] = -1
    
    x = torch.zeros((len(idx), data.shape[1] - 1))
    y = torch.zeros((len(idx), 1))
    for i, index in enumerate(idx):
        x[i, 0] = torch.tensor(data[index, 0])
        x[i, 1] = torch.tensor(data[index, 1])
        x[i, 2] = torch.tensor(data[index, 2])
        x[i, 3] = torch.tensor(data[index, 3])
        y[i, 0] = torch.tensor(data[index, 4])
    
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size = 1024, shuffle = True)
        
def model():
    invardefs = [
        ('SepalLengthCm', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('SepalWidthCm', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('PetalLengthCm', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2])),
        ('PetalWidthCm', make_gauss_mfs(sigma = 0.2, mu_list = [0.4, 1.2]))
        ]
    outvars = ['Species']
    
    model = anfis.AnfisNet('iris', invardefs, outvars)
    return model

if __name__ == '__main__':
    model = model()
    train_data = data(partition = 'train')
    train_anfis(model, data = train_data, epochs = 20, show_plots = True)
    test_data = data(partition = 'test')
    test_anfis(model, data = test_data, show_plots = True)
