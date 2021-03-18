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

from cmenas import cmenas
from norm import normalize, denormalize

dtype = torch.float

def data(partition):
    # loading data set
    data = pd.read_csv('data/breastcancer.csv').dropna().drop(columns = ['id'])
    # print(data.columns)
    data = data.drop(columns = [
                                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                                'smoothness_worst', 'compactness_worst', 'concavity_worst',
                                'symmetry_worst', 'fractal_dimension_worst', 'concave points_worst',
                                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                                'smoothness_se', 'compactness_se', 'concavity_se',
                                'concave points_se', 'symmetry_se', 'fractal_dimension_se'
                               ])
    # 70 / 30
    data = data.to_numpy()
    # data = data[:300, :]
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
    data[idx, 0] = y

    # torch
    x = torch.tensor(np.array(x, dtype = np.float))
    y = torch.tensor(np.array(y, dtype = np.float))

    # copiei do anterior pq acima deu problema de float / double

    x = torch.zeros((len(idx), data.shape[1] - 1))
    y = torch.zeros((len(idx), 1))
    for i, index in enumerate(idx):
        for n_feature in range(data.shape[1] - 1):
            x[i, n_feature] = torch.tensor(data[index, n_feature + 1])

        y[i, 0] = torch.tensor(data[index, 0])

    td = TensorDataset(x, y)
    return DataLoader(td, batch_size = 1024, shuffle = True)

def model(data, n_rules):

    # numpy and norm
    x, y = data.dataset.tensors
    y = y.float()
    x = x.numpy()
    x, minimum, maximum = normalize(data = x)
    x = x.astype(float)

    # cmeans
    # como o numero de entradas é constante n de regras = n de funcoes de
    # pertinencia = numero de centros

    modelo = cmenas(k = n_rules)
    modelo.train(data = x, MAX = 15, tol = 1e-2)
    centros = modelo.C

    # denorm
    centros = denormalize(data = centros, m = minimum, M = maximum)

    names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean' 'conc_mean',
             'points_mean', 'symmetry_mean', 'fractal_dimension_mean']
            # 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            # 'smoothness_se', 'compactness_se', 'concavity_se' 'conc_se',
            #  'points_se', 'symmetry_se', 'fractal_se', 'd1']

    def mk_var(name, centros, i):
        return (name, make_gauss_mfs(1, [centros[n, i] for n in range(n_rules)]))

    invardefs = [mk_var(name, centros, i) for i, name in enumerate(names)]

    outvars = ['diagnosis']

    model = anfis.AnfisNet('breast-cancer', invardefs, outvars)
    return model

if __name__ == '__main__':
    train_data = data(partition = 'train')
    modelo = model(train_data, 1)
    train_anfis(modelo, data = train_data, epochs = 20, show_plots = False)
    test_data = data(partition = 'test')
    test_anfis(modelo, data = test_data, show_plots = True)
