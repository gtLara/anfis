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

from sklearn.model_selection import KFold

dtype = torch.float

def kfold_data(idx, k=10): #assume indices preprocessados

    kfold = KFold(k)
    folds = []

    for i, (train, test) in enumerate(kfold.split(idx)):

        train_data, _ = data(forced_idx = train)
        test_data, _ = data(forced_idx = test)
        folds.append({'train':train_data, 'test':test_data})

    return folds

def data(partition=None, matlab=False, forced_idx=None):
    # loading data set
    data = pd.read_csv('data/wdbc.data')
    data = data.drop(columns = data.columns[0])
    # 70 / 30
    data = data.to_numpy()
    length = len(data)
    idx = np.arange(length)
    np.random.shuffle(idx)

    if partition is not None:
        idx = np.arange(length)
        np.random.seed(4224)
        np.random.shuffle(idx)

        eta = 0.7

        if partition == 'train':
            idx = idx[:int(eta * length)]
        if partition == 'test':
            idx = idx[int(eta * length):]

    if forced_idx is not None:
        idx = forced_idx

    # x, y
    if matlab:
        x = data[idx, 1:data.shape[1]]
        y = data[idx, 0]
        y[y == 'B'] = -1
        y[y == 'M'] = 1
        data[idx, 0] = y
    else:
        x = data[idx, :-1]
        y = data[idx, -1]
        y[y == 4] = -1
        y[y == 2] = 1
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
    return DataLoader(td, batch_size = 1024, shuffle = True), idx

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
             'points_mean', 'symmetry_mean']

    def mk_var(name, centros, i):
        return (name, make_gauss_mfs(3, [centros[n, i] for n in range(n_rules)]))

    invardefs = [mk_var(name, centros, i) for i, name in enumerate(names)]

    outvars = ['diagnosis']

    model = anfis.AnfisNet('breast-cancer', invardefs, outvars)
    return model

if __name__ == '__main__':
    # train_data, _ = data(partition = 'train')
    # modelo = model(train_data, 1)
    # train_anfis(modelo, data = train_data, epochs = 20, show_plots = False)
    # test_data, _ = data(partition = 'test')
    # _, _, error = test_anfis(modelo, data = test_data, show_plots = True)


    #######

    train_data, idx = data(partition = 'train')
    folds = kfold_data(idx)

    rule_range = range(1)
    fold_eval = np.zeros((len(rule_range), len(folds)))

    for r, n_rules in enumerate(rule_range):
        for f, fold in enumerate(folds):
            fold_train_data = fold['train']
            fold_test_data = fold['test']
            anfis_model = model(fold_train_data, n_rules + 1)
            train_anfis(anfis_model, data = fold_train_data, epochs = 20, show_plots = False)
            _, _, perc_loss = test_anfis(anfis_model, data = fold_test_data, show_plots = False)
            fold_eval[r, f] = perc_loss

    # particao de teste é avaliada com os parametros determinados

    best_n_rule = np.argmax(np.mean(fold_eval, axis=1)) + 1

    anfis_model = model(train_data, best_n_rule)
    train_anfis(anfis_model, data = train_data, epochs = 20, show_plots = False)
    test_data, _ = data(partition = 'test')
    _, _, error = test_anfis(anfis_model, data = test_data, show_plots = True)

    print('erro percentual ={:.2f}%'.format(error))
