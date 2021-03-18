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

def data(partition=None, forced_idx=None):
    # loading data set
    data = pd.read_csv('data/iris.csv').dropna().drop(columns = ['Id'])


    data = data.to_numpy()
    length = len(data)

    # 70 / 30

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
    return DataLoader(td, batch_size = 1024, shuffle = True), idx

def model(data, n_rules):

    # numpy and norm
    x, y = data.dataset.tensors
    x = x.numpy()
    x, minimum, maximum = normalize(data = x)

    # cmeans
    # como o numero de entradas é constante n de regras = n de funcoes de
    # pertinencia = numero de centros

    modelo = cmenas(k = n_rules)
    modelo.train(data = x, MAX = 15, tol = 1e-2)
    centros = modelo.C

    # denorm
    centros = denormalize(data = centros, m = minimum, M = maximum)

    def mk_var(name, centros, i):   # de iris_example
        return (name, make_gauss_mfs(1, [centros[0, i], centros[1, i], centros[2, i]]))

    def mk_var(name, centros, i):   # de iris_example
        return (name, make_gauss_mfs(1, [centros[n, i] for n in range(n_rules)]))

    invardefs = [mk_var(name, centros, i) for i, name in
                 enumerate(['SepalLengthCm', 'SepalWidthCm',
                  'PetalLengthCm', 'PetalWidthCm'])]

    outvars = ['Species']

    model = anfis.AnfisNet('iris', invardefs, outvars)
    return model

if __name__ == '__main__':

    # particao de train é usada para k folds

    train_data, idx = data(partition = 'train')
    folds = kfold_data(idx)

    rule_range = range(5)
    fold_eval = np.zeros((len(rule_range), len(folds)))

    for r, n_rules in enumerate(rule_range):
        for f, fold in enumerate(folds):
            fold_train_data = fold['train']
            fold_test_data = fold['test']
            print(n_rules)
            anfis_model = model(fold_train_data, n_rules + 1)
            train_anfis(anfis_model, data = fold_train_data, epochs = 20, show_plots = False)
            _, _, perc_loss = test_anfis(anfis_model, data = fold_test_data, show_plots = False)
            fold_eval[r, f] = perc_loss

    # particao de teste é avaliada com os parametros determinados

    print(fold_eval)

    best_n_rule = np.argmax(np.mean(fold_eval, axis=1)) + 1

    anfis_model = model(train_data, best_n_rule)
    train_anfis(anfis_model, data = train_data, epochs = 20, show_plots = False)
    test_data, _ = data(partition = 'test')
    test_anfis(anfis_model, data = test_data, show_plots = True)
