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

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs
from membership import GaussMembFunc, make_gauss_mfs
from experimental import train_anfis, test_anfis

dtype = torch.float

##### Example 1: Modeling a Two-Input Nonlinear Function #####

def noisy_sin(x):
    return np.sin(x) + 0.1 * np.random.normal(0, 1, 1)

def make_noisy_sin(batch_size=1024):
    '''
        Gera funcao seno
    '''

    x = torch.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y = torch.tensor([noisy_sin(p) for p in x], dtype=dtype).reshape(-1, 1)

    plt.plot(x, y)
    plt.show()

    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)

def ex1_model():
    '''
        Define modelo e parametros para funcoes de pertinencia
    '''
    # invardefs = [
    #         ('x0', make_bell_mfs(3.33333, 2, list(np.linspace(0, 2*np.pi, 3))))
    #         ]
    invardefs = [('x0', 
                  make_gauss_mfs(sigma = 1.2, mu_list = np.linspace(0, 2*np.pi, 3)))
                 ]
    outvars = ['y0']

    anf = anfis.AnfisNet('Aproximacao senoidal', invardefs, outvars)
    return anf

if __name__ == '__main__':
    model = ex1_model()
    train_data = make_noisy_sin(batch_size = 100)
    train_anfis(model, train_data, 20, True)
