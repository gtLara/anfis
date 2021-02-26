#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: Examples from Jang's paper
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import sys
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs
from experimental import train_anfis, test_anfis

from scipy.io import loadmat

dtype = torch.float

def ex4_model():
    '''
        Example 4 model, from Jang's data; 4 variables with 2 MFs each.
        Predict x(t+6) based on x(t-18), x(t-12), x(t-6), x(t)
        These are the starting MFs values he suggests.
    '''
    invardefs = [
            ('xm18', make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('xm12', make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('xm6',  make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('x',    make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ]
    outvars = ['xp6']
    model = anfis.AnfisNet('Jang\'s example 4', invardefs, outvars)
    return model

def jang_ex4_data(partition = "train"):
    '''
        Read Jang's data for the MG function to be modelled.
    '''

    data = loadmat('mg.mat')['x']
    data = [i[0] for i in data]
    partition_size = int(len(data)/5)

    if partition == "train":
        data = data[:(partition_size*4)]
    if partition == "test":
        data = data[(partition_size*4):]

    num_samples = int(len(data)/5)

    x = torch.zeros((num_samples, 4))
    y = torch.zeros((num_samples, 1))

    i = 0

    for index in range(num_samples):
        x[index] = torch.tensor(data[i:i+4])
        y[index] = data[i+4]
        i += 5

    assert len(x) == len(y), "tamanhos diferentes"

    dl = DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=True)

    return dl

if __name__ == '__main__':
    show_plots = True
    model = ex4_model()
    train_data = jang_ex4_data("train")
    train_anfis(model, train_data, 500, show_plots)
    test_data = jang_ex4_data("test")
    test_anfis(model, test_data, show_plots)
