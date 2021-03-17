import numpy as np

def normalize(data):
    m = np.min(a = data, axis = 0)
    M = np.max(a = data, axis = 0)
    #xi = (xi - max) / (max - min)
    norm_data = (data - m) / (M - m)
    return norm_data, m, M

def denormalize(data, m, M):
    #xi = xi (max - min) + min
    dnorm_data = data * (M - m) + m
    return dnorm_data