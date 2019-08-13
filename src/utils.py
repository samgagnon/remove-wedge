"""
by Razvan Ciuca, 2017
This script contains generally useful functions

"""


import numpy as np
import torch as t
import scipy.misc
import numpy.fft as fft
import pickle
import os
import torch.nn as nn
from torch.autograd import Variable
#from model_def import Net

def linearizer(x, a, b, c, d, e, on=True):
    '''
    A function applied to each pixel of an output map to help it match the 
    answer map. The parameters should be determined by the NN during learning.
    '''
    if on:
        linear_tensor = a*t.log((b-x)*c)*(1+d*x+e*x**2)
        
        return linear_tensor
    else:
        return x

def constant_bin_height_histogram(m, n_bins):

    # total number of elements
    N = np.prod(m.shape)
    n_per_bin = int(N/n_bins)
    y = np.sort(m.flatten())
    output_intervals = []

    for i in range(0, n_bins):
        k = i*n_per_bin
        output_intervals.append(y[k])
    output_intervals.append(y[-1])

    return [n_per_bin/(output_intervals[i+1]-output_intervals[i])/N for i in range(0, n_bins)], output_intervals


def constant_bin_height_histogram_pytorch(m, n_bins):

    # total number of elements
    N = np.prod(list(m.size()))
    n_per_bin = int(N/n_bins)
    y, indices = t.sort(m.view(-1))
    output_intervals = []

    for i in range(0, n_bins):
        k = i*n_per_bin
        output_intervals.append(y[k])
    output_intervals.append(y[-1])

    return [n_per_bin/(output_intervals[i+1]-output_intervals[i])/N for i in range(0, n_bins)], output_intervals


# This returns a map with the top k pixels set to 1
def get_top_k_pixels(m, k):

    result = np.zeros(m.shape)
    p = np.argpartition(-m.squeeze(), k, axis=None)[:k]

    for index in p:
        i = index % m.shape[0]
        j = int((index - i)/m.shape[1])
        result[j][i] = 1

    return result


# This returns x normalized between 0 and 1
def normalize_map(x):
    """
    Subtracts mean from input and divides by standard deviation
    """
    x = (x - x.mean())/x.std()
    return x


# this returns an array with
def set_border_to_minimum(x, border_size, numpy=False):

    result = x.squeeze()
    minimum = result[border_size:-border_size, border_size:-border_size].min() if numpy else \
        result[border_size:-border_size, border_size:-border_size].min()[0]
    result[:border_size] = minimum
    result[-border_size:] = minimum
    result[:, :border_size] = minimum
    result[:, -border_size:] = minimum

    return result


# this pads the image by pad_size using repeated pixels
def pad_repeating(x, pad_size):
    s2 = x.size(2)
    s3 = x.size(3)
    x1 = t.cat([x[:, :, s2 - pad_size: s2], x, x[:, :, 0:pad_size]], 2)
    x2 = t.cat([x1[:, :, :, s3 - pad_size: s3], x1, x1[:, :, :, 0:pad_size]], 3)
    return x2


# takes the gradient of of the maps, input maps must be torch maps
def get_gradient(input_maps, numpy=False):

    maps = input_maps if not numpy else Variable(t.from_numpy(input_maps))

    maps = pad_repeating(maps, 1)

    # Sobel operators for gradients
    Wx = [[[[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]]]

    Wy = [[[[1, 2, 1],
            [0, 0, 0],
           [-1, -2, -1]]]]

    convx = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
    convy = nn.Conv2d(1, 1, 3, 1, 0, bias=False)

    convx.weight = nn.Parameter(t.Tensor(Wx))
    convy.weight = nn.Parameter(t.Tensor(Wy))

    gradient = t.sqrt(t.pow(convx.forward(maps), 2)+t.pow(convy.forward(maps), 2))

    minimum = gradient.min().item()
    gradient[:, :, :1] = minimum
    gradient[:, :, -1:] = minimum
    gradient[:, :, :, :1] = minimum
    gradient[:, :, :, -1:] = minimum

    return gradient.data.numpy() if numpy else gradient


