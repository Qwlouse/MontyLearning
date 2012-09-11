#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function

import numpy as np

def identity(x):
    return x

identity.reverse = identity

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_dx(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_reverse(x):
    return x * (1 - x)

sigmoid.reverse = sigmoid_reverse
