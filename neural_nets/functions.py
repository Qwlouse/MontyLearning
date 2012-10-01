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

def error_function(x):
    return 0.5*np.sum(x**2)

sigmoid.reverse = sigmoid_reverse

def estimate_gradient(f, theta, epsilon=1e-8):
    grad = np.zeros_like(theta)
    for i, t in enumerate(theta):
        theta[i] = t + epsilon
        e_plus = f(theta)
        theta[i] = t - epsilon
        e_minus = f(theta)
        grad[i] = (e_plus - e_minus) / (2 * epsilon)
    return grad
