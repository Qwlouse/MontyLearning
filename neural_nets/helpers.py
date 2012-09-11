#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np


def add_bias(X):
    X = np.atleast_2d(X)
    return np.hstack((X, np.ones((X.shape[0], 1))))